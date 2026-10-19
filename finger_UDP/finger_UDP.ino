/*
  R503 Fingerprint Sensor -> Gym Controller UDP bridge (ESP32)

  Replaces the FPM383C sketch. Captures a 192x192 image from the R503,
  expands it to 8-bit greyscale and streams it to the PC with the exact
  protocol fingerprint_gym.py already parses:

      20-byte header : "FPIMG\0\0\0" + width(u16) + height(u16) + size(u32) + timestamp(u32)  (little endian)
      image data     : raw 8-bit pixels in chunks of <=1400 bytes
      trailer        : "FPIMGEND"

  It also listens on UDP port 4210 for the status strings the PC sends
  during enrollment ("Finger 1 - Sample 2", "Success Finger 1") and for
  test commands (PING / STATUS / LED:x / CAPTURE / REBOOT) used by
  test/test_r503_esp32.py.

  ------------------------------------------------------------------
  Wiring (R503, 6 wires):
    red    V       -> 3.3V
    black  GND     -> GND
    yellow TXD     -> GPIO 18  (RX2_PIN)
    green  RXD     -> GPIO 19  (TX2_PIN)
    white  3.3VT   -> 3.3V     (powers the touch ring - required for WAKEUP)
    blue   WAKEUP  -> GPIO 17  (TOUCH_PIN, optional: set to -1 to poll instead)

  The R503 protocol here (ZhianTec / "ZFM") is implemented directly -
  no fingerprint library needed. Sensor image is 192x192 @ 508 dpi and
  UpImage delivers it 4-bit packed (2 px/byte); some clone modules send
  8-bit - both are handled.
  ------------------------------------------------------------------
  Wiring (serial RFID reader, same on the entrance and locker units):
    reader TX      -> GPIO 25  (RFID_RX_PIN)
    reader RX      -> GPIO 26  (RFID_TX_PIN, only if the reader needs commands)
    reader VCC/GND -> 5V / GND (or 3.3V if the module allows it)

  5V readers (RDM6300 and most 125 kHz modules) drive TX at 5V - put a
  divider on it (e.g. 1k from TX to GPIO 25, 2k from GPIO 25 to GND).

  Each card shown is sent to the PC once as "RFID:<uid>" on UDP 8891
  (entrance: check in/out) or 8892 (locker: open the member's locker),
  see rfid_gym.py. Both RDM6300 frames (STX + 10 hex ID + 2 hex checksum +
  ETX) and plain-text readers (ID followed by CR/LF) are understood.
  ------------------------------------------------------------------
*/

#include <WiFi.h>
#include <WiFiUdp.h>

// ---------------------------------------------------------------- role -----
// ROLE 0 = entrance (IP .120, image port 8889), ROLE 1 = locker (IP .121, 8890)
#define ROLE 0

// ---------------------------------------------------------------- network --
const char* ssid     = "HOME";
const char* password = "amin% 1993";

const char* pcIP = "192.168.1.100";       // PC running fingerprint_gym.py
#if ROLE == 0
const int   imagePort = 8889;             // UDP_PORT_ENTRANCE
IPAddress   local_IP(192, 168, 1, 120);   // ENTRANCE_ESP_IP
#else
const int   imagePort = 8890;             // UDP_PORT_LOCKER
IPAddress   local_IP(192, 168, 1, 121);   // LOCKER_ESP_IP
#endif
const int   commandPort = 4210;           // PC sends feedback/commands here

IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress primaryDNS(8, 8, 8, 8);
IPAddress secondaryDNS(8, 8, 4, 4);

WiFiUDP udp;

// ---------------------------------------------------------------- pins -----
#define RX2_PIN     18     // ESP32 receives here  <- sensor TX (yellow)
#define TX2_PIN     19     // ESP32 transmits here -> sensor RX (green)
#define TOUCH_PIN   17     // sensor WAKEUP (blue). -1 = poll GenImg instead
#define TOUCH_ACTIVE LOW  // WAKEUP level when a finger is present
#define LED_PIN      2     // onboard LED

// ---------------------------------------------------------------- rfid -----
// Serial RFID reader on the second UART. Readers repeat the ID for as long as
// the card stays in the field; one presentation must count as one read, so
// the same card is ignored until it has been away for RFID_REPEAT_MS.
// The PC tells the two readers apart by port: entrance cards check in/out,
// locker cards open the member's locker.
#define RFID_ENABLED    1      // 0 = no reader fitted on this unit
#define RFID_RX_PIN     25     // ESP32 receives here <- reader TX
#define RFID_TX_PIN     26     // ESP32 transmits here -> reader RX
#define RFID_BAUD       9600   // RDM6300 and most UART readers
#define RFID_REPEAT_MS  3000
#define RFID_IDLE_MS    60     // flush an unterminated ID after this much silence
#if ROLE == 0
const int rfidPort = 8891;     // UDP_PORT_RFID_ENTRANCE on the PC
#else
const int rfidPort = 8892;     // UDP_PORT_RFID_LOCKER on the PC
#endif

// Blank-frame gate: a real fingerprint has contrast; an empty/backlit window is
// nearly uniform. Frames with a pixel std-dev below this are not sent. The
// serial log prints the measured std of every capture - if genuine fingers get
// skipped, lower this; if empty frames still go through, raise it.
#define IMG_MIN_STD  18.0f

// Duplicate gate: the previous finger leaves a greasy residue image on the
// optical window, and that ghost is a GENUINE print image - GenImg and even
// Img2Tz accept it. But residue is static: consecutive captures of it are
// nearly pixel-identical, while a real re-press always lands shifted. Frames
// whose mean absolute difference vs the last SENT frame is below this are
// dropped. Genuine re-presses measure 40+; residue measures < 5.
#define DUP_MAD  10.0f

// ---------------------------------------------------------------- image ----
#define IMAGE_WIDTH   192
#define IMAGE_HEIGHT  192
#define IMAGE_BYTES   (IMAGE_WIDTH * IMAGE_HEIGHT)        // 36864 (8-bit, worst case)
#define PACKED_BYTES  (IMAGE_BYTES / 2)                   // 18432 (4-bit)
uint8_t* imageBuffer = NULL;

// Actual geometry of the last capture. R503 batches differ: datasheet units are
// 192x192, but newer/150-template batches deliver 160x160. UpImage may also be
// 4-bit packed or 8-bit. All four combinations are detected per capture and the
// real size goes into the UDP header (the PC parses it dynamically).
uint16_t imgWidth  = IMAGE_WIDTH;
uint16_t imgHeight = IMAGE_HEIGHT;
uint32_t imgBytes  = IMAGE_BYTES;

// Copy of the last frame actually sent, for the duplicate (residue) gate.
uint8_t* lastSentFrame = NULL;
uint32_t lastSentBytes = 0;

// ============================================================================
// Minimal R503 (ZFM protocol) driver
// ============================================================================

#define ZFM_START     0xEF01
#define PID_COMMAND   0x01
#define PID_DATA      0x02
#define PID_ACK       0x07
#define PID_DATA_END  0x08

// LED control (AuraLedConfig 0x35)
#define LED_CTRL_BREATH  0x01
#define LED_CTRL_FLASH   0x02
#define LED_CTRL_ON      0x03
#define LED_CTRL_OFF     0x04
#define LED_RED          0x01
#define LED_BLUE         0x02
#define LED_PURPLE       0x03

class R503 {
public:
  HardwareSerial& port;
  uint32_t address  = 0xFFFFFFFF;
  uint32_t passwd   = 0x00000000;
  long     baudUsed = 0;
  // ReadSysPara results
  uint16_t librarySize = 0, packetSize = 0, securityLevel = 0;

  R503(HardwareSerial& p) : port(p) {}

  void writePacket(uint8_t pid, const uint8_t* payload, uint16_t plen) {
    uint16_t len = plen + 2;
    uint8_t hdr[9] = {
      0xEF, 0x01,
      (uint8_t)(address >> 24), (uint8_t)(address >> 16),
      (uint8_t)(address >> 8),  (uint8_t)(address),
      pid, (uint8_t)(len >> 8), (uint8_t)(len)
    };
    uint16_t sum = pid + (uint8_t)(len >> 8) + (uint8_t)len;
    port.write(hdr, 9);
    for (uint16_t i = 0; i < plen; i++) { port.write(payload[i]); sum += payload[i]; }
    port.write((uint8_t)(sum >> 8));
    port.write((uint8_t)sum);
  }

  // Read one frame. Returns payload length (checksum stripped) or <0 on error.
  int readFrame(uint8_t& pid, uint8_t* payload, uint16_t maxlen, uint32_t timeoutMs) {
    uint32_t deadline = millis() + timeoutMs;
    // hunt for 0xEF 0x01
    int state = 0;
    while (millis() < deadline) {
      if (!port.available()) { delay(1); continue; }
      uint8_t b = port.read();
      if (state == 0)      { state = (b == 0xEF) ? 1 : 0; }
      else if (state == 1) { if (b == 0x01) break; state = (b == 0xEF) ? 1 : 0; }
    }
    if (millis() >= deadline) return -1;

    uint8_t head[7];   // addr(4) + pid + len(2)
    if (!readBytes(head, 7, deadline)) return -2;
    pid = head[4];
    uint16_t len = ((uint16_t)head[5] << 8) | head[6];
    if (len < 2 || (uint16_t)(len - 2) > maxlen) return -3;

    uint16_t plen = len - 2;
    if (!readBytes(payload, plen, deadline)) return -4;
    uint8_t ck[2];
    if (!readBytes(ck, 2, deadline)) return -5;

    uint16_t sum = pid + head[5] + head[6];
    for (uint16_t i = 0; i < plen; i++) sum += payload[i];
    if (sum != (((uint16_t)ck[0] << 8) | ck[1])) return -6;
    return plen;
  }

  bool readBytes(uint8_t* dst, uint16_t n, uint32_t deadline) {
    // Byte-by-byte so it compiles on every arduino-esp32 core version
    // (the read(buffer, size) overload only exists in core 2.x+).
    uint16_t got = 0;
    while (got < n) {
      if (millis() >= deadline) return false;
      if (port.available() <= 0) { delay(1); continue; }
      while (got < n && port.available() > 0) {
        int c = port.read();
        if (c < 0) break;
        dst[got++] = (uint8_t)c;
      }
    }
    return true;
  }

  // Send a command, wait for the ACK. Returns confirmation code or <0.
  int command(const uint8_t* payload, uint16_t plen,
              uint8_t* params = NULL, uint16_t maxParams = 0, uint32_t timeoutMs = 1000) {
    while (port.available()) port.read();          // drop stale bytes
    writePacket(PID_COMMAND, payload, plen);
    uint8_t pid, buf[64];
    int n = readFrame(pid, buf, sizeof(buf), timeoutMs);
    if (n < 1 || pid != PID_ACK) return -100 + n;
    if (params && n > 1) memcpy(params, buf + 1, min((int)maxParams, n - 1));
    return buf[0];
  }

  bool verifyPassword() {
    uint8_t p[5] = { 0x13,
      (uint8_t)(passwd >> 24), (uint8_t)(passwd >> 16),
      (uint8_t)(passwd >> 8),  (uint8_t)passwd };
    return command(p, 5) == 0x00;
  }

  bool readSysPara() {
    uint8_t p[1] = { 0x0F }, out[16];
    if (command(p, 1, out, 16) != 0x00) return false;
    librarySize   = ((uint16_t)out[4] << 8) | out[5];
    securityLevel = ((uint16_t)out[6] << 8) | out[7];
    uint16_t pcode = ((uint16_t)out[12] << 8) | out[13];
    packetSize = 32 << pcode;                     // 0:32 1:64 2:128 3:256
    return true;
  }

  int templateCount() {
    uint8_t p[1] = { 0x1D }, out[2];
    if (command(p, 1, out, 2) != 0x00) return -1;
    return ((int)out[0] << 8) | out[1];
  }

  // GenImg: 0x00 captured, 0x02 no finger, other = error
  int genImg() { uint8_t p[1] = { 0x01 }; return command(p, 1, NULL, 0, 700); }

  // Img2Tz: the sensor's own feature extraction on the captured image.
  // 0x00 = real finger with usable ridges. 0x06 (too messy), 0x07 (too few
  // minutiae), 0x15 (no valid image) = latent print / light / empty window.
  // GenImg alone is NOT a reliable finger detector on this optical sensor.
  int img2Tz(uint8_t slot = 1) {
    uint8_t p[2] = { 0x02, slot };
    return command(p, 2, NULL, 0, 1500);
  }

  bool setBaudRegister(uint8_t n) {               // baud = 9600 * n
    uint8_t p[3] = { 0x0E, 4, n };
    return command(p, 3) == 0x00;
  }

  bool led(uint8_t ctrl, uint8_t color, uint8_t speed = 128, uint8_t times = 0) {
    uint8_t p[5] = { 0x35, ctrl, speed, color, times };
    return command(p, 5) == 0x00;
  }

  // UpImage: stream the frame into dst (dstCap bytes). Returns bytes received, <0 on error.
  long upImage(uint8_t* dst, uint32_t dstCap) {
    uint8_t p[1] = { 0x0A };
    int code = command(p, 1, NULL, 0, 1500);
    if (code != 0x00) return -code;

    uint32_t received = 0;
    uint8_t pid, chunk[300];
    while (true) {
      int n = readFrame(pid, chunk, sizeof(chunk), 3000);
      if (n < 0) return -200 + n;                 // stream broke mid-transfer
      if (pid != PID_DATA && pid != PID_DATA_END) continue;
      if (received + n > dstCap) return -300;     // larger than any known format
      memcpy(dst + received, chunk, n);
      received += n;
      if (pid == PID_DATA_END) break;
    }
    return (long)received;
  }
};

R503 sensor(Serial2);

// ============================================================================
// Setup
// ============================================================================

bool sensorOk = false;

// last host that sent us a command (test replies go back there)
IPAddress lastCmdIP;
uint16_t  lastCmdPort = 0;
bool      forceCapture = false;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  if (TOUCH_PIN >= 0) pinMode(TOUCH_PIN, INPUT);
  Serial.println("\nR503 -> Gym Controller UDP bridge");

  imageBuffer = (uint8_t*)ps_malloc(IMAGE_BYTES);
  if (imageBuffer == NULL) imageBuffer = (uint8_t*)malloc(IMAGE_BYTES);
  if (imageBuffer == NULL) {
    Serial.println("FATAL: cannot allocate 36864-byte image buffer");
    while (1) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); delay(100); }
  }

  // Second buffer for the residue/duplicate gate; without it the gate is off.
  lastSentFrame = (uint8_t*)ps_malloc(IMAGE_BYTES);
  if (lastSentFrame == NULL) lastSentFrame = (uint8_t*)malloc(IMAGE_BYTES);
  if (lastSentFrame == NULL)
    Serial.println("WARNING: no RAM for duplicate detection - residue prints will not be filtered");

  // ---- WiFi ----
  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS))
    Serial.println("STA config failed");
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
  digitalWrite(LED_PIN, LOW);
  Serial.printf("\nWiFi OK  ip=%s  ->  images to %s:%d\n",
                WiFi.localIP().toString().c_str(), pcIP, imagePort);
  udp.begin(commandPort);                          // listen for PC feedback / test commands

#if RFID_ENABLED
  // Started before the fingerprint sensor so cards still work if the R503 is dead.
  // The buffer holds reads that arrive while an image upload blocks the loop.
  Serial1.setRxBufferSize(512);
  Serial1.begin(RFID_BAUD, SERIAL_8N1, RFID_RX_PIN, RFID_TX_PIN);
  Serial.printf("RFID reader on GPIO %d @ %d baud -> cards to %s:%d\n",
                RFID_RX_PIN, RFID_BAUD, pcIP, rfidPort);
#endif

  // ---- sensor: find its baud rate, prefer 115200 for the image upload ----
  Serial2.setRxBufferSize(4096);
  Serial2.begin(57600, SERIAL_8N1, RX2_PIN, TX2_PIN);
  delay(150);

  const long bauds[] = { 57600, 115200, 9600, 19200, 38400 };
  for (long b : bauds) {
    Serial2.updateBaudRate(b);
    delay(60);
    while (Serial2.available()) Serial2.read();
    if (sensor.verifyPassword()) { sensor.baudUsed = b; break; }
  }

  if (sensor.baudUsed == 0) {
    Serial.println("FATAL: R503 not responding (check TX/RX crossover and 3.3V supply)");
    sendStatus("R503 BOOT FAIL sensor-not-found");
  } else {
    if (sensor.baudUsed != 115200 && sensor.setBaudRegister(12)) {   // 9600*12
      delay(100);
      Serial2.updateBaudRate(115200);
      delay(60);
      if (sensor.verifyPassword()) sensor.baudUsed = 115200;
      else { Serial2.updateBaudRate(sensor.baudUsed); }              // revert
    }
    sensorOk = sensor.readSysPara();
    Serial.printf("R503 OK  baud=%ld  library=%u  security=%u  packet=%u  templates=%d\n",
                  sensor.baudUsed, sensor.librarySize, sensor.securityLevel,
                  sensor.packetSize, sensor.templateCount());
    runSelfTest();
  }
}

// Boot self test: exercises every subsystem once and reports to Serial + PC.
void runSelfTest() {
  bool wifi   = (WiFi.status() == WL_CONNECTED);
  bool sysOk  = sensor.readSysPara();
  int  tmpl   = sensor.templateCount();
  bool ledOk  = sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);
  int  gen    = sensor.genImg();                   // 0x02 (no finger) proves comms
  bool genOk  = (gen == 0x00 || gen == 0x02);

  Serial.println("---- self test ----");
  Serial.printf("  wifi      : %s\n", wifi  ? "OK" : "FAIL");
  Serial.printf("  syspara   : %s\n", sysOk ? "OK" : "FAIL");
  Serial.printf("  templates : %d\n", tmpl);
  Serial.printf("  aura led  : %s\n", ledOk ? "OK" : "FAIL");
  Serial.printf("  genimg    : %s (code %d)\n", genOk ? "OK" : "FAIL", gen);
  Serial.println("-------------------");

  char msg[120];
  snprintf(msg, sizeof(msg), "R503 BOOT %s role=%s baud=%ld lib=%u led=%d genimg=%d",
           (wifi && sysOk && genOk) ? "OK" : "FAIL",
           ROLE == 0 ? "entrance" : "locker",
           sensor.baudUsed, sensor.librarySize, ledOk, gen);
  sendStatus(msg);
  sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);   // idle: blue breathing
}

void sendStatus(const char* msg) {                 // to the PC monitor port
  udp.beginPacket(pcIP, 4211);
  udp.write((const uint8_t*)msg, strlen(msg));
  udp.endPacket();
  Serial.println(msg);
}

// ============================================================================
// Main loop
// ============================================================================

void loop() {
  handleUdpCommands();
  handleRfid();

  if (!sensorOk) { delay(30); return; }

  bool wantCapture = forceCapture;
  if (!wantCapture) {
    if (TOUCH_PIN >= 0) wantCapture = (digitalRead(TOUCH_PIN) == TOUCH_ACTIVE);
    else                wantCapture = true;        // no touch pin: poll GenImg
  }
  if (!wantCapture) { delay(30); return; }

  // Capture: wait for a *usable* finger. GenImg returns 0x00 for latent
  // prints, ambient light or an empty window, so every capture must also pass
  // the sensor's own feature extraction (Img2Tz) before it is worth uploading.
  uint32_t window = forceCapture ? 10000 : 1200;
  uint32_t start = millis();
  bool usable = false;
  while (millis() - start < window) {
    int code = sensor.genImg();
    if (code == 0x00) {
      int tz = sensor.img2Tz(1);
      if (tz == 0x00) { usable = true; break; }
      Serial.printf("Capture rejected by sensor (Img2Tz code %d) - not a finger\n", tz);
      delay(200);
    } else if (code != 0x02) {
      delay(100);                                  // real error: brief backoff
    } else {
      delay(60);                                   // no finger yet
    }
    handleUdpCommands();
    handleRfid();
  }
  if (!usable) {
    if (forceCapture) { replyToTester("CAPTURE:TIMEOUT"); forceCapture = false; }
    delay(TOUCH_PIN >= 0 ? 30 : 150);
    return;
  }

  // Upload the frame from the sensor
  sensor.led(LED_CTRL_ON, LED_PURPLE);
  Serial.println("Finger captured - uploading image from sensor...");
  uint32_t t0 = millis();
  long got = sensor.upImage(imageBuffer, IMAGE_BYTES);
  uint32_t uartMs = millis() - t0;

  // Identify the frame by its size. Known R503 outputs:
  //   36864 = 192x192 8-bit    18432 = 192x192 4-bit packed
  //   25600 = 160x160 8-bit    12800 = 160x160 4-bit packed
  struct Fmt { uint32_t bytes; uint16_t w, h; bool packed; };
  static const Fmt formats[] = {
    { 36864, 192, 192, false }, { 18432, 192, 192, true  },
    { 25600, 160, 160, false }, { 12800, 160, 160, true  },
  };
  const Fmt* fmt = NULL;
  for (const auto& f : formats) {
    if (got == (long)f.bytes) { fmt = &f; break; }
  }

  if (fmt == NULL) {
    Serial.printf("Image upload failed: %ld bytes is no known frame size\n", got);
    sensor.led(LED_CTRL_FLASH, LED_RED, 40, 4);
    if (forceCapture) { replyToTester("CAPTURE:UPLOAD_FAIL"); forceCapture = false; }
    delay(800);
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
    return;
  }

  imgWidth  = fmt->w;
  imgHeight = fmt->h;
  imgBytes  = (uint32_t)fmt->w * fmt->h;
  if (fmt->packed) {
    // Expand 4-bit to 8-bit in place, back to front (2*i >= i, so no clobbering).
    // Nibble replication (n -> n*17) maps the 16 levels over the full 0-255
    // range instead of 0-240. Note: 16 levels is all UpImage ever delivers -
    // the near black-and-white look comes from the sensor's internal
    // enhancement, not from this decode.
    for (long i = got - 1; i >= 0; i--) {
      uint8_t b = imageBuffer[i];
      uint8_t hi = b >> 4, lo = b & 0x0F;
      imageBuffer[2 * i]     = (uint8_t)(hi << 4 | hi);   // high nibble = first pixel
      imageBuffer[2 * i + 1] = (uint8_t)(lo << 4 | lo);
    }
  }
  // Backstop gate: reject visually blank frames that slipped past Img2Tz.
  uint64_t sum = 0, sumSq = 0;
  for (uint32_t i = 0; i < imgBytes; i++) {
    uint32_t v = imageBuffer[i];
    sum += v;
    sumSq += (uint64_t)v * v;
  }
  float mean = (float)sum / imgBytes;
  float var  = (float)sumSq / imgBytes - mean * mean;
  float stddev = (var > 0) ? sqrtf(var) : 0.0f;

  Serial.printf("Image received in %lu ms (%ux%u, %s, mean %.0f, std %.1f)\n",
                (unsigned long)uartMs, imgWidth, imgHeight,
                fmt->packed ? "4-bit packed" : "8-bit", mean, stddev);

  if (stddev < IMG_MIN_STD) {
    Serial.printf("Skipped blank frame (std %.1f < %.1f)\n", stddev, IMG_MIN_STD);
    if (forceCapture) { replyToTester("CAPTURE:BLANK"); forceCapture = false; }
    delay(400);
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
    return;
  }

  // Residue gate: a ghost image left on the glass by the previous finger is a
  // real print, so Img2Tz passes it - but it is static, so it is nearly
  // identical to the frame we already sent. A genuine re-press always shifts.
  if (lastSentFrame != NULL && lastSentBytes == imgBytes) {
    uint64_t diff = 0;
    for (uint32_t i = 0; i < imgBytes; i++) {
      int d = (int)imageBuffer[i] - (int)lastSentFrame[i];
      diff += (d >= 0) ? d : -d;
    }
    float mad = (float)diff / imgBytes;
    if (mad < DUP_MAD) {
      Serial.printf("Skipped residue print (MAD %.1f < %.1f vs last sent) - clean the window\n",
                    mad, DUP_MAD);
      if (forceCapture) { replyToTester("CAPTURE:DUPLICATE"); forceCapture = false; }
      delay(700);                                  // residue does not go away quickly
      sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
      return;
    }
    Serial.printf("Duplicate check: MAD %.1f (new print)\n", mad);
  }

  t0 = millis();
  sendImageViaUDP();
  if (lastSentFrame != NULL) {                     // remember what we sent
    memcpy(lastSentFrame, imageBuffer, imgBytes);
    lastSentBytes = imgBytes;
  }
  if (forceCapture) {
    char msg[64];
    snprintf(msg, sizeof(msg), "CAPTURE:SENT uart=%lums udp=%lums",
             (unsigned long)uartMs, (unsigned long)(millis() - t0));
    replyToTester(msg);
    forceCapture = false;
  }

  sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);
  delay(1500);                                     // cooldown, matches old sketch
  // wait for the finger to lift so one press = one image
  if (TOUCH_PIN >= 0) {
    uint32_t lift = millis();
    while (digitalRead(TOUCH_PIN) == TOUCH_ACTIVE && millis() - lift < 5000) { handleRfid(); delay(20); }
  }
  sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
}

// ============================================================================
// UDP image transmit - wire-identical to the old FPM383C sketch
// ============================================================================

void sendImageViaUDP() {
  struct __attribute__((packed)) ImageHeader {
    char     marker[8];                            // "FPIMG\0\0\0"
    uint16_t width, height;
    uint32_t size, timestamp;
  } header;

  memcpy(header.marker, "FPIMG\0\0\0", 8);
  header.width = imgWidth;                         // detected geometry, not assumed
  header.height = imgHeight;
  header.size = imgBytes;
  header.timestamp = millis();

  udp.beginPacket(pcIP, imagePort);
  udp.write((uint8_t*)&header, sizeof(header));
  udp.endPacket();
  delay(50);

  Serial.printf("Sending %lu bytes (%ux%u) to %s:%d\n",
                (unsigned long)imgBytes, imgWidth, imgHeight, pcIP, imagePort);

  const int chunkSize = 1400;                      // < PC's recvfrom(2048)
  uint32_t sent = 0;
  while (sent < imgBytes) {
    int n = min((uint32_t)chunkSize, imgBytes - sent);
    udp.beginPacket(pcIP, imagePort);
    udp.write(imageBuffer + sent, n);
    udp.endPacket();
    sent += n;
    delay(10);                                     // pacing so the PC socket keeps up
  }

  delay(50);
  udp.beginPacket(pcIP, imagePort);
  udp.write((const uint8_t*)"FPIMGEND", 8);
  udp.endPacket();
  Serial.println("Image sent");
}

// ============================================================================
// RFID reader (Serial1) -> PC
// ============================================================================

char     rfidBuf[24];
uint8_t  rfidLen = 0;
bool     rfidFramed = false;          // inside an RDM6300 STX..ETX frame
uint32_t rfidLastByteMs = 0;
char     lastCard[24] = "";
uint32_t lastCardMs = 0;
uint32_t rfidLedUntil = 0;            // when the ring goes back to idle (0 = not lit)

static uint8_t hexNibble(char c) { return (c <= '9') ? c - '0' : c - 'A' + 10; }

// Card feedback on the R503 ring, reset to idle breathing by handleRfid().
void rfidLed(uint8_t ctrl, uint8_t color, uint32_t holdMs) {
  if (!sensorOk) return;                           // the ring belongs to the R503
  sensor.led(ctrl, color, 40, 0);
  rfidLedUntil = millis() + holdMs;
  if (rfidLedUntil == 0) rfidLedUntil = 1;
}

void handleRfid() {
#if RFID_ENABLED
  while (Serial1.available() > 0) {
    int c = Serial1.read();
    if (c < 0) break;
    rfidLastByteMs = millis();
    if (c == 0x02) {                               // STX: start of an RDM6300 frame
      rfidFramed = true;
      rfidLen = 0;
    } else if (c == 0x03 || c == '\r' || c == '\n') {   // ETX or end of line
      rfidFlush();
    } else if (isalnum(c)) {
      if (rfidLen < sizeof(rfidBuf) - 1) rfidBuf[rfidLen++] = (char)toupper(c);
      else rfidLen = 0;                            // runaway line: drop it
    }
  }
  // Readers that send the ID with no terminator: take it once the line goes quiet
  if (rfidLen > 0 && millis() - rfidLastByteMs > RFID_IDLE_MS) rfidFlush();

  if (rfidLedUntil != 0 && (int32_t)(millis() - rfidLedUntil) >= 0) {
    rfidLedUntil = 0;
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
  }
#endif
}

void rfidFlush() {
  bool framed = rfidFramed;
  uint8_t len = rfidLen;
  rfidBuf[len] = 0;
  rfidFramed = false;
  rfidLen = 0;
  if (len == 0) return;

  // RDM6300: 10 hex ID chars + 2 hex chars of XOR checksum over the 5 ID bytes
  if (framed && len == 12) {
    uint8_t x = 0;
    for (int i = 0; i < 10; i += 2) x ^= (hexNibble(rfidBuf[i]) << 4) | hexNibble(rfidBuf[i + 1]);
    uint8_t ck = (hexNibble(rfidBuf[10]) << 4) | hexNibble(rfidBuf[11]);
    if (x != ck) { Serial.printf("RFID checksum error (%s) - ignored\n", rfidBuf); return; }
    rfidBuf[10] = 0;
    len = 10;
  }
  if (len < 4) return;                             // line noise, not an ID

  uint32_t now = millis();
  bool repeat = (strcmp(rfidBuf, lastCard) == 0 && now - lastCardMs < RFID_REPEAT_MS);
  lastCardMs = now;                                // a held card keeps extending the window
  if (repeat) return;
  strcpy(lastCard, rfidBuf);

  char msg[40];
  snprintf(msg, sizeof(msg), "RFID:%s", rfidBuf);
  udp.beginPacket(pcIP, rfidPort);
  udp.write((const uint8_t*)msg, strlen(msg));
  udp.endPacket();
  Serial.printf("Card %s -> %s:%d\n", rfidBuf, pcIP, rfidPort);

  rfidLed(LED_CTRL_ON, LED_PURPLE, 2000);          // "read"; the PC's verdict replaces it
}

// ============================================================================
// Command / feedback channel (port 4210)
// ============================================================================

void replyToTester(const char* msg) {
  if (lastCmdPort == 0) return;
  udp.beginPacket(lastCmdIP, lastCmdPort);
  udp.write((const uint8_t*)msg, strlen(msg));
  udp.endPacket();
  Serial.printf("-> %s\n", msg);
}

void handleUdpCommands() {
  int size = udp.parsePacket();
  if (size <= 0) return;

  char buf[161];
  int n = udp.read(buf, sizeof(buf) - 1);
  if (n <= 0) return;
  buf[n] = 0;

  // ---- RFID verdict from rfid_gym.py (kept apart from the tester's reply address) ----
  if (strncasecmp(buf, "RFID:", 5) == 0) {
    const char* v = buf + 5;
    Serial.printf("RFID verdict: %s\n", v);
    if      (strncasecmp(v, "OK", 2) == 0)       rfidLed(LED_CTRL_ON, LED_BLUE, 1500);
    else if (strncasecmp(v, "ENROLLED", 8) == 0) rfidLed(LED_CTRL_FLASH, LED_BLUE, 2000);
    else                                          rfidLed(LED_CTRL_FLASH, LED_RED, 2000);  // UNKNOWN / ENROLL_FAIL
    return;
  }

  lastCmdIP = udp.remoteIP();
  lastCmdPort = udp.remotePort();
  Serial.printf("UDP cmd from %s:%u : %s\n", lastCmdIP.toString().c_str(), lastCmdPort, buf);

  // ---- explicit test commands ----
  if (strncasecmp(buf, "PING", 4) == 0) {
    char msg[96];
    snprintf(msg, sizeof(msg), "PONG role=%s ip=%s up=%lus sensor=%s",
             ROLE == 0 ? "entrance" : "locker",
             WiFi.localIP().toString().c_str(),
             (unsigned long)(millis() / 1000), sensorOk ? "ok" : "fail");
    replyToTester(msg);
    return;
  }
  if (strncasecmp(buf, "STATUS", 6) == 0) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "STATUS sensor=%s baud=%ld lib=%u sec=%u packet=%u templates=%d rssi=%d touch=%d",
             sensorOk ? "ok" : "fail", sensor.baudUsed, sensor.librarySize,
             sensor.securityLevel, sensor.packetSize, sensor.templateCount(), WiFi.RSSI(),
             (TOUCH_PIN >= 0) ? digitalRead(TOUCH_PIN) : -1);
    replyToTester(msg);
    return;
  }
  if (strncasecmp(buf, "LED:", 4) == 0) {
    const char* c = buf + 4;
    bool ok = false;
    if      (strncasecmp(c, "RED", 3) == 0)    ok = sensor.led(LED_CTRL_ON, LED_RED);
    else if (strncasecmp(c, "BLUE", 4) == 0)   ok = sensor.led(LED_CTRL_ON, LED_BLUE);
    else if (strncasecmp(c, "PURPLE", 6) == 0) ok = sensor.led(LED_CTRL_ON, LED_PURPLE);
    else if (strncasecmp(c, "BREATH", 6) == 0) ok = sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
    else if (strncasecmp(c, "OFF", 3) == 0)    ok = sensor.led(LED_CTRL_OFF, LED_BLUE);
    replyToTester(ok ? "LED:OK" : "LED:FAIL");
    return;
  }
  if (strncasecmp(buf, "CAPTURE", 7) == 0) {       // test-driven capture window
    forceCapture = true;
    replyToTester("CAPTURE:WAIT");
    return;
  }
  if (strncasecmp(buf, "REBOOT", 6) == 0) {
    replyToTester("REBOOTING");
    delay(100);
    ESP.restart();
  }

  // ---- enrollment feedback from fingerprint_gym.py ----
  if (strstr(buf, "Sample") != NULL) {             // "Finger 1 - Sample 2" -> prompt
    sensor.led(LED_CTRL_FLASH, LED_BLUE, 80, 0);
  } else if (strstr(buf, "Success") != NULL) {     // "Success Finger 1"
    sensor.led(LED_CTRL_ON, LED_BLUE);
    delay(1200);
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
  }
}
