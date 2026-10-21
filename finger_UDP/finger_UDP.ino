/*
  R503 Fingerprint Sensor -> Gym Controller UDP bridge (ESP32)

  Replaces the FPM383C sketch. Captures the R503's grayscale image (uploaded
  before Img2Tz, which would overwrite it with a ridge skeleton), expands it
  to 8-bit greyscale and streams it to the PC with the exact protocol
  fingerprint_gym.py already parses:

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
#include <ArduinoOTA.h>

// ---------------------------------------------------------------- role -----
// ROLE 0 = entrance (IP .120, image port 8889), ROLE 1 = locker (IP .121, 8890)
// finger_UDP/ota_upload.py builds each unit with -DROLE=<n>; this default is
// what the Arduino IDE builds.
#ifndef ROLE
#define ROLE 0
#endif

// ---------------------------------------------------------------- OTA ------
// Once this sketch is on a unit (one USB flash), later versions go over WiFi:
// the Arduino IDE lists the units as network ports ("gym-fp-entrance",
// "gym-fp-locker"), or run finger_UDP/ota_upload.py for both. Uploads need
// this password (ota_upload.py reads it from here).
const char* otaPassword = "gym-ota-2026";

// ---------------------------------------------------------------- network --
const char* ssid     = "Autoro";
const char* password = "#Autoro@2027";

const char* pcIP = "192.168.1.2";       // PC running fingerprint_gym.py
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
#define RFID_ENABLED    0      // 0 = no reader fitted on this unit
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
uint32_t lastSentStamp = 0;      // header timestamp of that frame: the PC names the press by it
bool     lastSentPacked = false; // the sensor delivered it 4-bit packed (CONFIRM sends it back so)

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
    if (code != 0x00) return code > 0 ? -code : code - 1000;   // refused / no ACK
    return readDataStream(dst, dstCap);
  }

  // UpChar: upload the feature file in CharBuffer `slot` - 384 bytes after
  // Img2Tz, 768 after RegModel (datasheet). Same data stream as UpImage.
  long upChar(uint8_t slot, uint8_t* dst, uint32_t dstCap) {
    uint8_t p[2] = { 0x08, slot };
    int code = command(p, 2, NULL, 0, 1500);
    if (code != 0x00) return code > 0 ? -code : code - 1000;
    return readDataStream(dst, dstCap);
  }

  // ReadInfPage: the 512-byte information page, same data stream as UpImage.
  long readInfPage(uint8_t* dst, uint32_t dstCap) {
    uint8_t p[1] = { 0x16 };
    int code = command(p, 1, NULL, 0, 1500);
    if (code != 0x00) return code > 0 ? -code : code - 1000;
    return readDataStream(dst, dstCap);
  }

  // Match: 1:1 compare CharBuffer1 against CharBuffer2. 0x00 match, 0x08 no match.
  int matchBuffers(uint16_t* score) {
    uint8_t p[1] = { 0x03 }, out[2] = { 0, 0 };
    int code = command(p, 1, out, 2, 2000);
    if (score) *score = ((uint16_t)out[0] << 8) | out[1];
    return code;
  }

  // RegModel: merge CharBuffer1 + CharBuffer2 into one template.
  int regModel() { uint8_t p[1] = { 0x05 }; return command(p, 1, NULL, 0, 2000); }

  // DownImage / DownChar: the host sends the data after the ACK.
  int downImage(const uint8_t* src, uint32_t len) {
    uint8_t p[1] = { 0x0B };
    int code = command(p, 1, NULL, 0, 1500);
    if (code != 0x00) return code;
    return writeDataStream(src, len);
  }

  int downChar(uint8_t slot, const uint8_t* src, uint32_t len) {
    uint8_t p[2] = { 0x09, slot };
    int code = command(p, 2, NULL, 0, 1500);
    if (code != 0x00) return code;
    return writeDataStream(src, len);
  }

  // Send src as data packets of packetSize bytes, the last one flagged end.
  // Some firmwares ACK the stream and some don't; waiting briefly here keeps a
  // late ACK from being read as the reply to the next command.
  int writeDataStream(const uint8_t* src, uint32_t len) {
    uint16_t ps = packetSize ? packetSize : 128;
    for (uint32_t off = 0; off < len; off += ps) {
      uint16_t n = (uint16_t)min((uint32_t)ps, len - off);
      writePacket(off + n >= len ? PID_DATA_END : PID_DATA, src + off, n);
    }
    port.flush();
    uint8_t pid, buf[16];
    int n = readFrame(pid, buf, sizeof(buf), 200);
    if (n >= 1 && pid == PID_ACK) return buf[0];
    return 0;
  }

  // Collect the data packets that follow an UpImage/UpChar ACK.
  long readDataStream(uint8_t* dst, uint32_t dstCap) {
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
bool      pairRequested = false;                   // PAIR: template research capture
bool      probeRequested = false;                  // PROBE: template export check
bool      img2TmplRequested = false;               // IMG2TMPL: template from an uploaded image
bool      img2TmplSkeleton = true;                 // IMG2TMPL FAST: skip the skeleton upload
bool      confirmRequested = false;                // CONFIRM <stamp>: double-check of a sent press
uint32_t  confirmStamp = 0;
bool      matchTmplRequested = false;              // MATCHTMPL: sensor Match of an uploaded template
bool      liveTmplRequested = false;               // TMPL: template from a live press
bool      liveTmplRaw = false;                     // TMPL RAW: one capture, via DownImage

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

  ArduinoOTA.setHostname(ROLE == 0 ? "gym-fp-entrance" : "gym-fp-locker");
  ArduinoOTA.setPassword(otaPassword);
  ArduinoOTA.onStart([]() { Serial.println("OTA update starting"); });
  ArduinoOTA.onEnd([]() { Serial.println("OTA update done - rebooting"); });
  ArduinoOTA.onError([](ota_error_t e) { Serial.printf("OTA update failed (error %u)\n", (unsigned)e); });
  ArduinoOTA.begin();
  Serial.printf("OTA ready: %s\n", ROLE == 0 ? "gym-fp-entrance" : "gym-fp-locker");

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
  ArduinoOTA.handle();
  handleUdpCommands();
  handleRfid();

  if (!sensorOk) { delay(30); return; }

  if (confirmRequested) { confirmRequested = false; runConfirm(confirmStamp); return; }
  if (pairRequested) { pairRequested = false; runPairCapture(); return; }
  if (probeRequested) { probeRequested = false; runProbe(); return; }
  if (img2TmplRequested) { img2TmplRequested = false; runImg2Tmpl(); return; }
  if (matchTmplRequested) { matchTmplRequested = false; runMatchTmpl(); return; }
  if (liveTmplRequested) { liveTmplRequested = false; runLiveTemplate(liveTmplRaw); return; }

  bool wantCapture = forceCapture;
  if (!wantCapture) {
    if (TOUCH_PIN >= 0) wantCapture = (digitalRead(TOUCH_PIN) == TOUCH_ACTIVE);
    else                wantCapture = true;        // no touch pin: poll GenImg
  }
  if (!wantCapture) { delay(30); return; }

  // Capture: wait for a *usable* finger. GenImg returns 0x00 for latent
  // prints, ambient light or an empty window, so every capture must also pass
  // the sensor's own feature extraction (Img2Tz). The image is uploaded BEFORE
  // Img2Tz: Img2Tz overwrites the image buffer with its two-level ridge
  // skeleton, and the PC's matcher (fingerprint_sift.py) needs the grayscale
  // capture. A press Img2Tz rejects therefore costs one wasted upload.
  uint32_t window = forceCapture ? 10000 : 1200;
  uint32_t start = millis();
  bool usable = false;
  long got = 0;
  uint32_t uartMs = 0;
  while (millis() - start < window) {
    int code = sensor.genImg();
    if (code == 0x00) {
      sensor.led(LED_CTRL_ON, LED_PURPLE);
      uint32_t t0 = millis();
      got = sensor.upImage(imageBuffer, IMAGE_BYTES);
      uartMs = millis() - t0;
      int tz = sensor.img2Tz(1);
      if (tz == 0x00) { usable = true; break; }
      Serial.printf("Capture rejected by sensor (Img2Tz code %d) - not a finger\n", tz);
      sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
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
  Serial.println("Finger captured - grayscale image uploaded before Img2Tz");

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
    // range instead of 0-240. 16 levels is all UpImage delivers; uploaded after
    // Img2Tz it would be a 2-level skeleton instead (see the capture above).
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

  uint32_t t0 = millis();
  uint32_t stamp = sendImageViaUDP();
  if (lastSentFrame != NULL) {                     // remember what we sent
    memcpy(lastSentFrame, imageBuffer, imgBytes);
    lastSentBytes = imgBytes;
    lastSentStamp = stamp;
    lastSentPacked = fmt->packed;
  }
  if (forceCapture) {
    char msg[64];
    snprintf(msg, sizeof(msg), "CAPTURE:SENT uart=%lums udp=%lums",
             (unsigned long)uartMs, (unsigned long)(millis() - t0));
    replyToTester(msg);
    forceCapture = false;
  }

  sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);
  // Cooldown (as the old sketch had), then wait for the finger to lift so one
  // press = one image. The PC may ask to double-check this press meanwhile.
  uint32_t cool = millis();
  while (millis() - cool < 1500) { serviceWhileWaiting(); delay(10); }
  if (TOUCH_PIN >= 0) {
    uint32_t lift = millis();
    while (digitalRead(TOUCH_PIN) == TOUCH_ACTIVE && millis() - lift < 5000) { serviceWhileWaiting(); delay(20); }
  }
  sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
}

// Between a press and the next capture: cards, PC commands, a pending double-check.
void serviceWhileWaiting() {
  handleRfid();
  handleUdpCommands();
  if (confirmRequested) { confirmRequested = false; runConfirm(confirmStamp); }
}

// ============================================================================
// UDP image transmit - wire-identical to the old FPM383C sketch
// ============================================================================

// Returns the header timestamp - the name the PC uses for this press (CONFIRM).
uint32_t sendImageViaUDP() {
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
  return header.timestamp;
}

// ============================================================================
// PAIR - template research capture (test/capture_template_pairs.py)
// ============================================================================
//
// One press, three uploads, each sent to the tester exactly as the sensor
// delivered it (no unpacking, no gates):
//   'R'  UpImage BEFORE Img2Tz - the image as GenImg captured it
//   'P'  UpImage AFTER Img2Tz  - what the normal capture path uploads
//   'C'  UpChar(1)             - the sensor's own feature file for this press
// R vs P shows whether Img2Tz overwrites the image buffer (which would explain
// why every normal capture is two-valued). P + C pair the sensor's skeleton
// with the features it extracted from it - the material for decoding the
// template format. The normal capture path is untouched.

#define CHAR_BYTES_MAX 2048
uint8_t charBuffer[CHAR_BYTES_MAX];

// Host uploads (FPUP packets, see handleUpload): image 'I' into imageBuffer,
// template 'B' for MATCHTMPL.
uint8_t  tmplB[CHAR_BYTES_MAX];
uint32_t upTotal[3] = { 0, 0, 0 }, upGot[3] = { 0, 0, 0 };   // indexed I, (unused), B

// Packet: "FPBD" + kind(1) + total(u32 LE) + offset(u32 LE) + payload.
// The offset lets the PC reassemble even if UDP reorders.
void sendBlob(char kind, const uint8_t* data, uint32_t size) {
  const uint16_t chunk = 1380;
  static uint8_t pkt[13 + chunk];
  uint32_t off = 0;
  do {
    uint16_t n = (uint16_t)min((uint32_t)chunk, size - off);
    memcpy(pkt, "FPBD", 4);
    pkt[4] = (uint8_t)kind;
    memcpy(pkt + 5, &size, 4);                     // ESP32 is little endian
    memcpy(pkt + 9, &off, 4);
    memcpy(pkt + 13, data + off, n);
    udp.beginPacket(lastCmdIP, lastCmdPort);
    udp.write(pkt, 13 + n);
    udp.endPacket();
    off += n;
    delay(10);
  } while (off < size);
}

void runPairCapture() {
  char msg[96];
  sensor.led(LED_CTRL_BREATH, LED_PURPLE, 100, 0);

  // Wait for a press the sensor's own extractor accepts. The raw image has to
  // be uploaded before Img2Tz runs, so a rejected press costs one upload.
  uint32_t start = millis();
  int tz = -1;
  long raw = 0;
  while (millis() - start < 15000) {
    handleRfid();
    if (TOUCH_PIN >= 0 && digitalRead(TOUCH_PIN) != TOUCH_ACTIVE) { delay(30); continue; }
    if (sensor.genImg() != 0x00) { delay(60); continue; }
    raw = sensor.upImage(imageBuffer, IMAGE_BYTES);
    if (raw <= 0) break;
    tz = sensor.img2Tz(1);
    if (tz == 0x00) break;
    snprintf(msg, sizeof(msg), "PAIR:REJECT tz=%d", tz);
    replyToTester(msg);
    delay(200);
  }

  if (raw <= 0 || tz != 0x00) {
    if (raw < 0) snprintf(msg, sizeof(msg), "PAIR:FAIL step=raw code=%ld", raw);
    else         snprintf(msg, sizeof(msg), "PAIR:TIMEOUT tz=%d", tz);
    replyToTester(msg);
    sensor.led(LED_CTRL_FLASH, LED_RED, 40, 3);
    delay(600);
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
    return;
  }

  sendBlob('R', imageBuffer, (uint32_t)raw);

  long proc = sensor.upImage(imageBuffer, IMAGE_BYTES);
  if (proc > 0) sendBlob('P', imageBuffer, (uint32_t)proc);

  long chr = sensor.upChar(1, charBuffer, CHAR_BYTES_MAX);
  if (chr > 0) sendBlob('C', charBuffer, (uint32_t)chr);

  snprintf(msg, sizeof(msg), "PAIR:%s raw=%ld proc=%ld char=%ld",
           (proc > 0 && chr > 0) ? "DONE" : "FAIL", raw, proc, chr);
  replyToTester(msg);
  sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);

  // one press = one PAIR, and the normal path must not pick this press up
  waitForLift();
  sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
}

void waitForLift() {
  if (TOUCH_PIN < 0) { delay(1500); return; }
  uint32_t lift = millis();
  while (digitalRead(TOUCH_PIN) == TOUCH_ACTIVE && millis() - lift < 5000) { handleRfid(); delay(20); }
  delay(300);
}

// ============================================================================
// PROBE - does this module export templates at all? (capture_template_pairs.py --probe)
// ============================================================================
//
// PAIR showed UpChar(1) after a single Img2Tz returns 2048 zero bytes. This
// separates "single feature files are not exported, merged templates are"
// from "template export is disabled":
//   'I'  ReadProdInfo (0x3C)  - 46 bytes: model, sensor, image size, template size
//   'S'  ReadSysPara  (0x0F)  - 16 bytes
//   'F'  ReadInfPage  (0x16)  - 512 bytes, if supported
//   'a'  UpChar(1) after press 1 -> Img2Tz(1)
//   'b'  UpChar(2) after press 2 -> Img2Tz(2)
//        Match(1,2)            - proves the buffers hold real features
//   'T'  UpChar(1) after RegModel
//   'U'  UpChar(2) after RegModel
// Read-only: nothing is stored to the sensor's library.

// Wait for a press the sensor's extractor accepts, features into `slot`.
int pressInto(uint8_t slot, uint32_t timeoutMs, const char* who) {
  uint32_t start = millis();
  int tz = -1;
  char msg[48];
  while (millis() - start < timeoutMs) {
    handleRfid();
    if (TOUCH_PIN >= 0 && digitalRead(TOUCH_PIN) != TOUCH_ACTIVE) { delay(30); continue; }
    if (sensor.genImg() != 0x00) { delay(60); continue; }
    tz = sensor.img2Tz(slot);
    if (tz == 0x00) return 0;
    snprintf(msg, sizeof(msg), "%s:REJECT tz=%d", who, tz);
    replyToTester(msg);
    delay(200);
  }
  return tz;
}

void runProbe() {
  char msg[160];
  uint8_t out[48];

  uint8_t p1[1] = { 0x3C };
  int code = sensor.command(p1, 1, out, 46, 1500);
  if (code == 0x00) sendBlob('I', out, 46);
  snprintf(msg, sizeof(msg), "PROBE:INFO code=%d", code);
  replyToTester(msg);

  uint8_t p2[1] = { 0x0F };
  if (sensor.command(p2, 1, out, 16, 1500) == 0x00) sendBlob('S', out, 16);

  long inf = sensor.readInfPage(charBuffer, CHAR_BYTES_MAX);
  if (inf > 0) sendBlob('F', charBuffer, (uint32_t)inf);

  sensor.led(LED_CTRL_BREATH, LED_PURPLE, 100, 0);
  replyToTester("PROBE:PLACE");
  int tz1 = pressInto(1, 20000, "PROBE");
  if (tz1 != 0) {
    snprintf(msg, sizeof(msg), "PROBE:TIMEOUT press=1 tz=%d", tz1);
    replyToTester(msg);
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
    return;
  }
  long a = sensor.upChar(1, charBuffer, CHAR_BYTES_MAX);
  if (a > 0) sendBlob('a', charBuffer, (uint32_t)a);
  sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);
  replyToTester("PROBE:LIFT");
  waitForLift();

  sensor.led(LED_CTRL_BREATH, LED_PURPLE, 100, 0);
  replyToTester("PROBE:PLACE_AGAIN");
  int tz2 = pressInto(2, 20000, "PROBE");
  if (tz2 != 0) {
    snprintf(msg, sizeof(msg), "PROBE:TIMEOUT press=2 tz=%d", tz2);
    replyToTester(msg);
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
    return;
  }
  long b = sensor.upChar(2, charBuffer, CHAR_BYTES_MAX);
  if (b > 0) sendBlob('b', charBuffer, (uint32_t)b);

  uint16_t score = 0;
  int match = sensor.matchBuffers(&score);
  int reg = sensor.regModel();
  long t = -1, u = -1;
  if (reg == 0x00) {
    t = sensor.upChar(1, charBuffer, CHAR_BYTES_MAX);
    if (t > 0) sendBlob('T', charBuffer, (uint32_t)t);
    u = sensor.upChar(2, charBuffer, CHAR_BYTES_MAX);
    if (u > 0) sendBlob('U', charBuffer, (uint32_t)u);
  }

  snprintf(msg, sizeof(msg),
           "PROBE:DONE match=%d score=%u reg=%d a=%ld b=%ld T=%ld U=%ld inf=%ld",
           match, score, reg, a, b, t, u, inf);
  replyToTester(msg);
  sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);
  waitForLift();
  sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
}

// ============================================================================
// Template lab - the PC drives the sensor's extractor and matcher
// (test/r503_template_lab.py). No finger needed, library untouched.
// ============================================================================
//
// IMG2TMPL  uploaded image -> DownImage -> Img2Tz(1) -> UpImage 'P' ->
//           DownImage -> Img2Tz(2) -> RegModel -> UpChar(1) 'T'.
//           A template from ONE image: recorded presses, and edited copies of
//           them (shifted, rotated), become templates - which is how the
//           format gets decoded. Img2Tz alone exports zeros on this firmware,
//           hence the RegModel of the image with itself.
// MATCHTMPL uploaded template B -> DownChar(2) -> Match against the template
//           the last RegModel made (IMG2TMPL, or a live TMPL press). Measured:
//           Match ignores a template sent with DownChar(1) and keeps comparing
//           the RegModel output, so that is the side the PC cannot replace.
//           This is 1:1 verification of a live press against a template the
//           PC stores, by the sensor's own matcher.

void runImg2Tmpl() {
  char msg[160];
  if (upTotal[0] == 0 || upGot[0] != upTotal[0]) {
    replyToTester("IMG2TMPL:FAIL no image uploaded");
    return;
  }
  static uint8_t* skel = NULL;                     // keeps imageBuffer intact for the 2nd DownImage
  if (skel == NULL) skel = (uint8_t*)ps_malloc(IMAGE_BYTES);
  if (skel == NULL) skel = (uint8_t*)malloc(IMAGE_BYTES);

  uint32_t len = upTotal[0];
  int d1 = sensor.downImage(imageBuffer, len);
  int tz1 = (d1 == 0) ? sensor.img2Tz(1) : -1;
  long proc = -1;
  if (tz1 == 0 && skel != NULL && img2TmplSkeleton) {
    proc = sensor.upImage(skel, IMAGE_BYTES);
    if (proc > 0) sendBlob('P', skel, (uint32_t)proc);
  }
  int d2 = (tz1 == 0) ? sensor.downImage(imageBuffer, len) : -1;
  int tz2 = (d2 == 0) ? sensor.img2Tz(2) : -1;
  int reg = (tz2 == 0) ? sensor.regModel() : -1;
  long t = -1;
  if (reg == 0) {
    t = sensor.upChar(1, charBuffer, CHAR_BYTES_MAX);
    if (t > 0) sendBlob('T', charBuffer, (uint32_t)t);
  }
  snprintf(msg, sizeof(msg), "IMG2TMPL:%s down=%d tz1=%d proc=%ld down2=%d tz2=%d reg=%d tmpl=%ld",
           t > 0 ? "DONE" : "FAIL", d1, tz1, proc, d2, tz2, reg, t);
  replyToTester(msg);
}

// TMPL      live press -> template the PC decodes and matches (r503_template.py).
//           A template only exists after RegModel, which needs two feature
//           files, and Img2Tz cannot run twice on one capture: the first run
//           marks the image buffer used (measured: a second Img2Tz answers 0x07
//           in 3 ms). So the second feature file comes from a second capture
//           of the same press, taken while the finger is still down - no image
//           crosses the UART. If the finger was already lifted, it asks for the
//           same finger again (TMPL:PLACE_AGAIN).
// TMPL RAW  one capture only: it is uploaded before Img2Tz(1) and sent back
//           with DownImage for Img2Tz(2) - exactly how the evaluated templates
//           were built, at the cost of ~2.2 s of UART transfer. The image is
//           also sent to the PC ('R'), for comparing matchers on one press.
void runLiveTemplate(bool rawMode) {
  char msg[112];
  sensor.led(LED_CTRL_BREATH, LED_PURPLE, 100, 0);

  // first capture -> CharBuffer1 (RAW mode keeps a copy of the image first)
  uint32_t start = millis();
  int tz1 = -1;
  long raw = 0;
  while (millis() - start < 15000) {
    handleRfid();
    if (TOUCH_PIN >= 0 && digitalRead(TOUCH_PIN) != TOUCH_ACTIVE) { delay(30); continue; }
    if (sensor.genImg() != 0x00) { delay(60); continue; }
    if (rawMode) {
      raw = sensor.upImage(imageBuffer, IMAGE_BYTES);
      if (raw <= 0) { tz1 = -2; break; }
    }
    tz1 = sensor.img2Tz(1);
    if (tz1 == 0x00) break;
    snprintf(msg, sizeof(msg), "TMPL:REJECT tz=%d", tz1);
    replyToTester(msg);
    delay(200);
  }
  if (tz1 != 0x00) {
    snprintf(msg, sizeof(msg), "TMPL:TIMEOUT tz=%d", tz1);
    replyToTester(msg);
    sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
    return;
  }

  // second feature file -> CharBuffer2
  uint32_t t1 = millis();
  int tz2 = -1, captures = 1;
  if (rawMode) {
    int d = sensor.downImage(imageBuffer, (uint32_t)raw);
    tz2 = (d == 0) ? sensor.img2Tz(2) : -1;
  } else {
    bool asked = false;
    while (millis() - t1 < 10000) {
      handleRfid();
      bool touching = (TOUCH_PIN < 0) || digitalRead(TOUCH_PIN) == TOUCH_ACTIVE;
      if (touching && sensor.genImg() == 0x00) {
        captures++;
        tz2 = sensor.img2Tz(2);
        if (tz2 == 0x00) break;
      } else if (!asked && millis() - t1 > 600) {  // finger already gone
        replyToTester("TMPL:PLACE_AGAIN");
        sensor.led(LED_CTRL_FLASH, LED_PURPLE, 60, 0);
        asked = true;
      }
      delay(40);
    }
  }

  int reg = (tz2 == 0x00) ? sensor.regModel() : -1;
  long t = -1;
  if (reg == 0x00) {
    t = sensor.upChar(1, charBuffer, CHAR_BYTES_MAX);
    if (t > 0) sendBlob('T', charBuffer, (uint32_t)t);
    // RAW: the PC also gets the image the template was built from, so image
    // matchers can be compared with the template on identical presses
    if (t > 0 && rawMode) sendBlob('R', imageBuffer, (uint32_t)raw);
  }
  snprintf(msg, sizeof(msg), "TMPL:%s mode=%s tz2=%d reg=%d tmpl=%ld captures=%d ms=%lu",
           t > 0 ? "DONE" : "FAIL", rawMode ? "raw" : "fast", tz2, reg, t, captures,
           (unsigned long)(millis() - start));
  replyToTester(msg);
  sensor.led(LED_CTRL_FLASH, t > 0 ? LED_BLUE : LED_RED, 60, 2);
  waitForLift();
  sensor.led(LED_CTRL_BREATH, LED_BLUE, 100, 0);
}

// CONFIRM <stamp>  the sensor double-check (fingerprint_gym.py, r503_link.py).
//           The PC found the last press it received a borderline match and asks
//           the R503's own matcher. The press is still in lastSentFrame: it goes
//           back to the sensor (DownImage), its template is built (Img2Tz x2 on
//           two downloads, RegModel), and is left as the side Match compares
//           against; the PC then sends the member's stored templates with
//           MATCHTMPL. <stamp> must name the frame still held here, or another
//           press could be checked by mistake.
uint8_t confirmPacked[IMAGE_BYTES / 2];

void runConfirm(uint32_t stamp) {
  char msg[96];
  if (lastSentFrame == NULL || lastSentBytes == 0 || stamp != lastSentStamp) {
    snprintf(msg, sizeof(msg), "CONFIRM:FAIL stale have=%lu", (unsigned long)lastSentStamp);
    replyToTester(msg);
    return;
  }
  // back in the format the sensor delivered it (8-bit values are nibble n * 17)
  const uint8_t* src = lastSentFrame;
  uint32_t len = lastSentBytes;
  if (lastSentPacked) {
    for (uint32_t i = 0; i < lastSentBytes / 2; i++)
      confirmPacked[i] = (lastSentFrame[2 * i] & 0xF0) | (lastSentFrame[2 * i + 1] >> 4);
    src = confirmPacked;
    len = lastSentBytes / 2;
  }
  uint32_t t0 = millis();
  int d1 = sensor.downImage(src, len);
  int tz1 = (d1 == 0) ? sensor.img2Tz(1) : -1;
  int d2 = (tz1 == 0) ? sensor.downImage(src, len) : -1;
  int tz2 = (d2 == 0) ? sensor.img2Tz(2) : -1;
  int reg = (tz2 == 0) ? sensor.regModel() : -1;
  snprintf(msg, sizeof(msg), "CONFIRM:%s down=%d tz1=%d down2=%d tz2=%d reg=%d ms=%lu",
           reg == 0 ? "DONE" : "FAIL", d1, tz1, d2, tz2, reg, (unsigned long)(millis() - t0));
  replyToTester(msg);
}

void runMatchTmpl() {
  char msg[96];
  if (upTotal[2] == 0 || upGot[2] != upTotal[2]) {
    replyToTester("MATCHTMPL:FAIL template B not uploaded");
    return;
  }
  int dB = sensor.downChar(2, tmplB, upTotal[2]);
  uint16_t score = 0;
  int m = (dB == 0) ? sensor.matchBuffers(&score) : -1;
  snprintf(msg, sizeof(msg), "MATCHTMPL:%s downB=%d match=%d score=%u",
           (dB == 0 && m >= 0) ? "DONE" : "FAIL", dB, m, score);
  replyToTester(msg);
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

// Packet: "FPUP" + slot(1) + total(u32 LE) + offset(u32 LE) + payload. The PC
// sends chunks in order and waits for each "UP:<slot> <got>/<total>" ack.
void handleUpload(const uint8_t* pkt, int n) {
  char slot = (char)pkt[4];
  uint32_t total, off, len = (uint32_t)(n - 13);
  memcpy(&total, pkt + 5, 4);
  memcpy(&off, pkt + 9, 4);
  int i = (slot == 'I') ? 0 : (slot == 'B') ? 2 : -1;
  uint32_t cap = (i == 0) ? IMAGE_BYTES : CHAR_BYTES_MAX;
  char msg[48];
  if (i < 0 || total > cap || off + len > total) {
    snprintf(msg, sizeof(msg), "UP:%c ERR", slot);
    replyToTester(msg);
    return;
  }
  uint8_t* dst = (i == 0) ? imageBuffer : tmplB;
  memcpy(dst + off, pkt + 13, len);
  upTotal[i] = total;
  upGot[i] = off + len;
  snprintf(msg, sizeof(msg), "UP:%c %lu/%lu", slot, (unsigned long)upGot[i], (unsigned long)total);
  replyToTester(msg);
}

void handleUdpCommands() {
  int size = udp.parsePacket();
  if (size <= 0) return;

  static uint8_t pkt[1472];
  int n = udp.read(pkt, sizeof(pkt));
  if (n <= 0) return;

  if (n >= 13 && memcmp(pkt, "FPUP", 4) == 0) {   // template lab upload
    lastCmdIP = udp.remoteIP();
    lastCmdPort = udp.remotePort();
    handleUpload(pkt, n);
    return;
  }

  char buf[161];
  n = min(n, (int)sizeof(buf) - 1);
  memcpy(buf, pkt, n);
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
  if (strncasecmp(buf, "PAIR", 4) == 0) {          // image + feature file, one press
    pairRequested = true;
    replyToTester("PAIR:WAIT");
    return;
  }
  if (strncasecmp(buf, "PROBE", 5) == 0) {         // template export check, two presses
    probeRequested = true;
    replyToTester("PROBE:WAIT");
    return;
  }
  if (strncasecmp(buf, "IMG2TMPL", 8) == 0) {      // template from the uploaded image
    img2TmplRequested = true;
    img2TmplSkeleton = (strstr(buf, "FAST") == NULL);
    replyToTester("IMG2TMPL:WAIT");
    return;
  }
  if (strncasecmp(buf, "CONFIRM", 7) == 0) {       // "CONFIRM <stamp>": double-check that press
    confirmStamp = (uint32_t)strtoul(buf + 7, NULL, 10);
    confirmRequested = true;
    replyToTester("CONFIRM:WAIT");
    return;
  }
  if (strncasecmp(buf, "MATCHTMPL", 9) == 0) {     // sensor Match of an uploaded template
    matchTmplRequested = true;
    replyToTester("MATCHTMPL:WAIT");
    return;
  }
  if (strncasecmp(buf, "TMPL", 4) == 0) {          // template from a live press ("TMPL RAW")
    liveTmplRequested = true;
    liveTmplRaw = (strncasecmp(buf + 4, " RAW", 4) == 0);
    replyToTester("TMPL:WAIT");
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
