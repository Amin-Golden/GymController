# Using the R503's own matcher (firmware additions)

Why: the sensor contains a commercial minutiae matcher on its own DSP, running
on the full-resolution image *before* the enhancement that flattens what reaches
the PC. Its published false-accept rate at `sec=3` is under 0.001 %. Matching
PC-side on the uploaded skeleton tops out around EER 9 % (see
`../FINGERPRINT_DIAGNOSIS.md`). This also removes the need for NBIS, which is
what will not build on the OrangePi 5B.

Your unit reports `lib=150 sec=3 templates=0`, so the internal library holds
**150 templates = 75 members at two fingers each**. Past that, see section 6.

These are additions only. Nothing in the current sketch has to be removed, and
the existing image-upload path keeps working alongside them.

---

## 1. Sensor commands

Add these to the sensor class, next to `img2Tz()`. They all go through the
existing `command()` helper, so framing, ACK parsing and timeouts are already
handled.

```cpp
  // RegModel: merge CharBuffer1 + CharBuffer2 into one template, left in both.
  // 0x00 ok, 0x0A = the two impressions did not combine (ask for a re-press).
  int regModel() { uint8_t p[1] = { 0x05 }; return command(p, 1, NULL, 0, 2000); }

  // Store: write the template in `slot` (1 or 2) to library position `page`.
  int storeModel(uint16_t page, uint8_t slot = 1) {
    uint8_t p[4] = { 0x06, slot, (uint8_t)(page >> 8), (uint8_t)page };
    return command(p, 4, NULL, 0, 2000);
  }

  // Search: 1:N over `count` slots from `start`. On 0x00, page/score are set.
  // 0x09 = no match. This one call is the whole identification step.
  int searchLibrary(uint16_t* page, uint16_t* score,
                    uint8_t slot = 1, uint16_t start = 0, uint16_t count = 0xFFFF) {
    uint8_t p[6] = { 0x04, slot, (uint8_t)(start >> 8), (uint8_t)start,
                     (uint8_t)(count >> 8), (uint8_t)count };
    uint8_t out[4] = { 0, 0, 0, 0 };
    int code = command(p, 6, out, 4, 3000);
    if (code == 0x00) {
      if (page)  *page  = ((uint16_t)out[0] << 8) | out[1];
      if (score) *score = ((uint16_t)out[2] << 8) | out[3];
    }
    return code;
  }

  // Match: 1:1 compare CharBuffer1 against CharBuffer2. 0x00 ok, 0x08 no match.
  int matchBuffers(uint16_t* score) {
    uint8_t p[1] = { 0x03 }, out[2] = { 0, 0 };
    int code = command(p, 1, out, 2, 2000);
    if (code == 0x00 && score) *score = ((uint16_t)out[0] << 8) | out[1];
    return code;
  }

  int deleteModel(uint16_t page, uint16_t n = 1) {
    uint8_t p[5] = { 0x0C, (uint8_t)(page >> 8), (uint8_t)page,
                     (uint8_t)(n >> 8), (uint8_t)n };
    return command(p, 5, NULL, 0, 2000);
  }

  int emptyLibrary() { uint8_t p[1] = { 0x0D }; return command(p, 1, NULL, 0, 3000); }
```

---

## 2. Enrollment

Two presses of the same finger, merged into one template, stored at `page`.
`page` is just a slot number, so keep a `client_id + finger -> page` map in the
database; that is how the PC turns a search hit back into a member.

```cpp
// 0 on success, else the sensor's confirmation code (>0), or -1 on timeout.
int enrollFinger(uint16_t page) {
  for (uint8_t pass = 1; pass <= 2; pass++) {
    sensor.led(LED_CTRL_BREATH, LED_PURPLE, 100, 0);
    replyToTester(pass == 1 ? "ENROLL:PLACE" : "ENROLL:PLACE_AGAIN");

    uint32_t deadline = millis() + 15000;
    int code = -1;
    while (millis() < deadline) {
      code = sensor.genImg();
      if (code == 0x02) { delay(60); continue; }      // no finger on the window yet
      if (code == 0x00) code = sensor.img2Tz(pass);   // buffer 1, then buffer 2
      if (code == 0x00) break;
      delay(60);
    }

    if (code != 0x00) {
      sensor.led(LED_CTRL_FLASH, LED_RED, 40, 4);
      replyToTester("ENROLL:FAIL_CAPTURE");
      return code > 0 ? code : -1;
    }

    sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);
    if (pass == 1) {
      replyToTester("ENROLL:LIFT");
      // one press must not be read twice
      if (TOUCH_PIN >= 0) {
        uint32_t lift = millis();
        while (digitalRead(TOUCH_PIN) == TOUCH_ACTIVE && millis() - lift < 5000) delay(20);
      }
      delay(400);
    }
  }

  int code = sensor.regModel();
  if (code != 0x00) { replyToTester("ENROLL:FAIL_MERGE"); return code; }

  code = sensor.storeModel(page);
  if (code != 0x00) { replyToTester("ENROLL:FAIL_STORE"); return code; }

  char msg[48];
  snprintf(msg, sizeof(msg), "ENROLL:OK page=%u", page);
  replyToTester(msg);
  sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 3);
  return 0;
}
```

Two passes is the sensor's documented minimum and matches its datasheet flow.

---

## 3. Identification

This replaces the entire upload-and-match-on-the-PC path.

```cpp
// true on a hit; page/score then identify the member.
bool identifyFinger(uint16_t* page, uint16_t* score) {
  if (sensor.genImg()  != 0x00) return false;
  if (sensor.img2Tz(1) != 0x00) return false;        // the sensor's own quality gate
  return sensor.searchLibrary(page, score) == 0x00;
}
```

In the touch handler, where `sendImageViaUDP()` is called today:

```cpp
  uint16_t page = 0, score = 0;
  if (identifyFinger(&page, &score)) {
    char msg[64];
    snprintf(msg, sizeof(msg), "MATCH page=%u score=%u", page, score);
    udp.beginPacket(pcIP, matchPort);                // a small text port, not 8889
    udp.write((const uint8_t*)msg, strlen(msg));
    udp.endPacket();
    sensor.led(LED_CTRL_FLASH, LED_BLUE, 60, 2);
  } else {
    replyToTester("MATCH:NONE");
    sensor.led(LED_CTRL_FLASH, LED_RED, 40, 3);
  }
```

The PC looks `page` up in its `page -> client_id` table. No image crosses the
network, no PC-side minutiae extraction, no NBIS.

Keep `sendImageViaUDP()` reachable behind a command. Raw images are still worth
collecting for the record and for any future algorithm work.

---

## 4. Command handlers

Add to `handleUdpCommands()` so the PC can drive enrollment:

```cpp
  else if (strncmp(buf, "ENROLL ", 7) == 0) {          // "ENROLL 12"
    enrollFinger((uint16_t)atoi(buf + 7));
  }
  else if (strncmp(buf, "DELETE ", 7) == 0) {
    int code = sensor.deleteModel((uint16_t)atoi(buf + 7));
    replyToTester(code == 0 ? "DELETE:OK" : "DELETE:FAIL");
  }
  else if (strcmp(buf, "IDENTIFY") == 0) {
    uint16_t page = 0, score = 0;
    char msg[64];
    if (identifyFinger(&page, &score))
      snprintf(msg, sizeof(msg), "MATCH page=%u score=%u", page, score);
    else
      snprintf(msg, sizeof(msg), "MATCH:NONE");
    replyToTester(msg);
  }
  else if (strcmp(buf, "EMPTY") == 0) {
    replyToTester(sensor.emptyLibrary() == 0 ? "EMPTY:OK" : "EMPTY:FAIL");
  }
```

Check it from the PC with the existing harness (no gym app running), watching
`templates=N` climb as you enroll:

```bash
py -3 test/test_r503_esp32.py --esp 192.168.1.120 --skip-scan
```

---

## 5. Tuning

`sec=3` is the security level reported by `readSysPara`. It trades false accepts
against false rejects and is settable with `SetSysPara` (`0x0E`, parameter 5,
value 1-5). 3 is the default and a sensible start. Raise to 4 if you ever see a
false accept in testing; drop to 2 only if genuine members are rejected and the
print quality is known to be good.

`Search` also returns a score. Log it for the first few weeks: it shows how much
headroom real members actually have, which is the only way to set the level from
evidence rather than by feel.

---

## 6. Beyond 150 members

`lib=150` caps on-sensor storage at 75 members with two fingers each. Past that,
keep the sensor's matcher but hold the templates on the PC:

* **Enroll:** `Img2Tz` -> `RegModel` -> `UpChar` (`0x08`) uploads the template
  bytes; store them in `fingerprint_features` where `xyt_template` goes today.
* **Verify:** `Img2Tz(1)` on the live finger, then per candidate `DownChar`
  (`0x09`) into buffer 2 + `matchBuffers()`.

Unlimited members, same matcher, but one UART round trip per candidate
(~50-100 ms). That is fine for 1:1 - the member is identified by face, card or
keypad first and the fingerprint confirms - and fine for the locker. It is too
slow for a cold 1:N over hundreds of members at the door.

The practical layout for a large gym: the 150 slots hold the frequent members
for instant `Search`; everyone else falls back to `DownChar` + `Match` against
their own stored template once another identifier has narrowed it to one person.
