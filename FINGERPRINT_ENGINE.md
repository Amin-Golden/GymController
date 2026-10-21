# Fingerprint engine (grayscale SIFT) - setup and operation

The gym identifies members by fingerprint alone, among everyone enrolled
(1000+ members), with `fingerprint_sift.py`:

1. The R503 sends its **grayscale** image. The firmware uploads it before the
   sensor's own processing turns it into a black-and-white ridge outline.
2. SIFT finds about 300-900 small image patches per press. Each patch is stored
   with its position and angle.
3. A FAISS index shortlists the 5 fingers whose patches the press resembles most.
4. Each shortlisted finger is verified by counting the matched patches that
   agree on **one rotation and shift** of the finger. A member is accepted at
   20 or more.
5. **Sensor double-check.** A borderline match (20-39) is accepted only if the
   R503's own matcher agrees. It compares the press with the member's stored
   sensor templates. That matcher is a different algorithm, so a stranger who
   fools one by chance is very unlikely to fool both. Clear matches (typical
   member presses score ~170) are not slowed down.

Measured on this sensor, enrolled one day and tested the next with shifted,
rotated, light and firm presses (`test/r503_template_lab.py compare`):

| | |
|---|---|
| members identified | 71 of 72 presses (the miss was a very weak press, rejected - not mistaken) |
| wrong member | 0 |
| strangers (fingers never enrolled) let in | 0 |
| best score any stranger reached | 2-5 (threshold 20) |
| typical genuine score | 170 (5th percentile 45) |
| speed, 1000 members, dev PC | ~0.2 s per press after the image arrives |

Earlier approaches on the same presses: SIFT without the geometry check had a
heavy impostor tail. The sensor's decoded minutiae template
(`r503_template.py`) identified 65/72. The minutiae engines reading the
outline image did worse. See `FINGERPRINT_DIAGNOSIS.md` for the history.

## Setup

1. **Python packages** (OrangePi and PC): `pip install faiss-cpu opencv-python`
   (OpenCV 4.4 or newer has SIFT). Check with `python3 fingerprint_sift.py --selftest`.
2. **Database:** `psql -U postgres -d gym-db -f migrate_fingerprint_sift_gray.sql`
   (adds `sift_gray` and `sensor_template` to `fingerprint_features`).
   Enrollment refuses to save until this has run. Run it **again** if you ran
   an earlier version: without `sensor_template`, fingers are saved but not
   double-checked.
3. **Firmware:** flash `finger_UDP/finger_UDP.ino` to **both** ESP32s: `ROLE 0` for
   the entrance unit, `ROLE 1` for the locker unit. This version can be updated
   over WiFi afterwards (see below), so this is the last USB flash. With older
   firmware the PC receives the black-and-white outline, and identification logs
   `the sensor sent a two-level skeleton image`.
4. **Restart** the gym application. `FINGERPRINT_ENGINE=auto` (the default)
   selects this engine.

## Enrollment - every member again, one finger at a time

Existing fingerprints were made from the outline image and are not read by
this engine. At startup the log says how many members still need to enroll
again.

Each finger is enrolled on its own. The admin app sends one notification per
finger, for example with one button per finger:

```sql
SELECT pg_notify('client_changes', '{"action":"FINGERPRINT_ENROLL","client_id":123,"finger_index":1}');
SELECT pg_notify('client_changes', '{"action":"FINGERPRINT_ENROLL","client_id":123,"finger_index":2}');
```

Enrolling finger 1 replaces only finger 1, and finger 2 is kept. Without
`finger_index`, both fingers are enrolled in one go.

The member presses the same finger `ENROLL_PRESSES` times (default 8), moving
it a little as the LCD prompts. A press is rejected, and costs one of
`ENROLL_RETRY_BUDGET` retries, when:

| LCD message | meaning |
|---|---|
| `no usable ridges` | empty or smeared press |
| `only N features (need 150)` | too faint - press firmer |
| `does not match the earlier presses` | another finger, or moved too far |

When all presses are in, the finger is searched against every **other**
member. If it matches someone, `This finger matches member X ...` is shown and
the finger is **not saved**. That is a double enrollment, or two members the
door would confuse. Set `SIFT_ENROLL_COLLISION=warn` to save anyway. Otherwise
the log shows `Closest other member: ... with N matches (threshold 20)`.

Last, the LCD shows `Preparing the sensor check...`. The sensor builds its own
template for the 3 strongest presses (about 10 s), and they are saved with the
finger for the double-check. If the sensor cannot be reached, the finger is
still saved. It is then just not double-checked, and the log says so.

## Sensor double-check

```
press -> PC matcher -> score
   >= 40   clear match       -> open
   20-39   borderline        -> sensor double-check -> open only if it agrees
   < 20    no match          -> reject
```

For a borderline press, the PC sends `CONFIRM` to the ESP32 that sent it. The
ESP32 still holds that image. The sensor rebuilds its template, and the PC
then sends the member's stored sensor templates one by one (`MATCHTMPL`) until
one matches. This adds about 4 s, and only for borderline presses. The log
shows `sensor confirms member ...` or `the sensor did not confirm`.

If the ESP32 already holds a newer press, or does not answer, the press is
rejected and the member presses again. Members enrolled before this existed
have no sensor template. Their borderline matches are accepted unchecked and
logged `not double-checked`; enrolling the finger again enables the check.

**Measure it on your sensor before relying on it.** The sensor's matcher is
strict, and how often it confirms a real member decides how many borderline
members it turns away. With the ESP32 on, and recorded sessions from
`r503_template_lab.py --raw capture`:

```bash
py -3 test/r503_template_lab.py confirmtest
```

It prints how many next-session presses the sensor confirms for their own
finger, and for the closest *wrong* finger (that one must be 0). If own-finger
confirmations are low, store more templates per finger
(`SENSOR_TEMPLATES_PER_FINGER=5`) or narrow the band (`SIFT_CONFIRM_BELOW=30`).

## Updating the ESP32s over WiFi

After this firmware is on both units (one USB flash), later versions go over
WiFi from any PC on the same network:

```bash
py -3 finger_UDP/ota_upload.py                 # builds and updates entrance + locker
py -3 finger_UDP/ota_upload.py --units locker  # one unit
```

Each unit is built with its own `ROLE` and updated in ~30 s. It reboots and is
checked with `PING`. The Arduino IDE also lists the units as network ports
(`gym-fp-entrance`, `gym-fp-locker`). The upload password is `otaPassword` in
the sketch: change it there before the first USB flash, and the script reads it
from the same place.

## Checking accuracy as the gym grows

A stranger at the door is compared with every enrolled finger, so the
important number is how close **different** members come to each other. Measure
it from the real enrolled members, on the OrangePi or any PC that reaches the database:

```bash
python3 fingerprint_sift.py --report --db-host <database host>
```

The report prints, for each finger, the closest *other* member, then the worst
pairs. While the worst stays far below 20, the threshold has headroom. If two
members reach 20, re-enroll those fingers or raise `SIFT_MATCH_INLIERS`.

Speed and memory on the actual board, at gym scale. This needs labelled
presses from `r503_template_lab.py --raw capture`:

```bash
python3 fingerprint_sift.py --bench --images test/template_lab/session_a --members 1000
```

## Memory and storage

About 75 KB per press. At 1000 members x 2 fingers x 8 presses that is
~1.2 GB in the database and in RAM, plus ~0.4 GB of index. Up to ~95 members the
index is exact. Past that, an approximate index is trained once in the
background (about 2 minutes on the dev PC) while the exact one keeps serving.
The index is saved under `fingerprint_storage/sift_index/`, so restarts do not
rebuild it.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `FINGERPRINT_ENGINE` | `auto` | `auto`/`gray` = this engine; `local`/`nbis`/`sift` = the old ones |
| `SIFT_MATCH_INLIERS` | `20` | geometrically consistent matches needed to accept |
| `SIFT_MIN_KEYPOINTS` | `150` | fewer features than this rejects an enrollment press |
| `SIFT_ENROLL_COLLISION` | `reject` | `warn` = save a finger that matches another member |
| `SIFT_CONFIRM_BELOW` | `40` | matches below this (and >= 20) need the sensor double-check; `0` = off |
| `SENSOR_TEMPLATES_PER_FINGER` | `3` | sensor templates stored per finger and tried at a double-check |
| `ENROLL_PRESSES` | `8` | presses per finger |
| `SIFT_NPROBE` | `8` | approximate index: lists searched (higher = slower, more thorough) |
| `SIFT_SHORTLIST` | `5` | fingers verified per press |
| `SIFT_INDEX_DIR` | `fingerprint_storage/sift_index` | saved index |
| `FINGERPRINT_SAVE_QUERY_IMAGES` | `0` | keep door presses (for tuning) |
