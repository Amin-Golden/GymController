# NBIS fingerprint engine - setup and operation

The gym system now matches fingerprints with **NBIS** (NIST Biometric Image
Software): `MINDTCT` extracts a minutiae template from the sensor image and
`BOZORTH3` compares templates. It is public domain, designed for ~500 dpi
images, and it is a real fingerprint algorithm - unlike the SIFT keypoint
matcher, which is a generic image-correspondence method.

The SIFT engine has **not** been removed. It stays as a fallback for members
enrolled before this migration and can be forced back on at any time.

---

## 1. Install NBIS

The wrapper needs two executables, `mindtct` and `bozorth3` (and optionally
`cwsq`, used only if this build of `mindtct` refuses PGM input).

**Linux / Raspberry Pi / WSL** - build from the NIST source release:

```bash
sudo apt install build-essential cmake
# unpack the NBIS release, then from its top directory:
./setup.sh /usr/local/nbis --without-X11
make config
make it
sudo make install
ls /usr/local/nbis/bin/mindtct /usr/local/nbis/bin/bozorth3
```

**Windows** - build the same source under MSYS2/MinGW-w64 (produces
`mindtct.exe` / `bozorth3.exe`), or copy the binaries from a machine that has
them. Nothing else from NBIS is required.

Then either put that `bin` directory on `PATH`, or point the system at it:

```powershell
setx NBIS_BIN_DIR "C:\Program Files\NBIS\bin"
```

Verify:

```powershell
py -3 fingerprint_nbis.py --selftest
```

The self test locates the binaries, works out which image format this build of
`mindtct` accepts, and probes which multi-file syntax this build of `bozorth3`
supports. It prints exactly what it decided - if it falls back to
"one process per comparison", matching still works but is slower.

## 2. Migrate the database

```powershell
psql -U <user> -d <database> -f migrate_fingerprint_nbis.sql
```

This adds `xyt_template`, `minutiae_count`, `image_path` and `nbis_updated_at`
to `fingerprint_features`. Nothing is dropped: the `descriptors` and
`keypoints_data` columns the SIFT engine uses are untouched.

Until the migration runs, the system keeps working on the SIFT engine and warns
once at startup.

## 3. Existing members

New enrollments store the raw image, the SIFT descriptors and the NBIS template
together. Members enrolled *before* this change have no template, so:

```powershell
py -3 backfill_nbis_templates.py --dry-run   # what can be recovered
py -3 backfill_nbis_templates.py             # write the templates
```

Backfill works only where the raw image still exists on disk. For anyone else,
NBIS returns no match and the system falls back to the SIFT engine
(`NBIS_FALLBACK_TO_SIFT=1`, the default) until they enroll again.

## 4. Choose a threshold

BOZORTH3 returns an integer score, not a percentage. NIST's usual operating
points are 24 (loose) and 40 (standard); higher means fewer false accepts and
more rejected members. For 1:N identification over a growing member list, start
at 40 and measure:

```powershell
py -3 test/test_nbis_pipeline.py
```

It runs both engines over the stored captures, holds out one sample per client
as the query, and prints rank-1 accuracy plus FAR/FRR at a range of thresholds.
Pick the lowest threshold with 0% FAR on your data, then add headroom - your
test set is smaller than your member list, and false accepts grow with it.

```powershell
setx NBIS_MATCH_THRESHOLD 45
```

If two members ever score close to each other, also enable the separation
requirement - the winner must beat the runner-up by this ratio:

```powershell
setx NBIS_MATCH_MARGIN 1.3
```

---

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `FINGERPRINT_ENGINE` | `auto` | `auto` (NBIS if installed, else SIFT), `nbis`, or `sift` |
| `NBIS_FALLBACK_TO_SIFT` | `1` | In `auto`, try SIFT when NBIS reports no match |
| `NBIS_BIN_DIR` | *(empty)* | Directory holding mindtct/bozorth3/cwsq |
| `NBIS_MATCH_THRESHOLD` | `40` | BOZORTH3 score required to accept a match |
| `NBIS_MATCH_MARGIN` | `1.0` | Winner must beat the runner-up by this ratio (1.0 = off) |
| `NBIS_PPI` | `508` | Sensor resolution (R503 and FPM383 are both 508 dpi) |
| `NBIS_CONTRAST_BOOST` | `1` | Pass `-b` to mindtct; helps low-contrast sensor images |
| `NBIS_GALLERY_DIR` | `fingerprint_storage/nbis_gallery` | Where enrolled templates are materialised |
| `SIFT_MATCH_THRESHOLD` | `0.12` | Threshold for the fallback engine (ratio-test score, not a percentage) |
| `FINGERPRINT_SAVE_ENROLLMENT_IMAGES` | `1` | Keep raw enrollment captures |
| `FINGERPRINT_SAVE_QUERY_IMAGES` | `0` | Keep identification captures (threshold tuning) |
| `FINGERPRINT_QUERY_RETENTION_DAYS` | `30` | Auto-delete query images after N days |
| `FINGERPRINT_IMAGE_WIDTH` / `_HEIGHT` | `192` | Fallback geometry when the UDP header is missed |

## Raw image retention

`fingerprint_storage/images/enrolled/<client_id>/f<finger>_s<sample>_<time>.png`

Enrollment used to store only SIFT descriptors, which meant that changing the
algorithm forced every member to enroll again. Raw captures are now kept as
lossless PNG so any future engine can be rebuilt from them - that is what makes
`backfill_nbis_templates.py` possible.

Two consequences worth knowing:

- **These files are biometric data.** They sit next to the database, not in it.
  Back them up together, restrict access to the directory, and delete a
  member's directory when you delete the member.
- Query images are **off** by default (`FINGERPRINT_SAVE_QUERY_IMAGES=0`).
  Turn them on while tuning a threshold, then turn them off again;
  `fingerprint_image_store.prune_query_images()` clears old ones.

Storage is roughly 15-25 KB per capture: 10 captures per member is about
0.2 MB, so 1000 members is around 200 MB.

## Operational notes

- **Minutiae count is a quality signal.** Enrollment warns below 12 minutiae -
  that usually means a dry, partial or off-centre press. A member enrolled from
  poor captures will keep failing at the door; re-enroll them instead of
  lowering the threshold.
- **The gallery is cached on disk** and rebuilt when the enrolled set changes or
  after any enrollment. Identification writes one probe template and makes a
  single batch call to BOZORTH3.
- **Both engines are stored per sample**, so switching `FINGERPRINT_ENGINE`
  needs no re-enrollment in either direction.
- **The R503 transport still matters.** NBIS only sees what the ESP32 sends: a
  192x192 image is ~36 KB over UDP with no sequence numbers or retransmission
  in the current firmware protocol, so a single lost packet wastes the capture.
  That is a separate fix from this one.
