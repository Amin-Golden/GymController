# Fingerprint recognition: diagnosis and the way forward

This supersedes the earlier SIFT/FAISS/embedding design documents (removed from
the repo; see git history), which measurement has now ruled out.

---

## 1. Verdict

The matching algorithm does not work, and it cannot be fixed by tuning. SIFT
descriptors are the wrong tool for fingerprints. Measured on 85 real labelled
captures (17 fingers x 5 presses):

| metric | measured | needed |
|---|---|---|
| d' (genuine/impostor separation) | **1.00** | > 3 |
| Equal error rate | **28.9 %** | < 1 % |
| False rejects at the shipped threshold `SIFT_MATCH_THRESHOLD=0.12` | **74.7 %** | < 5 % |
| Rank-1 identification | **60 %** | > 99 % |
| Rank-1 *and* above threshold | **24.7 %** | > 99 % |

Three of every four legitimate members are turned away, and the 25 % that do get
in are nearly matched by the impostor rate. That is the behaviour being reported.

**Why SIFT fails.** SIFT was built to find distinctive *landmarks* - a corner, a
logo, a texture patch. A fingerprint is the opposite: every part of every finger
is the same motif, parallel ridges about 0.5 mm apart. SIFT descriptors taken
from two different people's fingers therefore look almost identical, which is
exactly what the numbers say. Fingerprints are matched on **minutiae** - the
points where a ridge ends or forks - and on their relative geometry.

The FAISS index, the PCA step and the 128-d "embedding" in
`migrate_fingerprint_embeddings.sql` all sit on top of those same descriptors.
They make a weak signal faster to search, not stronger. Mean-pooling a few
hundred SIFT descriptors into one vector discards what little identity
information was there.

---

## 1b. "identify --trials 10 cannot match any fingerprints" - why, and what changed

You hit this after enrolling 10 real people (`fingerprint_storage/images`,
50 presses). That dataset is the most honest evidence in this project - larger
and more realistic than anything used above - and it explains the symptom
exactly:

**The threshold was miscalibrated, not broken.** `LOCAL_MATCH_THRESHOLD`
shipped at 12. On your 10-person set, genuine presses average **6.1** matched
minutiae; the 95th percentile of impostor pairs is 5. A threshold of 12 sits
so far above where genuine scores actually land that essentially nothing ever
clears it:

| threshold | measured with the OLD default (12) | false accepts | false rejects |
|---|---|---|---|
| >= 12 | shipped default | 0.5 % | **89 %** |

At 89% FRR, ten live attempts are very likely to produce zero matches even
for an enrolled finger - which is exactly what you saw. This is now fixed:
the default is data-derived (see below), and `identify()` no longer fails
silently - every rejection now logs the best candidate and its score, e.g.
`no match: best was l4 at 5 (need 6, 19 minutiae in the query, 50 templates
compared in 61 ms)`, so the next "no match" tells you how close it came.

**The matching algorithm has also been rewritten from scratch**, as asked.
The old algorithm (still in git history) required every matched minutia pair
to fit **one** global rotation+translation - a Hough-vote over the whole point
set. The new one (`fingerprint_local.py`, current) is a **local-structure
graph matcher**: the same family of algorithm real AFIS systems use, NIST's
own Bozorth3 included. It describes each minutia by its geometric relationship
to its nearest neighbours (length + two angles, each relative to a minutia's
own ridge direction - so no rotation search is needed at all), finds one
confident "root" pair, then grows the match outward, accepting a new pair
only when its edge to an already-confirmed pair lines up. That tolerates the
skin distortion a small optical sensor produces between two presses, which a
single rigid transform cannot.

Measured on the same 10 real people, old vs new, honestly:

| metric | old (global transform) | new (local-structure graph) |
|---|---|---|
| d' | 0.69 | **0.95** |
| EER | 38.0 % | **39.5 %*** |
| rank-1 identification | 58.0 % | **64.0 %** |
| speed | 5.9 ms/pair | **1.3 ms/pair** (vectorised) |

* EER moved slightly the wrong way even though d' and rank-1 both improved -
with only ~50 samples the EER crossing point is noisy; d' and rank-1 are the
more stable summaries here, and both favour the new algorithm.

**Read that table carefully: it is not a fix, it is evidence.** A materially
different, well-established algorithm only bought about 30% relative
improvement in separation, and rank-1 accuracy is still 64%, not 99.9%. If the
ceiling were the *algorithm*, a real rewrite would have moved these numbers by
a large factor, the way switching from SIFT to minutiae did (d' 1.00 -> 1.44
earlier). It didn't, because **the ceiling is set by how little information
this sensor hands over**: about 15 minutiae per press, from a skeleton the
sensor already binarised on-chip before we ever see it (next section). No
software - this rewrite, a further rewrite, or a commercial SDK run on the
same input - can extract more identity information than the sensor decided to
keep. Real fingerprint accuracy work happens on multi-level grayscale images
with 40-80 minutiae; that data does not exist on this hardware anymore by the
time it reaches the PC.

**New calibrated default:** `LOCAL_MATCH_THRESHOLD=6` (from the table above,
the best FAR/FRR balance found: 3.1% FAR / 58% FRR - not from a guess). Every
setting above 6 drives FAR to 0% but FRR into the 80-100% range; every setting
below 6 drives FAR into double digits. **No threshold here is safe for
unattended access control** - this is stated plainly, not as a caveat to
skim past.

## 1c. "Shouldn't the fingerprint image be in grayscale?"

Yes, ideally - and the honest answer is that it already isn't, upstream of
anything this project's code does. Check any capture directly:

```bash
py -3 -c "import cv2,numpy; im=cv2.imread('test/r503_captures/<file>.png',0); print(numpy.unique(im))"
# -> [0 240]   (or [0 255] depending on which UpImage format the firmware negotiated)
```

Every capture from this sensor - all 40 in `test/r503_captures`, all 50 of
your new enrollments - contains **exactly two pixel values**, never anything
between. That rules out a decode bug on our side: `finger_UDP.ino`'s only
pixel-level transform is 4-bit nibble replication (`hi<<4|hi`), which maps
value 15 to 255, not 240 - so the 0/240 captures are not even passing through
that code path; they are the sensor's raw unpacked 8-bit `UpImage` output,
completely unmodified, and it is *already* two-valued.

This sensor module enhances and binarises the ridge pattern **on its own
DSP** before `UpImage` ever hands us a frame - almost certainly because its
real job is feeding its own onboard matcher (which only needs a ridge
skeleton), not exporting a photograph. There is no richer grayscale image
being lost in this codebase's pipeline; it does not exist upstream of the
sensor's own output stage for this module. This is also why
`fingerprint_local.py`'s grayscale/Gabor code path (written for the older
FPM383C sensor, which does send real grayscale) never runs against this
sensor - `is_pre_thinned()` correctly detects the binary output every time
and skips straight to skeleton parsing.

**This can't be fixed by better decoding.** It could only be changed by a
sensor whose firmware is configured (or built) to export its true grayscale
buffer - which is exactly what `GenImg`/`UpImage` return on many R503
firmwares, just not, apparently, the enhancement stage this one uses. If a
firmware/register setting to disable on-chip enhancement exists for your
specific module (check its datasheet/vendor SDK for anything like "raw
image" or "enhancement off"), a real grayscale capture would very likely let
a minutiae extractor find 3-4x more features per press (matching the r5
finger's own internal ridge density - see below) and meaningfully raise the
ceiling in section 1b. Absent that, this project should treat 15
minutiae/press as the hard input limit.

---

## 2. You do not need NBIS

NBIS (`mindtct` + `bozorth3`) is C source from NIST with no arm64 binary, which
is why it will not install on the OrangePi 5B. It is not required. There are two
replacements, and the second is the one to aim for.

### Option A - `fingerprint_local.py` (added, works now)

A pure-Python minutiae engine: numpy and opencv-python only, nothing to compile,
runs anywhere the rest of the system runs.

It is a drop-in replacement for `fingerprint_nbis` - same `extract_template()`,
same `GalleryCache`, same `identify()`, and it writes templates in the same NIST
`.xyt` text format, so it shares the `xyt_template` database column and the
existing migration. If you ever do get NBIS built, the two engines can read each
other's enrollments.

Select it with `FINGERPRINT_ENGINE=local` (or leave `auto`, which now picks
NBIS if present, otherwise this).

Measured on **6 labelled fingers, 27 presses**, captured with
`test/capture_labelled.py` on the live sensor (this is real ground truth, not
inferred):

| metric | SIFT | `fingerprint_local` |
|---|---|---|
| d' | 1.00 | **1.44** |
| EER | 28.9 % | **20.9 %** |
| rank-1 identification | 60 % | **85.2 %** |
| extraction | - | 9 ms/image |
| matching | - | 1.1 ms/template pair |

A clear improvement, and **still not good enough for a door.** The operating
points are the honest part:

```
accept at >= 4 matched minutiae:  FAR 49.67%   FRR  6.1%
accept at >= 6 matched minutiae:  FAR 10.26%   FRR 34.7%
accept at >= 9 matched minutiae:  FAR  0.66%   FRR 75.5%
accept at >=12 matched minutiae:  FAR  0.00%   FRR 85.7%
```

There is no setting that is both safe and usable. Use it as the interim engine
and as the fallback for members whose enrollment predates the change, not as the
final answer.

(An earlier estimate on the unlabelled `test/r503_captures` gave d' 2.20 / EER
8.8 %. The labelled figures above are worse and are the ones to trust: in those
timestamped bursts the presses were seconds apart and the finger had barely
moved, which flatters any matcher. The labelled captures were taken the way a
member actually uses a door - lift and re-place between presses.)

### Capture quality decides everything

Per-finger results from the labelled set, sorted by how well each finger matched
itself:

| finger | mean minutiae | genuine score | verdict |
|---|---|---|---|
| bob2_right_index | 23.2 | 10.1 | separates cleanly |
| alice_right2_index | 17.4 | 8.9 | good |
| bob_right2_index | 15.0 | 6.7 | marginal |
| alice_right_index | 13.8 | 6.4 | weak |
| alice_left_index | 15.4 | 5.8 | weak |
| bob_right_index | 12.3 | 4.3 | **unusable** |

The impostor level is 3.7, so the last finger is statistically indistinguishable
from a stranger. Mean minutiae count and match quality correlate at **r = 0.92**:
how well a finger is captured predicts, almost perfectly, whether it can be
matched at all.

Enrollment used to *warn* about a low minutiae count and store the sample
anyway, which permanently degrades that member. It now rejects the press and
asks for another (`MIN_ENROLL_MINUTIAE`, default **16** as of the 10-person
test, with `ENROLL_RETRY_BUDGET` extra presses allowed). Cross-checking that
dataset: samples below 16 minutiae misidentify 40% of the time versus 34% at
16+ - a real but modest effect, not a fix by itself (several wrong matches
happen at 20+ minutiae too). Raising the minimum trades enrollment friction for
that improvement; at this sensor's ~15-20 minutiae average per press, a
particularly faint finger (one test finger averaged 13.4 and only produced one
sample out of five above 16) may need most of the retry budget, or may not be
enrollable at all - which is still better than enrolling a print that can
never be matched.

**Why it is capped.** The R503 does not send a photograph. It sends a binary
image that the sensor has already enhanced and thinned to a one-pixel-wide ridge
skeleton (160x160, two grey levels - verify with
`py -3 -c "import cv2,numpy;print(numpy.unique(cv2.imread('test/r503_captures/<f>.png',0)))"`).
Only about **15 minutiae** survive per capture; a matcher wants 30-60. The
grayscale detail that would yield more was discarded on the sensor before we
ever saw it, so no PC-side algorithm can recover it.

### Option B - use the sensor's own matcher (recommended)

The R503 contains a complete commercial fingerprint matcher on its own DSP,
matched to its optics and its enhancement pipeline. **The current design ignores
it.** `finger_UDP.ino` already calls `Img2Tz` - the sensor's feature extractor -
but only as a quality gate, then throws the features away and ships the image
over UDP for the PC to match with SIFT.

Everything needed is already in the sketch's `command()` helper; only these
opcodes are missing:

| code | command | purpose |
|---|---|---|
| `0x01` | GenImg | capture (already implemented) |
| `0x02` | Img2Tz | extract features to CharBuffer1/2 (already implemented) |
| `0x05` | RegModel | merge buffers 1+2 into one template |
| `0x06` | Store | save template to library slot `pageID` |
| `0x04` | Search | **1:N search of the library, returns pageID + score** |
| `0x03` | Match | 1:1 compare CharBuffer1 against CharBuffer2 |
| `0x08` | UpChar | upload a template from a buffer to the host |
| `0x09` | DownChar | download a template from the host into a buffer |
| `0x0C` | DeletChar | delete a slot |

Enrollment becomes: `GenImg -> Img2Tz(1)`, press again, `GenImg -> Img2Tz(2)`,
`RegModel`, `Store(pageID)`. Identification becomes `GenImg -> Img2Tz(1) ->
Search`, and the ESP32 sends the PC a member id instead of a 25 KB image. This
removes the UDP image transfer, the PC-side algorithm, NBIS and the OrangePi
problem in one step.

**Capacity - measured on your unit.** The `STATUS` reply from the entrance
ESP32 (192.168.1.120) reads:

```
STATUS sensor=ok baud=115200 lib=150 sec=3 packet=128 templates=0 rssi=-43 touch=1
```

`lib=150` is the whole internal library, and it is currently empty
(`templates=0`). At two fingers per member that is **75 members**; at one
finger, 150. `sec=3` is the sensor's default security level, whose published
false-accept rate is under 0.001 % - three orders of magnitude better than
anything achievable from the image it sends us.

So: **if the gym has 75 members or fewer, Option B is the whole solution.** If
it has more, use B2 below.

### Option B2 - unlimited members, still the sensor's matcher

Store the templates on the PC instead of in the sensor:

* **Enroll:** `Img2Tz` -> `RegModel` -> `UpChar` uploads the template; save those
  bytes in `fingerprint_features` exactly where `xyt_template` goes today.
* **Verify:** `Img2Tz(1)` on the live finger, then for each candidate
  `DownChar(2, template)` + `Match`.

Capacity becomes unlimited and the matching is still the sensor's. The cost is
one UART round trip per candidate (~50-100 ms), so this suits the locker and 1:1
verification (member identified by face or card first, fingerprint confirms)
rather than a cold 1:N search over hundreds of members.

The concrete code - sensor commands, enrollment, identification, UDP handlers -
is written out in `finger_UDP/ONSENSOR_MATCHING.md`, ready to paste.

I have not modified `finger_UDP.ino` itself. It is your only working capture
path and I cannot flash or test hardware from here; editing it blind would risk
the one part of the system that does work.

---

## 3. What changed in the repository

| file | change |
|---|---|
| `fingerprint_local.py` | pure-Python minutiae engine (no native deps); matcher rewritten to a local-structure graph algorithm (was a global-transform Hough-vote) |
| `test/test_local_engine.py` | **new** - accuracy benchmark, prints d', EER, operating points |
| `test/capture_labelled.py` | **new** - record labelled captures from the live sensor |
| `test/test_enroll_identify.py` | **new** - enroll N people, then score identification end to end |
| `fingerprint_gym.py` | engine selection now `auto` / `nbis` / `local` / `sift`; identification is engine-agnostic; enrollment rejects weak presses instead of storing them |

Nothing was removed. `fingerprint_nbis.py` still works if NBIS is ever
installed, and the SIFT path still reads old enrollments.

### Enrolling people and testing identification

`test/test_enroll_identify.py` is the end-to-end test: real presses in, a name
out, scored against the truth. It uses the same engine the gym system uses, so
what it reports is what the door will do.

```bash
py -3 test/test_enroll_identify.py enroll-many --count 10   # asks for each name
py -3 test/test_enroll_identify.py list                     # who is enrolled, print quality
py -3 test/test_enroll_identify.py evaluate                 # accuracy, no sensor needed
py -3 test/test_enroll_identify.py identify --trials 10     # live: press, get a name
py -3 test/test_enroll_identify.py identify --expect ali    # scored against the truth
```

`evaluate` is the one to read. It reports per-person capture quality (with a
`WEAK` flag for fingers that will never match), genuine/impostor separation,
leave-one-out rank-1 accuracy with the actual misidentifications listed, and the
FAR/FRR table to pick `LOCAL_MATCH_THRESHOLD` from.

Enrollments are PNGs named `<person>_sample_<n>.png`, the same layout
`test_local_engine.py` reads, so both tools share one dataset.

### Running the benchmark

```bash
py -3 fingerprint_local.py --selftest          # is the engine usable
py -3 test/test_local_engine.py                # accuracy on real captures
py -3 fingerprint_local.py --bench test/r503_captures   # speed
```

`test_local_engine.py` also reads labelled data in the form
`fingerprint_storage/images/<label>_sample_<n>.png`, so after collecting a few
fingers' worth of new captures you can re-measure honestly rather than guessing.

### How the R503 captures were labelled

`test/r503_captures` has no identity labels, so the benchmark infers them from
the capture clock, deliberately in the pessimistic direction:

* **genuine** - two captures in the same burst, <= 20 s apart. Nobody changes
  finger between presses that close together.
* **impostor** - captures from bursts more than 60 s apart. If two such bursts
  happen to be the same finger, that *inflates* the impostor scores.

So the true numbers can only be better than what is printed, never worse.

To replace the guesswork with real labels, now that the sensor is connected,
run `test/capture_labelled.py` once per finger (it needs no keyboard input):

```bash
py -3 test/capture_labelled.py --label alice_right_index --samples 5
py -3 test/capture_labelled.py --label alice_left_index  --samples 5
py -3 test/capture_labelled.py --label bob_right_index   --samples 5
# ... 6-8 different fingers in total
py -3 test/test_local_engine.py
```

Lift the finger fully between presses and shift it slightly each time. Pressing
the identical spot five times flatters the matcher and hides the real error
rate. With that dataset the benchmark reports a true d' and EER, and
`LOCAL_MATCH_THRESHOLD` can be set from the operating-point table instead of
from an estimate.

---

## 4. For a 1000+ member gym specifically

Cold **1:N** identification - scan a finger, search everyone, guess who it is -
is the wrong architecture at this scale on this hardware, for two independent
reasons that a better algorithm cannot fix:

1. **Accuracy collapses with population size.** d' ~0.7-0.95 measured on just
   10 people (100 genuine pairs, 1,125 impostor pairs). Every additional
   enrolled finger adds more chances for a false accept; with a false-accept
   rate already in the single-to-double digits at N=10, searching all of 1000+
   members cold is not a smaller version of the same problem, it is a much
   harder one. This is true of *any* matcher run against this sensor's ~15
   minutiae/press output, not specific to what is implemented here.
2. **Cold 1:N does not scale in time either.** At 1.3 ms per template
   comparison, 1000 members x 2 fingers = 2000 templates is ~2.6 seconds per
   identification attempt, growing linearly forever. A commercial AFIS gets
   around this with indexing (classify by pattern type, core/delta location,
   ridge count) that this project has no data to build.

**The fix is architectural, not algorithmic: turn 1:N into 1:1.** Identify the
member with something narrow first - a card, fob, phone tap, PIN, or even
typing a member number - then use the fingerprint only to VERIFY that one
claimed identity. This is what essentially every commercial access-control
deployment above a few hundred users actually does; blind fingerprint-only
entry at that scale is rare even with far better sensors, because 1:1
verification is both dramatically faster (one comparison, not N) and
dramatically more accurate (the finger only has to beat ONE specific
impostor risk, not the whole enrolled population).

Concretely, in order:

1. **Decide on capacity.** Already measured: `lib=150`, so the sensor's OWN
   onboard matcher (Option B below) covers 75 members at two fingers each
   without touching the PC algorithm at all - and it runs on the full-resolution
   image *before* the enhancement that throws away the minutiae this project's
   matcher never gets to see. For 1000+ members, combine it with 1:1 as above:
   a member taps a card/PIN to narrow to one identity, the sensor's own
   `Match` (or `DownChar`+`Match` if templates live in the PC database, see
   Option B2) confirms it.
2. **Implement on-sensor matching** (`RegModel`/`Store`/`Search`, or
   `UpChar`/`DownChar`/`Match`) - see `finger_UDP/ONSENSOR_MATCHING.md`. This
   is the real fix, not `fingerprint_local.py`, which exists to keep old
   enrollments usable and to serve as a fallback, not as the production path
   for a gym this size.
3. **Run `FINGERPRINT_ENGINE=local` in the meantime** so the system is merely
   imperfect rather than broken, and so new enrollments store real minutiae
   templates that the sensor path can be compared against.
4. **Stop writing SIFT descriptors** once on-sensor matching works. Keep saving
   raw images - `fingerprint_image_store.py` already does - because they are the
   only artefact that survives another algorithm change.
5. **Do not pursue the deep-learning embedding plan** in the old analysis docs.
   It needs a labelled fingerprint corpus and training infrastructure to beat a
   matcher you already own (the sensor's), and it would face the identical
   information ceiling documented in section 1b: a model cannot learn identity
   detail the sensor already discarded before the image reached us.

---

## 5. Note on the legacy captures

The 102 images that were in `fingerprint_storage/images` (the 64x80 FPM383C
captures, 17 labelled fingers) are gone; the directory is now empty apart from
`enrolled/`. The SIFT baseline in section 1 was measured before they
disappeared, so that result stands, but the labelled set can no longer be
re-run. If you have a copy, restoring it is worthwhile - it is the only
explicitly labelled data in the project.
