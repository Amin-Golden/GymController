-- Migration: grayscale SIFT fingerprint engine (fingerprint_sift.py).
--
-- Run once, before enrolling with the new firmware/engine:
--     psql -U <user> -d <database> -f migrate_fingerprint_sift_gray.sql
--
-- Adds:
--   sift_gray   per press: SIFT keypoints (x, y, angle) + uint8 descriptors from
--               the sensor's grayscale image, packed by fingerprint_sift.pack()
--               (~75 KB). Presses without it belong to the old engines and are
--               not matched any more - those members enroll again.
--   sensor_template  the R503's own 2 KB template of the press, kept for a few
--               presses per finger: the sensor double-check of borderline
--               matches sends it back to the sensor's matcher
--   image_path  path of the raw capture on disk (if an earlier migration did
--               not add it already), so a future algorithm can be rebuilt
--               without asking members to enroll again
--
-- Nothing is dropped. Safe to run more than once - run it again if you ran an
-- earlier version (before sensor_template existed).

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'fingerprint_features'
                     AND column_name = 'sift_gray') THEN
        ALTER TABLE public.fingerprint_features ADD COLUMN sift_gray bytea;
        RAISE NOTICE 'Added column sift_gray';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'fingerprint_features'
                     AND column_name = 'sensor_template') THEN
        ALTER TABLE public.fingerprint_features ADD COLUMN sensor_template bytea;
        RAISE NOTICE 'Added column sensor_template';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'fingerprint_features'
                     AND column_name = 'image_path') THEN
        ALTER TABLE public.fingerprint_features ADD COLUMN image_path text;
        RAISE NOTICE 'Added column image_path';
    END IF;
END $$;

COMMENT ON COLUMN public.fingerprint_features.sift_gray IS
    'Grayscale SIFT keypoints + descriptors of one press (fingerprint_sift.pack)';
COMMENT ON COLUMN public.fingerprint_features.sensor_template IS
    'R503 template of the press (2 KB) - sensor double-check of borderline matches';

-- Re-enrollment progress (safe to run any time):
SELECT COUNT(DISTINCT client_id) FILTER (WHERE sift_gray IS NOT NULL) AS members_enrolled_new,
       COUNT(DISTINCT client_id) FILTER (WHERE sift_gray IS NULL)     AS members_with_old_presses,
       COUNT(*) FILTER (WHERE sift_gray IS NOT NULL)                  AS new_presses,
       COUNT(*) FILTER (WHERE sensor_template IS NOT NULL)            AS presses_with_sensor_template
FROM public.fingerprint_features;
