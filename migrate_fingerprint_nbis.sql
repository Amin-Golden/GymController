-- Migration: NBIS (MINDTCT + BOZORTH3) support and raw image retention.
--
-- Run once before enabling the NBIS engine:
--     psql -U <user> -d <database> -f migrate_fingerprint_nbis.sql
--
-- Adds:
--   xyt_template    MINDTCT minutiae template (ASCII "x y theta quality" lines)
--   minutiae_count  number of minutiae in that template (quality indicator)
--   image_path      path of the raw capture on disk, so a future algorithm can
--                   be rebuilt without asking members to enroll again
--   nbis_updated_at when the template was generated
--
-- Nothing is dropped: the existing descriptors/keypoints_data columns used by
-- the SIFT engine stay exactly as they are.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'fingerprint_features'
                     AND column_name = 'xyt_template') THEN
        ALTER TABLE public.fingerprint_features ADD COLUMN xyt_template text;
        RAISE NOTICE 'Added column xyt_template';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'fingerprint_features'
                     AND column_name = 'minutiae_count') THEN
        ALTER TABLE public.fingerprint_features ADD COLUMN minutiae_count integer;
        RAISE NOTICE 'Added column minutiae_count';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'fingerprint_features'
                     AND column_name = 'image_path') THEN
        ALTER TABLE public.fingerprint_features ADD COLUMN image_path text;
        RAISE NOTICE 'Added column image_path';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'fingerprint_features'
                     AND column_name = 'nbis_updated_at') THEN
        ALTER TABLE public.fingerprint_features ADD COLUMN nbis_updated_at timestamp;
        RAISE NOTICE 'Added column nbis_updated_at';
    END IF;
END $$;

-- Identification loads every template that exists; this index keeps that scan cheap.
CREATE INDEX IF NOT EXISTS idx_fingerprint_features_xyt
    ON public.fingerprint_features (client_id, finger_index)
    WHERE xyt_template IS NOT NULL;

COMMENT ON COLUMN public.fingerprint_features.xyt_template IS
    'MINDTCT minutiae template (.xyt text) used by BOZORTH3 matching';
COMMENT ON COLUMN public.fingerprint_features.minutiae_count IS
    'Number of minutiae extracted by MINDTCT (low counts indicate a poor capture)';
COMMENT ON COLUMN public.fingerprint_features.image_path IS
    'Raw enrollment image on disk - lets the matching algorithm be changed without re-enrollment';

-- How far along the migration is (safe to run any time):
SELECT COUNT(*) AS samples,
       COUNT(xyt_template) AS with_template,
       COUNT(image_path) AS with_image,
       COUNT(DISTINCT client_id) AS clients
FROM public.fingerprint_features;
