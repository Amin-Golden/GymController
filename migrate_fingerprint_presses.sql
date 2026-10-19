-- ========================================
-- Allow more than 5 presses per finger + the whole-finger template
-- ========================================
-- Enrollment now takes 8 presses per finger (ENROLL_PRESSES) and stores the
-- stitched whole-finger template as sample_number 0. If fingerprint_features
-- was created with a CHECK limiting sample_number (e.g. 1..5), every save of a
-- finger is refused and the member shows "Finger N enrollment failed".
-- Until this runs, enrollment falls back to storing the best 5 presses.
--
-- 1. Look first - lists every constraint on the table:
--      SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint
--      WHERE conrelid = 'public.fingerprint_features'::regclass;
--
-- 2. Run this file. It drops only CHECK constraints that mention sample_number,
--    and prints each one it drops (pgAdmin: Messages tab). Safe to run again.
--
-- If your admin app shows samples per finger, check it copes with sample 0 and
-- samples 6-8 (they hold no SIFT data, only a minutiae template).

DO $$
DECLARE
    c record;
BEGIN
    FOR c IN
        SELECT conname, pg_get_constraintdef(oid) AS def
        FROM pg_constraint
        WHERE conrelid = 'public.fingerprint_features'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) ILIKE '%sample_number%'
    LOOP
        RAISE NOTICE 'Dropping constraint %: %', c.conname, c.def;
        EXECUTE format('ALTER TABLE public.fingerprint_features DROP CONSTRAINT %I', c.conname);
    END LOOP;
END $$;
