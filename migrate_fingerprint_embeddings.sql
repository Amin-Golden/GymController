-- Migration script to add embedding column to fingerprint_features table
-- Run this script on your database before using the improved fingerprint processor

-- Add embedding column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'fingerprint_features' 
        AND column_name = 'embedding'
    ) THEN
        ALTER TABLE public.fingerprint_features 
        ADD COLUMN embedding bytea;
        
        RAISE NOTICE 'Column embedding added to fingerprint_features table';
    ELSE
        RAISE NOTICE 'Column embedding already exists in fingerprint_features table';
    END IF;
END $$;

-- Add index for faster queries (optional, but recommended)
CREATE INDEX IF NOT EXISTS idx_fingerprint_features_embedding 
ON public.fingerprint_features(client_id, finger_index) 
WHERE embedding IS NOT NULL;

-- Add comment
COMMENT ON COLUMN public.fingerprint_features.embedding IS 
'Stores aggregated fingerprint embedding (128-dim vector) for fast similarity search';

