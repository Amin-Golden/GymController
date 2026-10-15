-- ========================================
-- RFID card entrance
-- ========================================
-- Adds the card ID column used by rfid_gym.py. Safe to run more than once.
--
-- IDs are stored normalised: upper-case, separators removed, exactly as the
-- entrance ESP32 sends them (an RDM6300 card is 10 hex chars, e.g. '0F00A1B2C3').
--
-- Link a card to a member by having them show it at the entrance after:
--     NOTIFY client_changes, '{"action": "RFID_ENROLL", "client_id": 42}';
-- or set it directly:
--     UPDATE clients SET rfid_uid = '0F00A1B2C3' WHERE id = 42;
-- Remove a lost card:
--     UPDATE clients SET rfid_uid = NULL WHERE id = 42;

ALTER TABLE public.clients ADD COLUMN IF NOT EXISTS rfid_uid character varying(32);

-- One card, one member
CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_rfid_uid
    ON public.clients(rfid_uid) WHERE rfid_uid IS NOT NULL;
