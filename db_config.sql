-- Option 1: Using pgAdmin

-- Open pgAdmin
-- Connect to your database
-- Right-click on your database → Query Tool
-- Paste the entire script above
-- Click Execute (F5)
-- psql -U postgres -d gym_db -f database_setup.sql

-- ========================================
-- 1. CREATE GYM SESSIONS TABLE
-- ========================================
-- This table tracks entrance and exit times for gym visits
CREATE TABLE IF NOT EXISTS public.gym_sessions
(
    id bigserial NOT NULL,
    client_id bigint NOT NULL,
    entrance_time timestamp without time zone NOT NULL,
    exit_time timestamp without time zone,
    locker_number integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_gym_sessions_client FOREIGN KEY (client_id) 
        REFERENCES public.clients(id) ON DELETE CASCADE
);

-- Add indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_gym_sessions_client_id ON public.gym_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_gym_sessions_entrance_time ON public.gym_sessions(entrance_time);
CREATE INDEX IF NOT EXISTS idx_gym_sessions_exit_time ON public.gym_sessions(exit_time);
CREATE INDEX IF NOT EXISTS idx_gym_sessions_active ON public.gym_sessions(client_id, exit_time) 
    WHERE exit_time IS NULL;

-- Add comment
COMMENT ON TABLE public.gym_sessions IS 'Tracks gym entrance and exit times with locker assignments';


-- ========================================
-- 2. CREATE FACE EMBEDDINGS TABLE
-- ========================================
-- This table stores face recognition embeddings for each client
CREATE TABLE IF NOT EXISTS public.face_embeddings 
(
    id bigserial NOT NULL,
    client_id bigint NOT NULL,
    embedding bytea NOT NULL,
    confidence float,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_face_embeddings_client FOREIGN KEY (client_id) 
        REFERENCES public.clients(id) ON DELETE CASCADE,
    CONSTRAINT unique_client_embedding UNIQUE (client_id)
);

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_face_embeddings_client_id ON public.face_embeddings(client_id);

-- Add comment
COMMENT ON TABLE public.face_embeddings IS 'Stores face recognition embeddings for client authentication';


-- ========================================
-- 3. CREATE ACCESS LOGS TABLE
-- ========================================
-- This table logs all face recognition attempts (successful and failed)
CREATE TABLE IF NOT EXISTS public.access_logs 
(
    id bigserial NOT NULL,
    client_id bigint,
    access_granted boolean NOT NULL,
    confidence float,
    timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_access_logs_client FOREIGN KEY (client_id) 
        REFERENCES public.clients(id) ON DELETE SET NULL
);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_access_logs_client_id ON public.access_logs(client_id);
CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON public.access_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_access_logs_granted ON public.access_logs(access_granted);

-- Add comment
COMMENT ON TABLE public.access_logs IS 'Logs all face recognition access attempts';


-- ========================================
-- 4. MODIFY CLIENTS TABLE (if needed)
-- ========================================
-- Ensure clients table has locker column
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'clients' AND column_name = 'locker'
    ) THEN
        ALTER TABLE public.clients ADD COLUMN locker integer;
    END IF;
END $$;

-- Add index on locker for faster queries 
CREATE INDEX IF NOT EXISTS idx_clients_locker ON public.clients(locker) WHERE locker IS NOT NULL;


-- ========================================
-- 5. CREATE NOTIFICATION FUNCTION
-- ========================================
-- This function sends notifications when clients are added or updated
CREATE OR REPLACE FUNCTION notify_client_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify with client ID and image path
    IF (TG_OP = 'INSERT') THEN
        PERFORM pg_notify('client_changes', json_build_object(
            'action', 'INSERT',
            'client_id', NEW.id,
            'image_path', NEW.image_path,
            'fname', NEW.fname,
            'lname', NEW.lname
        )::text);
        RETURN NEW;
    ELSIF (TG_OP = 'UPDATE') THEN
        -- Only notify if image_path changed
        IF (OLD.image_path IS DISTINCT FROM NEW.image_path) THEN
            PERFORM pg_notify('client_changes', json_build_object(
                'action', 'UPDATE',
                'client_id', NEW.id,
                'image_path', NEW.image_path,
                'fname', NEW.fname,
                'lname', NEW.lname
            )::text);
        END IF;
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Add comment
COMMENT ON FUNCTION notify_client_change() IS 'Sends PostgreSQL notification when client is inserted or updated';


-- ========================================
-- 6. CREATE TRIGGER FOR CLIENT CHANGES
-- ========================================
-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS client_change_trigger ON public.clients;

-- Create trigger
CREATE TRIGGER client_change_trigger
AFTER INSERT OR UPDATE ON public.clients
FOR EACH ROW
EXECUTE FUNCTION notify_client_change();


-- ========================================
-- 7. CREATE INDEXES ON MEMBERSHIPS TABLE
-- ========================================
-- Optimize membership queries
CREATE INDEX IF NOT EXISTS idx_memberships_client_id ON public.memberships(client_id);
CREATE INDEX IF NOT EXISTS idx_memberships_end_date ON public.memberships(end_date);
CREATE INDEX IF NOT EXISTS idx_memberships_is_paid ON public.memberships(is_paid);
CREATE INDEX IF NOT EXISTS idx_memberships_active ON public.memberships(client_id, end_date, is_paid) 
    WHERE is_paid = TRUE AND end_date >= CURRENT_DATE;


-- ========================================
-- 8. GRANT PERMISSIONS TO gym_user
-- ========================================
-- Grant all necessary permissions to gym_user

-- Tables
GRANT ALL PRIVILEGES ON TABLE public.clients TO gym_user;
GRANT ALL PRIVILEGES ON TABLE public.memberships TO gym_user;
GRANT ALL PRIVILEGES ON TABLE public.gym_sessions TO gym_user;
GRANT ALL PRIVILEGES ON TABLE public.face_embeddings TO gym_user;
GRANT ALL PRIVILEGES ON TABLE public.access_logs TO gym_user;

-- Sequences
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO gym_user;

-- Specific sequences (if the above doesn't work)
GRANT USAGE, SELECT ON SEQUENCE public.clients_id_seq TO gym_user;
GRANT USAGE, SELECT ON SEQUENCE public.memberships_id_seq TO gym_user;
GRANT USAGE, SELECT ON SEQUENCE public.gym_sessions_id_seq TO gym_user;
GRANT USAGE, SELECT ON SEQUENCE public.face_embeddings_id_seq TO gym_user;
GRANT USAGE, SELECT ON SEQUENCE public.access_logs_id_seq TO gym_user;


-- ========================================
-- 9. USEFUL VIEWS (OPTIONAL)
-- ========================================

-- View: Active Gym Sessions
CREATE OR REPLACE VIEW public.active_gym_sessions AS
SELECT 
    gs.id,
    gs.client_id,
    c.fname,
    c.lname,
    c.locker,
    gs.entrance_time,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - gs.entrance_time))/60 AS duration_minutes
FROM public.gym_sessions gs
JOIN public.clients c ON gs.client_id = c.id
WHERE gs.exit_time IS NULL
ORDER BY gs.entrance_time DESC;

GRANT SELECT ON public.active_gym_sessions TO gym_user;

-- View: Today's Gym Activity
CREATE OR REPLACE VIEW public.today_gym_activity AS
SELECT 
    gs.id,
    c.fname,
    c.lname,
    gs.entrance_time,
    gs.exit_time,
    gs.locker_number,
    CASE 
        WHEN gs.exit_time IS NULL THEN 'ACTIVE'
        ELSE 'COMPLETED'
    END as status,
    CASE 
        WHEN gs.exit_time IS NOT NULL THEN 
            EXTRACT(EPOCH FROM (gs.exit_time - gs.entrance_time))/60
        ELSE 
            EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - gs.entrance_time))/60
    END as duration_minutes
FROM public.gym_sessions gs
JOIN public.clients c ON gs.client_id = c.id
WHERE DATE(gs.entrance_time) = CURRENT_DATE
ORDER BY gs.entrance_time DESC;

GRANT SELECT ON public.today_gym_activity TO gym_user;

-- View: Clients with Active Memberships
CREATE OR REPLACE VIEW public.active_members AS
SELECT 
    c.id,
    c.fname,
    c.lname,
    c.email,
    c.phone_number,
    c.locker,
    m.start_date,
    m.end_date,
    m.status,
    (m.end_date - CURRENT_DATE) as days_remaining,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM gym_sessions gs 
            WHERE gs.client_id = c.id AND gs.exit_time IS NULL
        ) THEN 'IN_GYM'
        ELSE 'OUT'
    END as current_status
FROM public.clients c
JOIN public.memberships m ON c.id = m.client_id
WHERE m.is_paid = TRUE 
    AND m.end_date >= CURRENT_DATE
ORDER BY c.lname, c.fname;

GRANT SELECT ON public.active_members TO gym_user;


-- ========================================
-- 10. HELPER FUNCTIONS (OPTIONAL)
-- ========================================

-- Function: Get available lockers count
CREATE OR REPLACE FUNCTION get_available_lockers_count()
RETURNS integer AS $$
DECLARE
    total_lockers integer := 200;
    assigned_count integer;
BEGIN
    SELECT COUNT(*) INTO assigned_count
    FROM clients
    WHERE locker IS NOT NULL;
    
    RETURN total_lockers - assigned_count;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION get_available_lockers_count() TO gym_user;


-- Function: Get gym occupancy
CREATE OR REPLACE FUNCTION get_current_gym_occupancy()
RETURNS TABLE (
    total_in_gym integer,
    total_lockers_used integer,
    active_sessions json
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::integer as total_in_gym,
        COUNT(DISTINCT locker_number)::integer as total_lockers_used,
        json_agg(json_build_object(
            'client_id', gs.client_id,
            'name', c.fname || ' ' || c.lname,
            'entrance_time', gs.entrance_time,
            'locker', gs.locker_number
        )) as active_sessions
    FROM gym_sessions gs
    JOIN clients c ON gs.client_id = c.id
    WHERE gs.exit_time IS NULL;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION get_current_gym_occupancy() TO gym_user;


-- ========================================
-- 11. CLEANUP OLD DATA (OPTIONAL)
-- ========================================

-- Function: Archive old gym sessions (older than 1 year)
CREATE OR REPLACE FUNCTION archive_old_gym_sessions()
RETURNS integer AS $$
DECLARE
    deleted_count integer;
BEGIN
    -- You might want to move to archive table instead of deleting
    DELETE FROM gym_sessions
    WHERE entrance_time < CURRENT_DATE - INTERVAL '1 year'
        AND exit_time IS NOT NULL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION archive_old_gym_sessions() TO gym_user;


-- Function: Cleanup old access logs (older than 6 months)
CREATE OR REPLACE FUNCTION cleanup_old_access_logs()
RETURNS integer AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM access_logs
    WHERE timestamp < CURRENT_DATE - INTERVAL '6 months';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION cleanup_old_access_logs() TO gym_user;


-- ========================================
-- 12. VERIFICATION QUERIES
-- ========================================

-- Check if all tables exist
DO $$
BEGIN
    RAISE NOTICE 'Verifying database setup...';
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'clients') THEN
        RAISE NOTICE '✓ clients table exists';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'memberships') THEN
        RAISE NOTICE '✓ memberships table exists';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'gym_sessions') THEN
        RAISE NOTICE '✓ gym_sessions table exists';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'face_embeddings') THEN
        RAISE NOTICE '✓ face_embeddings table exists';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'access_logs') THEN
        RAISE NOTICE '✓ access_logs table exists';
    END IF;
    
    RAISE NOTICE 'Database setup complete!';
END $$;