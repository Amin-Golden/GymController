-- ========================================
-- GYM DATABASE INITIALIZATION SCRIPT
-- ========================================

BEGIN;

-- Create public schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS public;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;

-- Set search path for database
-- ALTER DATABASE gym-db SET search_path TO public;
-- ========================================
-- DROP EXISTING OBJECTS (Optional - Uncomment if you want fresh install)
-- ========================================
-- DROP TABLE IF EXISTS public.attendance CASCADE;
-- DROP TABLE IF EXISTS public.sessions CASCADE;
-- DROP TABLE IF EXISTS public.access_logs CASCADE;
-- DROP TABLE IF EXISTS public.gym_sessions CASCADE;
-- DROP TABLE IF EXISTS public.face_embeddings CASCADE;
-- DROP TABLE IF EXISTS public.payments CASCADE;
-- DROP TABLE IF EXISTS public.memberships CASCADE;
-- DROP TABLE IF EXISTS public.instructors CASCADE;
-- DROP TABLE IF EXISTS public.packages CASCADE;
-- DROP TABLE IF EXISTS public.clients CASCADE;
-- DROP TABLE IF EXISTS public.admins CASCADE;

-- ========================================
-- CREATE TABLES
-- ========================================

CREATE TABLE IF NOT EXISTS public.clients
(
    id bigserial NOT NULL,
    fname character varying(50) NOT NULL,
    lname character varying(50),
    dob date NOT NULL,
    is_male boolean NOT NULL,
    email character varying(50),
    phone_number character varying(13) NOT NULL,
    social_number character varying(10) NOT NULL,
    image_path text,
    description text,
    locker integer,
    weight real,
    height real,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.packages
(
    id bigserial NOT NULL,
    package_name text NOT NULL,
    image_path text,
    duration text NOT NULL,
    price integer NOT NULL,
    days integer NOT NULL,
    description text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.instructors
(
    id bigserial NOT NULL,
    package_id bigint NOT NULL,
    fname character varying(50) NOT NULL,
    lname character varying(50),
    dob date NOT NULL,
    is_male boolean NOT NULL,
    salary real NOT NULL,
    email character varying(50),
    title text NOT NULL,
    description text,
    phone_number character varying(13) NOT NULL,
    image_path text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.memberships
(
    id bigserial NOT NULL,
    client_id bigint NOT NULL,
    package_id bigint NOT NULL,
    instructor_id bigint NOT NULL,
    status text NOT NULL,
    start_date date NOT NULL,
    end_date date NOT NULL,
    payment_date timestamp without time zone NOT NULL,
    is_paid boolean NOT NULL,
    description text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    RemainSessions INTEGER NOT NULL DEFAULT 0;
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.payments
(
    id bigserial NOT NULL,
    client_id bigint NOT NULL,
    payment_type character varying(10) NOT NULL,
    description text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.admins
(
    id bigserial NOT NULL,
    adminID character varying(50) NOT NULL,
    fname character varying(50) NOT NULL,
    lname character varying(50) NOT NULL,
    dob date NOT NULL,
    is_male boolean NOT NULL,
    password text NOT NULL,
    phone_number character varying(13) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.sessions
(
    id bigserial NOT NULL,
    instructor_id bigint NOT NULL,
    membership_id bigint NOT NULL,
    destination_date date NOT NULL,
    is_attended boolean NOT NULL,
    description text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.attendance
(
    id bigserial NOT NULL,
    client_id bigint NOT NULL,
    session_id bigint NOT NULL,
    description text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.gym_sessions
(
    id bigserial NOT NULL,
    client_id bigint NOT NULL,
    entrance_time timestamp without time zone NOT NULL,
    exit_time timestamp without time zone,
    locker_number integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS public.face_embeddings
(
    id bigserial NOT NULL,
    client_id bigint NOT NULL,
    embedding bytea NOT NULL,
    confidence real,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT unique_client_embedding UNIQUE (client_id)
);

CREATE TABLE IF NOT EXISTS public.access_logs
(
    id bigserial NOT NULL,
    client_id bigint,
    access_granted boolean NOT NULL,
    confidence real,
    timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

-- ========================================
-- ADD FOREIGN KEY CONSTRAINTS
-- ========================================

DO $$ 
BEGIN
    -- Add foreign keys only if they don't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_gym_sessions_client'
    ) THEN
        ALTER TABLE public.gym_sessions
            ADD CONSTRAINT fk_gym_sessions_client FOREIGN KEY (client_id)
            REFERENCES public.clients(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_face_embeddings_client'
    ) THEN
        ALTER TABLE public.face_embeddings
            ADD CONSTRAINT fk_face_embeddings_client FOREIGN KEY (client_id)
            REFERENCES public.clients(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_access_logs_client'
    ) THEN
        ALTER TABLE public.access_logs
            ADD CONSTRAINT fk_access_logs_client FOREIGN KEY (client_id)
            REFERENCES public.clients(id) ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_memberships_client'
    ) THEN
        ALTER TABLE public.memberships
            ADD CONSTRAINT fk_memberships_client FOREIGN KEY (client_id)
            REFERENCES public.clients(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_memberships_package'
    ) THEN
        ALTER TABLE public.memberships
            ADD CONSTRAINT fk_memberships_package FOREIGN KEY (package_id)
            REFERENCES public.packages(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_memberships_instructor'
    ) THEN
        ALTER TABLE public.memberships
            ADD CONSTRAINT fk_memberships_instructor FOREIGN KEY (instructor_id)
            REFERENCES public.instructors(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_payments_client'
    ) THEN
        ALTER TABLE public.payments
            ADD CONSTRAINT fk_payments_client FOREIGN KEY (client_id)
            REFERENCES public.clients(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_attendance_client'
    ) THEN
        ALTER TABLE public.attendance
            ADD CONSTRAINT fk_attendance_client FOREIGN KEY (client_id)
            REFERENCES public.clients(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_attendance_session'
    ) THEN
        ALTER TABLE public.attendance
            ADD CONSTRAINT fk_attendance_session FOREIGN KEY (session_id)
            REFERENCES public.sessions(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_sessions_instructor'
    ) THEN
        ALTER TABLE public.sessions
            ADD CONSTRAINT fk_sessions_instructor FOREIGN KEY (instructor_id)
            REFERENCES public.instructors(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_sessions_membership'
    ) THEN
        ALTER TABLE public.sessions
            ADD CONSTRAINT fk_sessions_membership FOREIGN KEY (membership_id)
            REFERENCES public.memberships(id) ON DELETE CASCADE;
    END IF;
END $$;

-- ========================================
-- CREATE INDEXES
-- ========================================

CREATE INDEX IF NOT EXISTS idx_gym_sessions_client_id ON public.gym_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_gym_sessions_entrance_time ON public.gym_sessions(entrance_time);
CREATE INDEX IF NOT EXISTS idx_gym_sessions_exit_time ON public.gym_sessions(exit_time);
CREATE INDEX IF NOT EXISTS idx_gym_sessions_active ON public.gym_sessions(client_id, exit_time) WHERE exit_time IS NULL;

CREATE INDEX IF NOT EXISTS idx_face_embeddings_client_id ON public.face_embeddings(client_id);

CREATE INDEX IF NOT EXISTS idx_access_logs_client_id ON public.access_logs(client_id);
CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON public.access_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_access_logs_granted ON public.access_logs(access_granted);

CREATE INDEX IF NOT EXISTS idx_clients_locker ON public.clients(locker) WHERE locker IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memberships_client_id ON public.memberships(client_id);
CREATE INDEX IF NOT EXISTS idx_memberships_end_date ON public.memberships(end_date);
CREATE INDEX IF NOT EXISTS idx_memberships_is_paid ON public.memberships(is_paid);
-- Fixed: Removed CURRENT_DATE from index predicate
CREATE INDEX IF NOT EXISTS idx_memberships_active ON public.memberships(client_id, end_date, is_paid) WHERE is_paid = TRUE;

-- ========================================
-- CREATE ROLE AND GRANT PERMISSIONS
-- ========================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'postgres') THEN
        CREATE ROLE postgres LOGIN PASSWORD '123456';
        RAISE NOTICE 'Role postgres created';
    ELSE
        RAISE NOTICE 'Role postgres already exists';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE public.clients TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.packages TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.instructors TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.memberships TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.payments TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.admins TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.sessions TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.attendance TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.gym_sessions TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.face_embeddings TO postgres;
GRANT ALL PRIVILEGES ON TABLE public.access_logs TO postgres;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- ========================================
-- CREATE NOTIFICATION FUNCTION
-- ========================================
-- ========================================
-- CREATE NOTIFICATION FUNCTION (UPDATED)
-- ========================================

CREATE OR REPLACE FUNCTION notify_client_change()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        PERFORM pg_notify('client_changes', json_build_object(
            'action', 'INSERT',
            'client_id', NEW.id,
            'image_path', NEW.image_path,
            'fname', NEW.fname,
            'lname', NEW.lname,
            'image_changed', true
        )::text);
        RETURN NEW;
        
    ELSIF (TG_OP = 'DELETE') THEN
        PERFORM pg_notify('client_changes', json_build_object(
            'action', 'DELETE',
            'client_id', OLD.id
        )::text);
        RETURN OLD;
    END IF;
    
    -- No notification for UPDATE
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- CREATE TRIGGER
-- ========================================

DROP TRIGGER IF EXISTS client_change_trigger ON public.clients;
CREATE TRIGGER client_change_trigger
AFTER INSERT OR UPDATE ON public.clients
FOR EACH ROW
EXECUTE FUNCTION notify_client_change();

-- ========================================
-- CREATE VIEWS
-- ========================================

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

GRANT SELECT ON public.active_gym_sessions TO postgres;

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

GRANT SELECT ON public.today_gym_activity TO postgres;

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

GRANT SELECT ON public.active_members TO postgres;

-- ========================================
-- CREATE HELPER FUNCTIONS
-- ========================================

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

GRANT EXECUTE ON FUNCTION get_available_lockers_count() TO postgres;

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

GRANT EXECUTE ON FUNCTION get_current_gym_occupancy() TO postgres;

CREATE OR REPLACE FUNCTION archive_old_gym_sessions()
RETURNS integer AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM gym_sessions
    WHERE entrance_time < CURRENT_DATE - INTERVAL '1 year'
        AND exit_time IS NOT NULL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

GRANT EXECUTE ON FUNCTION archive_old_gym_sessions() TO postgres;

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

GRANT EXECUTE ON FUNCTION cleanup_old_access_logs() TO postgres;

-- ========================================
-- ADD COMMENTS
-- ========================================

COMMENT ON TABLE public.gym_sessions IS 'Tracks gym entrance and exit times with locker assignments';
COMMENT ON TABLE public.face_embeddings IS 'Stores face recognition embeddings for client authentication';
COMMENT ON TABLE public.access_logs IS 'Logs all face recognition access attempts';
COMMENT ON FUNCTION notify_client_change() IS 'Sends PostgreSQL notification when client is inserted or updated';

-- ========================================
-- VERIFICATION
-- ========================================

DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'DATABASE SETUP VERIFICATION';
    RAISE NOTICE '========================================';
    
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
    
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'postgres') THEN
        RAISE NOTICE '✓ postgres role exists';
    END IF;
    
    RAISE NOTICE '========================================';
    RAISE NOTICE 'DATABASE SETUP COMPLETE!';
    RAISE NOTICE '========================================';
END $$;

COMMIT;