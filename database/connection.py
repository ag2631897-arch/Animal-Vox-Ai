"""
AnimalVox AI — Database Connection & Schema Management (NeonDB PostgreSQL)
"""

import os
import logging
import threading
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_iBPpa19reWkq@ep-small-cell-b338w5p1-pooler.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)

_pool_lock = threading.Lock()
_connection_pool: pool.ThreadedConnectionPool | None = None


class _PooledConnectionProxy:
    """
    Wraps a psycopg2 connection from ThreadedConnectionPool so that calling
    `.close()` returns the connection to the pool instead of closing the socket.
    """

    def __init__(self, raw_conn, conn_pool: pool.ThreadedConnectionPool):
        self._conn = raw_conn
        self._pool = conn_pool
        self._returned = False

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    @property
    def closed(self):
        return self._conn.closed

    def close(self):
        if self._returned:
            return
        self._returned = True
        try:
            if self._conn is not None and not self._conn.closed:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                self._pool.putconn(self._conn)
            elif self._conn is not None:
                self._pool.putconn(self._conn, close=True)
        except Exception:
            try:
                if self._conn is not None and not self._conn.closed:
                    self._conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        else:
            try:
                self._conn.commit()
            except Exception:
                pass
        self.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _get_pool() -> pool.ThreadedConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=DATABASE_URL,
                    connect_timeout=10,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5,
                )
    return _connection_pool


def get_connection():
    """Return a healthy pooled psycopg2 connection to NeonDB PostgreSQL."""
    try:
        conn_pool = _get_pool()
        raw_conn = conn_pool.getconn()
        # Verify the pooled connection is still alive
        if raw_conn.closed:
            conn_pool.putconn(raw_conn, close=True)
            raw_conn = psycopg2.connect(
                DATABASE_URL,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
            return raw_conn
        try:
            with raw_conn.cursor() as cur:
                cur.execute("SELECT 1;")
        except Exception:
            try:
                conn_pool.putconn(raw_conn, close=True)
            except Exception:
                pass
            raw_conn = conn_pool.getconn()
        return _PooledConnectionProxy(raw_conn, conn_pool)
    except Exception:
        return psycopg2.connect(
            DATABASE_URL,
            connect_timeout=10,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )


def init_db():
    """Initialize database tables, extensions, and indices."""
    logger.info("Initializing NeonDB PostgreSQL schema...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Enable extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 1. Users Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    password_hash VARCHAR(255),
                    provider VARCHAR(50) DEFAULT 'local',
                    google_id VARCHAR(255) UNIQUE,
                    avatar_url TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);
            """)

            # 2. Pets Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pets (
                    id VARCHAR(100) PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    species VARCHAR(50) NOT NULL,
                    breed VARCHAR(100),
                    age_years REAL,
                    age TEXT DEFAULT '',
                    gender VARCHAR(50) DEFAULT 'Unknown',
                    notes TEXT DEFAULT '',
                    photo TEXT DEFAULT '',
                    emoji VARCHAR(32) DEFAULT '',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_pets_user ON pets(user_id);
                ALTER TABLE pets ADD COLUMN IF NOT EXISTS age TEXT DEFAULT '';
                ALTER TABLE pets ADD COLUMN IF NOT EXISTS gender VARCHAR(50) DEFAULT 'Unknown';
                ALTER TABLE pets ADD COLUMN IF NOT EXISTS notes TEXT DEFAULT '';
                ALTER TABLE pets ADD COLUMN IF NOT EXISTS photo TEXT DEFAULT '';
                ALTER TABLE pets ADD COLUMN IF NOT EXISTS emoji VARCHAR(32) DEFAULT '';
                ALTER TABLE pets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
            """)

            # 3. Translations Log Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id SERIAL PRIMARY KEY,
                    entry_id VARCHAR(100),
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    pet_id VARCHAR(100),
                    pet_name VARCHAR(255),
                    species VARCHAR(50) NOT NULL,
                    behavior VARCHAR(100),
                    translation TEXT NOT NULL,
                    confidence REAL,
                    intensity REAL,
                    urgency REAL,
                    acoustic_metrics JSONB,
                    result_json JSONB,
                    created_iso VARCHAR(64),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_translations_pet ON translations(pet_id);
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS entry_id VARCHAR(100);
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS pet_name VARCHAR(255);
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS result_json JSONB;
                ALTER TABLE translations ADD COLUMN IF NOT EXISTS created_iso VARCHAR(64);
                ALTER TABLE translations DROP CONSTRAINT IF EXISTS translations_pet_id_fkey;
                CREATE INDEX IF NOT EXISTS idx_translations_user ON translations(user_id, created_at DESC);
            """)

            # 4. Emotion Entries (Syncing SQLite Pet Diary to Postgres)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emotion_entries (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    pet_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    primary_emotion VARCHAR(100) NOT NULL,
                    primary_confidence REAL NOT NULL,
                    all_emotions JSONB,
                    stress_level REAL NOT NULL,
                    prosodic_snapshot JSONB,
                    species VARCHAR(50)
                );
                CREATE INDEX IF NOT EXISTS idx_emotion_pet_time ON emotion_entries(pet_id, timestamp);
                ALTER TABLE emotion_entries ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
                ALTER TABLE emotion_entries ADD COLUMN IF NOT EXISTS prosodic_snapshot JSONB;
            """)

            # 5. Wellbeing Reports Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wellbeing_reports (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    pet_id VARCHAR(100) NOT NULL,
                    composite_score REAL NOT NULL,
                    composite_label VARCHAR(50) NOT NULL,
                    trend VARCHAR(50) DEFAULT 'stable',
                    dimensions JSONB NOT NULL,
                    recommendations JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_wellbeing_pet_time ON wellbeing_reports(pet_id, created_at);
                ALTER TABLE wellbeing_reports ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
                ALTER TABLE wellbeing_reports ADD COLUMN IF NOT EXISTS trend VARCHAR(50) DEFAULT 'stable';
            """)

            # 6. Pet Baselines Table (Persistent Vocal Baselines per Pet)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pet_baselines (
                    pet_id VARCHAR(100) PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    baseline_json JSONB NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_pet_baselines_user ON pet_baselines(user_id);
            """)

            # 7. User Settings & Active Session State Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    active_pet_id VARCHAR(100) DEFAULT '',
                    theme JSONB,
                    preset_name VARCHAR(100) DEFAULT 'Midnight Forest',
                    voice VARCHAR(50) DEFAULT 'Nova',
                    speed REAL DEFAULT 1.0,
                    auto_species_detection BOOLEAN DEFAULT TRUE,
                    has_completed_onboarding BOOLEAN DEFAULT FALSE,
                    account_profile JSONB,
                    active_sessions JSONB,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            conn.commit()
            logger.info("NeonDB PostgreSQL schema initialized successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database schema successfully created!")
