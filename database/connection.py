"""
AnimalVox AI — Database Connection & Schema Management (NeonDB PostgreSQL)
"""

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_iBPpa19reWkq@ep-small-cell-b338w5p1-pooler.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)

def get_connection():
    """Create and return a raw psycopg2 connection to NeonDB PostgreSQL."""
    return psycopg2.connect(DATABASE_URL)

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
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_pets_user ON pets(user_id);
            """)

            # 3. Translations Log Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    pet_id VARCHAR(100) REFERENCES pets(id) ON DELETE SET NULL,
                    species VARCHAR(50) NOT NULL,
                    behavior VARCHAR(100),
                    translation TEXT NOT NULL,
                    confidence REAL,
                    intensity REAL,
                    urgency REAL,
                    acoustic_metrics JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_translations_pet ON translations(pet_id);
            """)

            # 4. Emotion Entries (Syncing SQLite Pet Diary to Postgres)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emotion_entries (
                    id SERIAL PRIMARY KEY,
                    pet_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    primary_emotion VARCHAR(100) NOT NULL,
                    primary_confidence REAL NOT NULL,
                    all_emotions JSONB,
                    stress_level REAL NOT NULL,
                    species VARCHAR(50)
                );
                CREATE INDEX IF NOT EXISTS idx_emotion_pet_time ON emotion_entries(pet_id, timestamp);
            """)

            # 5. Wellbeing Reports Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wellbeing_reports (
                    id SERIAL PRIMARY KEY,
                    pet_id VARCHAR(100) NOT NULL,
                    composite_score REAL NOT NULL,
                    composite_label VARCHAR(50) NOT NULL,
                    dimensions JSONB NOT NULL,
                    recommendations JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_wellbeing_pet_time ON wellbeing_reports(pet_id, created_at);
            """)

            conn.commit()
            logger.info("NeonDB PostgreSQL schema initialized successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Database schema successfully created!")
