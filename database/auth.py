"""
AnimalVox AI — Authentication & User Management Service (NeonDB PostgreSQL + Google OAuth)
"""

import os
import time
import logging
import bcrypt
import jwt
import requests
from typing import Optional, Dict, Any
from psycopg2.extras import RealDictCursor
from database.connection import get_connection
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET", "animalvox_jwt_secret_key_prod_2026_secured_neondb")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_SECONDS = 60 * 60 * 24 * 7  # 7 days

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def create_access_token(user_data: Dict[str, Any]) -> str:
    """Generate a JWT session token."""
    payload = {
        "sub": str(user_data["id"]),
        "email": user_data["email"],
        "name": user_data["name"],
        "provider": user_data.get("provider", "local"),
        "exp": int(time.time()) + JWT_EXPIRATION_SECONDS,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT session token."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception as e:
        logger.debug(f"JWT decode error: {e}")
        return None

def register_user(name: str, email: str, password: str) -> Dict[str, Any]:
    """Register a new user with email and password in NeonDB."""
    email_clean = email.strip().lower()
    name_clean = name.strip()
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check existing email
            cur.execute("SELECT id FROM users WHERE email = %s;", (email_clean,))
            if cur.fetchone():
                return {"success": False, "error": "An account with this email already exists."}
            
            pw_hash = hash_password(password)
            cur.execute("""
                INSERT INTO users (name, email, password_hash, provider)
                VALUES (%s, %s, %s, 'local')
                RETURNING id, name, email, provider, avatar_url, created_at;
            """, (name_clean, email_clean, pw_hash))
            user = cur.fetchone()
            conn.commit()

            token = create_access_token(user)
            return {
                "success": True,
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "provider": user["provider"],
                    "avatar_url": user["avatar_url"],
                    "is_new_user": True,
                },
                "token": token,
                "is_new_user": True,
            }
    finally:
        conn.close()

def login_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate an existing user with email and password from NeonDB."""
    email_clean = email.strip().lower()
    
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, name, email, password_hash, provider, avatar_url
                FROM users WHERE email = %s;
            """, (email_clean,))
            user = cur.fetchone()
            if not user or not user["password_hash"]:
                return {"success": False, "error": "Invalid email or password."}
            
            if not verify_password(password, user["password_hash"]):
                return {"success": False, "error": "Invalid email or password."}
            
            token = create_access_token(user)
            return {
                "success": True,
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "provider": user["provider"],
                    "avatar_url": user["avatar_url"],
                    "is_new_user": False,
                },
                "token": token,
                "is_new_user": False,
            }
    finally:
        conn.close()

def authenticate_google_user(
    credential: Optional[str] = None,
    user_info: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
    code: Optional[str] = None,
    redirect_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Authenticate a user via Google OAuth:
    1. If an authorization code is provided, exchanges it for tokens with Google.
    2. If an access_token is provided, queries Google's userinfo endpoint.
    3. If an ID token credential is provided, verifies with Google's tokeninfo API.
    4. Upserts user in NeonDB users table and issues JWT access token.
    """
    from dotenv import dotenv_values
    from pathlib import Path
    root_env = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    
    client_id = root_env.get("GOOGLE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID") or ""
    client_secret = root_env.get("GOOGLE_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET") or ""

    google_data = None

    # Step 1: Exchange authorization code if provided
    if code:
        try:
            token_resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri or "http://localhost:3000",
                    "grant_type": "authorization_code",
                },
                timeout=8.0,
            )
            if token_resp.status_code == 200:
                tokens = token_resp.json()
                if "access_token" in tokens:
                    access_token = tokens["access_token"]
                if "id_token" in tokens and not credential:
                    credential = tokens["id_token"]
        except Exception as e:
            logger.warning(f"Google code exchange error: {e}")

    # Step 2: Query Google userinfo using access_token
    if access_token and not google_data:
        try:
            resp = requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=6.0,
            )
            if resp.status_code == 200:
                google_data = resp.json()
        except Exception as e:
            logger.warning(f"Google userinfo fetch error: {e}")

    # Step 3: Verify Google ID token credential if provided
    if credential and not google_data:
        try:
            resp = requests.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}",
                timeout=6.0,
            )
            if resp.status_code == 200:
                google_data = resp.json()
        except Exception as e:
            logger.warning(f"Google token verification error: {e}")

    # Step 4: Fallback to direct user_info payload
    if not google_data and user_info:
        google_data = user_info

    if not google_data or not google_data.get("email"):
        return {"success": False, "error": "Google authentication failed. Unable to verify Google account."}

    email_clean = google_data["email"].strip().lower()
    name_clean = google_data.get("name") or email_clean.split("@")[0].capitalize()
    google_id = google_data.get("sub") or google_data.get("google_id")
    avatar_url = google_data.get("picture") or google_data.get("avatar_url")

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if user exists by email or google_id
            cur.execute("""
                SELECT id, name, email, provider, avatar_url, google_id
                FROM users WHERE email = %s OR (google_id IS NOT NULL AND google_id = %s);
            """, (email_clean, google_id))
            existing = cur.fetchone()

            if existing:
                # Update user with Google ID and avatar
                cur.execute("""
                    UPDATE users
                    SET google_id = COALESCE(google_id, %s),
                        avatar_url = COALESCE(%s, avatar_url),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, name, email, provider, avatar_url;
                """, (google_id, avatar_url, existing["id"]))
                user = cur.fetchone()
            else:
                # Create new Google user
                cur.execute("""
                    INSERT INTO users (name, email, provider, google_id, avatar_url)
                    VALUES (%s, %s, 'google', %s, %s)
                    RETURNING id, name, email, provider, avatar_url;
                """, (name_clean, email_clean, google_id, avatar_url))
                user = cur.fetchone()

            conn.commit()
            token = create_access_token(user)

            is_new_user = not bool(existing)
            return {
                "success": True,
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "provider": user["provider"],
                    "avatar_url": user["avatar_url"],
                    "is_new_user": is_new_user,
                },
                "token": token,
                "is_new_user": is_new_user,
            }
    finally:
        conn.close()

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve user details by ID."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, name, email, provider, avatar_url, created_at FROM users WHERE id = %s;", (user_id,))
            return cur.fetchone()
    finally:
        conn.close()
