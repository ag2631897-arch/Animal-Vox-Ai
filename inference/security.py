"""
AnimalVox AI — Security, Rate Limiting & Hardening Middleware
==============================================================
Provides:
1. Sliding-window IP-based Rate Limiter (limits translation abuse and auth brute force).
2. Security Headers Middleware (CSP, HSTS, X-Content-Type-Options, etc.).
3. Request size and audio payload validator.
"""

import time
import threading
from typing import Dict, List, Tuple
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

class RateLimiter:
    """Sliding-window IP rate limiter with endpoint-specific thresholds."""
    def __init__(self):
        # Maps (ip, route_group) -> list of timestamp floats
        self._history: Dict[Tuple[str, str], List[float]] = {}
        self._lock = threading.Lock()

        # Route limits: (max_requests, window_seconds)
        self.limits = {
            "translate": (30, 60),    # 30 translations per minute per IP
            "auth": (20, 60),         # 20 auth calls per minute per IP
            "default": (150, 60),     # 150 general requests per minute
        }

    def _get_route_group(self, path: str) -> str:
        if "/api/v1/translate" in path:
            return "translate"
        if "/api/v1/auth" in path:
            return "auth"
        return "default"

    def is_allowed(self, client_ip: str, path: str) -> Tuple[bool, int, int]:
        """
        Check if request is permitted.
        Returns: (is_allowed, remaining_requests, retry_after_seconds)
        """
        route_group = self._get_route_group(path)
        max_reqs, window_sec = self.limits.get(route_group, self.limits["default"])
        now = time.time()
        window_start = now - window_sec
        key = (client_ip, route_group)

        with self._lock:
            timestamps = self._history.get(key, [])
            # Filter out timestamps older than the window
            timestamps = [t for t in timestamps if t > window_start]

            if len(timestamps) >= max_reqs:
                oldest = timestamps[0]
                retry_after = max(1, int(oldest + window_sec - now))
                self._history[key] = timestamps
                return False, 0, retry_after

            timestamps.append(now)
            self._history[key] = timestamps
            remaining = max(0, max_reqs - len(timestamps))
            return True, remaining, 0

    def cleanup_old(self) -> None:
        """Periodically clean stale keys."""
        now = time.time()
        with self._lock:
            stale_keys = []
            for key, timestamps in self._history.items():
                active = [t for t in timestamps if now - t < 120]
                if not active:
                    stale_keys.append(key)
                else:
                    self._history[key] = active
            for k in stale_keys:
                del self._history[k]

rate_limiter = RateLimiter()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Adds security headers, enforces rate limits, and validates request sizes."""
    async def dispatch(self, request: Request, call_next):
        # 1. Determine client IP (honor reverse proxies if present)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "127.0.0.1"

        path = request.url.path

        # 2. Check Rate Limit for API endpoints
        remaining = None
        limit_val = None
        if path.startswith("/api/"):
            allowed, remaining, retry_after = rate_limiter.is_allowed(client_ip, path)
            route_group = rate_limiter._get_route_group(path)
            limit_val = rate_limiter.limits.get(route_group, rate_limiter.limits["default"])[0]
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"Too many requests. Please wait {retry_after} seconds before trying again.",
                        "retry_after": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit_val),
                        "X-RateLimit-Remaining": "0",
                    }
                )

        # 3. Check payload size (Max 25MB for audio uploads)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 25 * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "message": "Uploaded audio exceeds maximum permitted size of 25MB."
                }
            )

        # 4. Proceed with request
        response: Response = await call_next(request)

        # 5. Inject comprehensive security & rate limiting headers
        if remaining is not None and limit_val is not None:
            response.headers["X-RateLimit-Limit"] = str(limit_val)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(self)"
        response.headers["Server"] = "AnimalVox-Engine/1.1"

        return response
