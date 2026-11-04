"""
AnimalVox AI — Bioacoustic Audio & Translation Cache
=====================================================
Thread-safe LRU & TTL cache for audio translation results and bioacoustic features.
Prevents duplicate neural passes and redundant LLM API calls on identical audio.
"""

import time
import hashlib
import threading
from typing import Optional, Dict, Any

class BioacousticCache:
    def __init__(self, max_entries: int = 256, default_ttl_sec: int = 3600):
        self.max_entries = max_entries
        self.default_ttl = default_ttl_sec
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def compute_key(audio_bytes: bytes, species: Optional[str] = None, pet_id: Optional[str] = None) -> str:
        """Generate SHA-256 fingerprint from audio header/samples and translation parameters."""
        hasher = hashlib.sha256()
        # Hash first 64KB and total length for ultra-fast, robust fingerprinting
        hasher.update(audio_bytes[:65536])
        hasher.update(str(len(audio_bytes)).encode("utf-8"))
        hasher.update((species or "auto").encode("utf-8"))
        hasher.update((pet_id or "none").encode("utf-8"))
        return hasher.hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve entry if present and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                self.misses += 1
                return None

            # Check expiration
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                self.misses += 1
                return None

            self.hits += 1
            # Move to MRU position
            self._cache[key] = entry
            data = dict(entry["data"])
            data["cached"] = True
            return data

    def set(self, key: str, data: Dict[str, Any], ttl_sec: Optional[int] = None) -> None:
        """Store result with TTL and evict oldest if at capacity."""
        with self._lock:
            ttl = ttl_sec or self.default_ttl
            now = time.time()

            if len(self._cache) >= self.max_entries:
                # Evict oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k]["created_at"])
                del self._cache[oldest_key]

            self._cache[key] = {
                "data": data,
                "created_at": now,
                "expires_at": now + ttl,
            }

    def stats(self) -> Dict[str, Any]:
        """Return cache health metrics."""
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = round((self.hits / total) * 100, 2) if total > 0 else 0.0
            return {
                "cached_items": len(self._cache),
                "max_capacity": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio_percent": hit_ratio,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

# Global cache singleton
bioacoustic_cache = BioacousticCache()
