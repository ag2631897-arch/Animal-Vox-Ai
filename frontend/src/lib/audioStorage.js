/**
 * Per-user audio clip storage (in-memory cache + IndexedDB persistence).
 * Keeps uploaded/recorded audio clips alive across tab switches and page refreshes,
 * strictly isolated per user account.
 */

const DB_NAME = 'animalvox-audio-store'
const STORE_NAME = 'session-audio'
const DB_VERSION = 1

// In-memory cache keyed by `${userKey}:${slot}` -> { audioUrl, blob, fileName, fileSize, duration }
const memoryCache = new Map()

function makeKey(userKey, slot) {
  return `${userKey || 'guest'}::${slot}`
}

function openDb() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      resolve(null)
      return
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export function getCachedSessionAudio(userKey, slot) {
  if (!userKey) return null
  return memoryCache.get(makeKey(userKey, slot)) || null
}

export async function saveSessionAudio(userKey, slot, { blob, fileName = '', fileSize = 0, duration = 0 }) {
  if (!userKey || !(blob instanceof Blob)) return null
  const key = makeKey(userKey, slot)

  const prev = memoryCache.get(key)
  if (prev?.audioUrl && prev.blob !== blob) {
    try { URL.revokeObjectURL(prev.audioUrl) } catch { /* ignore */ }
  }

  const audioUrl = prev?.blob === blob && prev?.audioUrl ? prev.audioUrl : URL.createObjectURL(blob)
  const record = {
    blob,
    audioUrl,
    fileName: fileName || 'Recording',
    fileSize: fileSize || blob.size || 0,
    duration: duration || 0,
    updatedAt: Date.now(),
  }
  memoryCache.set(key, record)

  try {
    const db = await openDb()
    if (db) {
      await new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite')
        tx.objectStore(STORE_NAME).put(
          {
            blob,
            fileName: record.fileName,
            fileSize: record.fileSize,
            duration: record.duration,
            updatedAt: record.updatedAt,
          },
          key
        )
        tx.oncomplete = () => resolve()
        tx.onerror = () => reject(tx.error)
      })
    }
  } catch {
    // IndexedDB unavailable (e.g. strict private mode); memoryCache still works for session
  }

  return record
}

export async function loadSessionAudio(userKey, slot) {
  if (!userKey) return null
  const key = makeKey(userKey, slot)
  const cached = memoryCache.get(key)
  if (cached?.audioUrl) return cached

  try {
    const db = await openDb()
    if (!db) return null
    const stored = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly')
      const req = tx.objectStore(STORE_NAME).get(key)
      req.onsuccess = () => resolve(req.result || null)
      req.onerror = () => reject(req.error)
    })
    if (stored && stored.blob instanceof Blob && stored.blob.size > 0) {
      const audioUrl = URL.createObjectURL(stored.blob)
      const record = {
        blob: stored.blob,
        audioUrl,
        fileName: stored.fileName || 'Recording',
        fileSize: stored.fileSize || stored.blob.size || 0,
        duration: stored.duration || 0,
        updatedAt: stored.updatedAt || Date.now(),
      }
      memoryCache.set(key, record)
      return record
    }
  } catch {
    // Ignore IndexedDB read errors
  }

  return null
}

export async function clearSessionAudio(userKey, slot) {
  if (!userKey) return
  const key = makeKey(userKey, slot)
  const cached = memoryCache.get(key)
  if (cached?.audioUrl) {
    try { URL.revokeObjectURL(cached.audioUrl) } catch { /* ignore */ }
  }
  memoryCache.delete(key)

  try {
    const db = await openDb()
    if (db) {
      await new Promise((resolve) => {
        const tx = db.transaction(STORE_NAME, 'readwrite')
        tx.objectStore(STORE_NAME).delete(key)
        tx.oncomplete = () => resolve()
        tx.onerror = () => resolve()
      })
    }
  } catch {
    // Ignore
  }
}
