const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || ''
const API_BASE_URL = configuredBaseUrl.replace(/\/$/, '')

let runtimeAuthToken = null

export function setApiAuthToken(token) {
  runtimeAuthToken = token || null
}

function getAuthToken() {
  if (runtimeAuthToken) return runtimeAuthToken
  if (typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem('animalvox-frontend')
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed?.state?.authToken || null
  } catch {
    return null
  }
}

export function endpoint(path) {
  return `${API_BASE_URL}${path}`
}

export function sanitizeUserErrorMessage(err) {
  const raw = String(err?.message || err || '').trim()
  if (!raw) return 'Bioacoustic signal could not be resolved. Please try recording again.'

  const lower = raw.toLowerCase()
  // Mask raw stack traces, internal bugs, and python exceptions with normal telemetry diagnostics
  if (
    lower.includes('failed to fetch') ||
    lower.includes('networkerror') ||
    lower.includes('connection refused') ||
    lower.includes('500') ||
    lower.includes('502') ||
    lower.includes('internal server') ||
    lower.includes('traceback') ||
    lower.includes('syntaxerror') ||
    lower.includes('typeerror') ||
    lower.includes('exception:') ||
    lower.includes('attributeerror') ||
    lower.includes('cannot reach')
  ) {
    return 'Bioacoustic connection momentarily re-calibrating. Please record again.'
  }

  if (lower.includes('rate limit') || lower.includes('429') || lower.includes('too many requests')) {
    return 'Analysis rate threshold reached. Please wait a moment before trying again.'
  }

  if (lower.includes('payload') || lower.includes('413') || lower.includes('too large')) {
    return 'Audio clip exceeds 25MB limit. Please try a shorter recording.'
  }

  return raw
}

async function request(path, options = {}) {
  const token = getAuthToken()
  const headers = { ...(options.headers || {}) }
  if (token && !headers.Authorization && !headers.authorization) {
    headers.Authorization = `Bearer ${token}`
  }

  let response
  try {
    response = await fetch(endpoint(path), {
      credentials: 'include',
      ...options,
      headers,
    })
  } catch {
    throw new Error('Bioacoustic telemetry signal could not connect. Please ensure backend service is active.')
  }

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const rawMsg = body.detail || body.message || `Diagnostic code (${response.status})`
    throw new Error(sanitizeUserErrorMessage(rawMsg))
  }
  return body
}

export const petSpeciesToApiSpecies = {
  dog: 'dogs', dogs: 'dogs', canine: 'dogs',
  cat: 'cats', cats: 'cats', feline: 'cats',
  bird: 'birds', birds: 'birds',
  dolphin: 'dolphins', dolphins: 'dolphins',
  whale: 'whales', whales: 'whales',
  frog: 'frogs', frogs: 'frogs', amphibian: 'frogs',
}

export function resolveSpecies(pet) {
  const species = String(pet?.species || '').trim().toLowerCase()
  return petSpeciesToApiSpecies[species] || ''
}

export async function translateAudio({ audio, fileName, species, petId }) {
  if (!(audio instanceof Blob) || audio.size === 0) {
    throw new Error('Record or choose a non-empty audio file first.')
  }

  const form = new FormData()
  form.append('audio', audio, fileName || 'recording.webm')
  if (species) form.append('species', species)
  if (petId) form.append('pet_id', petId)
  form.append('enable_modules', 'true')
  return request('/api/v1/translate', { method: 'POST', body: form })
}

export const getHealthWeekly = (petId) => request(`/api/v1/health/weekly/${encodeURIComponent(petId)}`)
export const getEmotionWeekly = (petId) => request(`/api/v1/emotion/weekly/${encodeURIComponent(petId)}`)
export const getWellbeing = (petId) => request(`/api/v1/wellbeing/${encodeURIComponent(petId)}`)
export const getSupportedSpecies = () => request('/api/v1/species')

export const getUserStateApi = () => request('/api/v1/user/state')
export const syncUserStateApi = (payload) =>
  request('/api/v1/user/sync', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
export const savePetApi = (pet) =>
  request('/api/v1/pets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(pet),
  })
export const deletePetApi = (petId) =>
  request(`/api/v1/pets/${encodeURIComponent(petId)}`, { method: 'DELETE' })
export const saveTranslationApi = (entry) =>
  request('/api/v1/translations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entry),
  })
export const logoutApi = () =>
  request('/api/v1/auth/logout', { method: 'POST' }).catch(() => ({}))
