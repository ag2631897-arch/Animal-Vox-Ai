const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || ''
const API_BASE_URL = configuredBaseUrl.replace(/\/$/, '')

function endpoint(path) {
  return `${API_BASE_URL}${path}`
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(endpoint(path), options)
  } catch {
    throw new Error('Cannot reach the AnimalVox API. Start the backend and try again.')
  }

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(body.detail || body.message || `Request failed (${response.status})`)
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
