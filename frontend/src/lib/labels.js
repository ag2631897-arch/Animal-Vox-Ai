const signalLabels = {
  audio_only: 'Audio recording only',
  acoustic_profile: 'Acoustic profile',
  vocalization: 'Vocalization',
  contact_call: 'Contact Call',
  alarm_call: 'Alarm Call',
  alarm_ground: 'Ground Intruder Alert',
  alarm_aerial: 'Aerial Vigilance Alert',
  attention_seeking: 'Attention Seeking',
  territorial_warning: 'Territorial Warning',
  play_invitation: 'Play Invitation',
  play_vocalization: 'Play Vocalization',
  feeding_call: 'Food Solicitation',
  social_bonding: 'Social Bonding',
  distress_call: 'Distress Signal',
  mating_call: 'Courtship / Mating Call',
  aggression_warning: 'Defensive Aggression Warning',
  submissive_display: 'Submissive Deference',
  submission_signal: 'Submissive Signal',
  navigation_echolocation: 'Navigation / Echolocation',
  contentment: 'Contentment',
  greeting: 'Greeting',
  purr: 'Contentment Purr',
  purring: 'Contentment Purr',
  growl: 'Warning Growl',
  growling: 'Warning Growl',
  bark: 'Alert Bark',
  barking: 'Alert Bark',
  whine: 'Solicitation Whine',
  whining: 'Solicitation Whine',
  howl: 'Long-Range Howl',
  howling: 'Long-Range Howl',
  chirp: 'Avian Chirp',
  chirping: 'Avian Chirp',
  song: 'Territorial Song',
  birdsong: 'Territorial Song',
}

export function readableSignal(value, fallback = 'Unlabeled signal') {
  if (!value || typeof value !== 'string') return fallback
  const trimmed = value.trim()
  if (!trimmed) return fallback
  const normalized = trimmed.toLowerCase().replace(/[\s-]+/g, '_')
  if (signalLabels[normalized]) return signalLabels[normalized]

  // Intelligent fallback: convert snake_case or kebab-case to readable Title Case
  const formatted = trimmed
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1).toLowerCase()}`)
    .join(' ')

  return formatted || fallback
}

export function readableSpecies(value, fallback = 'Unlabeled signal') {
  if (!value || typeof value !== 'string') return fallback
  return value
    .trim()
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1).toLowerCase()}`)
    .join(' ')
}

export function confidenceTier(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null
  const percent = value <= 1 ? value * 100 : value
  if (percent > 85) return 'High confidence'
  if (percent >= 65) return 'Moderate confidence'
  if (percent >= 50) return 'Low confidence'
  return 'Not enough signal to name a result'
}
