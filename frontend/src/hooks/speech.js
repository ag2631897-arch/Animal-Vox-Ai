const profiles = {
  Nova: { pitch: 1.14, rate: 1.02, index: 0 },
  Ember: { pitch: 0.86, rate: 0.94, index: 1 },
  Echo: { pitch: 0.94, rate: 1.06, index: 2 },
  Aria: { pitch: 1.22, rate: 1.0, index: 3 },
  Default: { pitch: 1, rate: 1, index: -1 },
}

export function createSpeechUtterance(text, voiceName = 'Default', speed = 1) {
  const utterance = new SpeechSynthesisUtterance(text)
  const profile = profiles[voiceName] || profiles.Default
  const available = window.speechSynthesis?.getVoices?.() || []
  const namedMatch = available.find((voice) => voice.name.toLowerCase().includes(voiceName.toLowerCase()))
  const selected = namedMatch || (profile.index >= 0 && available.length ? available[profile.index % available.length] : null)
  if (selected) utterance.voice = selected
  utterance.pitch = profile.pitch
  utterance.rate = Math.max(0.1, Math.min(10, speed * profile.rate))
  return utterance
}
