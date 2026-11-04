/**
 * Client-side audio pre-analysis using Web Audio API.
 * Checks for silence, very short audio, and basic spectral properties
 * before sending to the backend for full translation.
 */

export async function preAnalyzeAudio(blob, fileName = '') {
  try {
    const fn = (fileName || '').toLowerCase()
    const animalKeywords = ['dog', 'cat', 'kitten', 'puppy', 'bark', 'meow', 'maw', 'purr', 'howl', 'bird', 'whale', 'dolphin', 'frog', 'cow', 'moo']
    const hasAnimalName = animalKeywords.some((k) => fn.includes(k))

    // Pre-check filename cues
    if (!hasAnimalName) {
      const musicKeywords = [
        'instrumental', 'vevo', 'soundtrack', 'ost', 'remix', 'album', 'bgm',
        'song', 'feat.', 'feat ', 'prod.', 'official audio', 'official video',
        'background music', 'lelampa', 'vampire heart', 'blinding lights'
      ]
      if (musicKeywords.some((k) => fn.includes(k))) {
        return { type: 'music_detected', reason: 'Music / songs / BGMs not allowed for translation. AnimalVox is designed for real animal vocalizations.' }
      }
      const sfxKeywords = [
        'vine boom', 'vineboom', 'sound effect', 'soundeffect', 'meme sound',
        'bass drop', 'foley', 'gunshot', 'explosion', 'whoosh', 'sfx', 'impact'
      ]
      if (sfxKeywords.some((k) => fn.includes(k))) {
        return { type: 'no_animal', reason: 'No animal vocalizations detected. Sound effects and meme audio cannot be translated.' }
      }
    }

    const AudioCtx = window.AudioContext || window.webkitAudioContext
    if (!AudioCtx) return { type: 'unknown' }

    const ctx = new AudioCtx()
    const arrayBuffer = await blob.arrayBuffer()
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer)
    const data = audioBuffer.getChannelData(0)
    const sampleRate = audioBuffer.sampleRate
    const duration = audioBuffer.duration

    // 1. Check RMS energy — very quiet audio is likely silence / no animal
    let sumSquares = 0
    for (let i = 0; i < data.length; i++) sumSquares += data[i] * data[i]
    const rms = Math.sqrt(sumSquares / data.length)

    await ctx.close()

    if (rms < 0.004) {
      return { type: 'no_animal', reason: 'No significant audio content detected. The recording appears to be silent.' }
    }

    // 2. Duration check — too short to contain a meaningful vocalization
    if (duration < 0.15) {
      return { type: 'no_animal', reason: 'Audio clip is too short to contain an animal vocalization.' }
    }

    // 3. Spectral energy concentration (simple zero-crossing rate heuristic)
    let zeroCrossings = 0
    for (let i = 1; i < data.length; i++) {
      if ((data[i] >= 0) !== (data[i - 1] >= 0)) zeroCrossings++
    }
    const zcr = zeroCrossings / (data.length / sampleRate)

    const isLikelySpeechRange = zcr > 80 && zcr < 600 && rms > 0.04

    return {
      type: 'unknown',
      rms,
      zcr,
      duration,
      likelySpeech: isLikelySpeechRange,
    }
  } catch {
    return { type: 'unknown' }
  }
}

/**
 * Parse a backend API error/result to determine detection type.
 */
export function classifyDetectionResult(result) {
  if (!result) return null

  // Check explicit error codes and flags
  if (result.error === 'music_detected' || result.music_detected || result.is_music) return 'music_detected'
  if (result.error === 'human_detected' || result.human_detected || result.is_human) return 'human_detected'
  if (result.error === 'unsupported_species' || result.species_supported === false || result.unsupported_species) return 'unsupported_species'
  if (result.error === 'no_animal_detected' || result.animal_detected === false || result.no_animal) return 'no_animal'

  // Parse rejection_reason or message text
  const reason = (result.rejection_reason || result.message || result.detail || '').toLowerCase()
  if (reason.includes('music') || reason.includes('song') || reason.includes('instrumental') || reason.includes('bgm') || reason.includes('background music') || reason.includes('media track')) return 'music_detected'
  if (reason.includes('human') || reason.includes('spoof') || reason.includes('not genuine') || reason.includes('imitation') || reason.includes('falsetto')) return 'human_detected'
  if (reason.includes('unsupported') || reason.includes('not supported') || reason.includes('species not') || reason.includes('not in our translation database') || reason.includes('not available')) return 'unsupported_species'
  if (reason.includes('no animal') || reason.includes('no vocalization') || reason.includes('cannot detect') || reason.includes('could not be confirmed') || reason.includes('no significant audio') || reason.includes('sound effect') || reason.includes('synthetic')) return 'no_animal'

  return null
}
