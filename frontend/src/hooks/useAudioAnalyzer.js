/**
 * Client-side audio pre-analysis using Web Audio API.
 * Checks for silence, very short audio, and basic spectral properties
 * before sending to the backend for full translation.
 */

export async function preAnalyzeAudio(blob) {
  try {
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

    // Human speech fundamental is typically 85–300 Hz → moderate ZCR
    // A very narrow ZCR band with high energy can hint at speech
    // This is a rough heuristic — the backend does the real detection
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

  // Check explicit fields the backend may return
  if (result.human_detected || result.is_human) return 'human_detected'
  if (result.species_supported === false) return 'unsupported_species'
  if (result.animal_detected === false || result.no_animal) return 'no_animal'

  // Parse rejection_reason or message text
  const reason = (result.rejection_reason || result.message || result.detail || '').toLowerCase()
  if (reason.includes('human') || reason.includes('spoof') || reason.includes('not genuine')) return 'human_detected'
  if (reason.includes('unsupported') || reason.includes('not supported') || reason.includes('species not')) return 'unsupported_species'
  if (reason.includes('no animal') || reason.includes('no vocalization') || reason.includes('cannot detect') || reason.includes('could not be confirmed')) return 'no_animal'

  return null
}
