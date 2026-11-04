import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AudioLines, Check, Clock3, Mic, Pause, Play, RefreshCw, Save, Share2, Square, Volume2 } from 'lucide-react'
import GlassCard from '../ui/GlassCard'
import { Shimmer } from '../ui/SkeletonCard'
import SonarRings from '../ui/SonarRings'
import WaveformBars from '../ui/WaveformBars'
import ConfidenceBar from '../ui/ConfidenceBar'
import DetectionOverlay from '../ui/DetectionOverlay'
import AudioDropZone from '../ui/AudioDropZone'
import AutoSpeciesToggle from '../ui/AutoSpeciesToggle'
import { SpeciesMark } from '../ui/SpeciesMark'
import { useAudioRecorder } from '../../hooks/useAudioRecorder'
import { useAppStore } from '../../store/useAppStore'
import { createSpeechUtterance } from '../../hooks/speech'
import { resolveSpecies, translateAudio } from '../../lib/api'
import { preAnalyzeAudio, classifyDetectionResult } from '../../hooks/useAudioAnalyzer'
import { readableSignal, readableSpecies } from '../../lib/labels'

function formatTime(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

function displayBehavior(result) { return readableSignal(result?.vocalization_type || result?.behaviors?.[0]?.name) }

export default function RecordScreen() {
  const activePet = useAppStore((s) => s.pets.find((p) => p.id === s.activePetId))
  const pets = useAppStore((s) => s.pets)
  const addTranslation = useAppStore((s) => s.addTranslation)
  const voice = useAppStore((s) => s.voice)
  const speed = useAppStore((s) => s.speed)
  const autoDetect = useAppStore((s) => s.autoSpeciesDetection)
  const toggleAutoDetect = useAppStore((s) => s.toggleAutoSpeciesDetection)

  const [analysis, setAnalysis] = useState(null)
  const [saved, setSaved] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [detectionType, setDetectionType] = useState(null)
  const [detectedAnimal, setDetectedAnimal] = useState('')
  const [uploadedFileName, setUploadedFileName] = useState('')
  const [visibleWords, setVisibleWords] = useState(0)
  const [typewriterDone, setTypewriterDone] = useState(false)
  const audioRef = useRef(null)
  const twTimerRef = useRef(null)
  const hasAutoSpokenRef = useRef(false)

  const analyzeRecording = useCallback(async ({ blob, fileName }) => {
    setDetectionType(null)
    setDetectedAnimal('')
    setAnalysis(null)
    setVisibleWords(0)
    setTypewriterDone(false)
    hasAutoSpokenRef.current = false
    setUploadedFileName(fileName || '')

    // Client-side pre-check
    const preCheck = await preAnalyzeAudio(blob, fileName)
    if (preCheck.type === 'no_animal' || preCheck.type === 'music_detected') {
      setDetectionType(preCheck.type)
      throw new Error(preCheck.reason || 'No animal vocalizations detected in this audio.')
    }

    let result
    try {
      result = await translateAudio({
        audio: blob,
        fileName,
        species: autoDetect ? '' : resolveSpecies(activePet),
        petId: activePet?.id,
      })
    } catch (err) {
      const det = classifyDetectionResult({ message: err?.message })
      setDetectionType(det || 'no_animal')
      throw err
    }

    if (result.error) {
      const det = classifyDetectionResult(result) || 'no_animal'
      setDetectionType(det)
      setDetectedAnimal(result.detected_animal || '')
      throw new Error(result.rejection_reason || result.message || 'Animal vocalization could not be confirmed.')
    }

    // Route to matching pet when auto-detect is on
    let matchedPet = activePet
    if (autoDetect && result.species && pets.length > 0) {
      const detected = (result.species || '').toLowerCase()
      const match = pets.find((p) => {
        const s = p.species.toLowerCase()
        return detected.includes(s) || s.includes(detected)
      })
      if (match) matchedPet = match
    }

    setAnalysis(result)
    setSaved(false)
    addTranslation({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      createdAt: new Date().toISOString(),
      petId: matchedPet?.id || '',
      petName: matchedPet?.name || result.species || 'Unknown animal',
      emoji: '',
      translation: result.translation,
      behavior: displayBehavior(result),
      confidence: result.behaviors?.[0]?.score ?? null,
      result,
    })
  }, [activePet, pets, addTranslation, autoDetect])

  const recorder = useAudioRecorder(analyzeRecording)
  const isRecording = recorder.status === 'recording'
  const isProcessing = recorder.status === 'processing'
  const isResult = recorder.status === 'result' && analysis
  const confidence = analysis?.behaviors?.[0]?.score || 0

  const detectedSpeciesNormalized = (analysis?.species || '').toLowerCase()
  const activePetMatchesSpecies = activePet && (
    (activePet.species || '').toLowerCase().includes(detectedSpeciesNormalized) ||
    detectedSpeciesNormalized.includes((activePet.species || '').toLowerCase())
  )

  const getDisplayHeader = () => {
    if (!analysis) return { species: '', name: 'Animal', isDetected: false }
    if (!autoDetect && activePet) {
      return { species: activePet.species, name: activePet.name, isDetected: false }
    }
    if (autoDetect && activePetMatchesSpecies) {
      return { species: activePet.species || analysis.species, name: activePet.name, isDetected: false }
    }
    if (autoDetect && pets?.length > 0) {
      const match = pets.find((p) => {
        const s = (p.species || '').toLowerCase()
        return detectedSpeciesNormalized.includes(s) || s.includes(detectedSpeciesNormalized)
      })
      if (match) return { species: match.species || analysis.species, name: match.name, isDetected: false }
    }
    const formatted = analysis.species
      ? analysis.species.charAt(0).toUpperCase() + analysis.species.slice(1)
      : 'Animal'
    return {
      species: analysis.species,
      name: readableSpecies(formatted, 'Animal'),
      isDetected: true,
    }
  }
  const translationWords = analysis?.translation?.split(/\s+/) || []

  // Typewriter effect + auto-speak when result arrives
  useEffect(() => {
    if (!isResult || !analysis?.translation) return
    if (hasAutoSpokenRef.current) return
    hasAutoSpokenRef.current = true

    const words = analysis.translation.split(/\s+/)
    let idx = 0
    twTimerRef.current = setInterval(() => {
      idx += 1
      setVisibleWords(idx)
      if (idx >= words.length) {
        clearInterval(twTimerRef.current)
        setTypewriterDone(true)
      }
    }, 65)

    // Auto-speak the translation
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      const utterance = createSpeechUtterance(analysis.translation, voice, speed)
      utterance.onend = () => setSpeaking(false)
      setSpeaking(true)
      window.speechSynthesis.speak(utterance)
    }

    return () => clearInterval(twTimerRef.current)
  }, [isResult, analysis?.translation, voice, speed])

  useEffect(() => { if (isRecording) setElapsed(recorder.seconds) }, [isRecording, recorder.seconds])
  useEffect(() => () => { window.speechSynthesis?.cancel(); clearInterval(twTimerRef.current) }, [])

  const replayVoice = () => {
    if (!analysis?.translation || !('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const utterance = createSpeechUtterance(analysis.translation, voice, speed)
    utterance.onend = () => setSpeaking(false)
    setSpeaking(true)
    window.speechSynthesis.speak(utterance)
  }

  const replayOriginal = async () => {
    if (!recorder.audioUrl) return
    if (!audioRef.current) audioRef.current = new Audio(recorder.audioUrl)
    audioRef.current.currentTime = 0
    try { await audioRef.current.play() } catch { /* gesture required */ }
  }

  const share = async () => {
    if (!analysis?.translation) return
    try {
      if (navigator.share) await navigator.share({ title: `${activePet?.name || 'Animal'} vocalization`, text: analysis.translation })
      else await navigator.clipboard?.writeText(analysis.translation)
    } catch { /* dismissed */ }
  }

  const chooseFile = (file) => {
    if (file) {
      setUploadedFileName(file.name)
      recorder.processFile(file)
    }
  }

  return (
    <div className="record-screen">
      <div className="record-layout">
        {/* ─── Left: Instrument Panel ─── */}
        <section className="record-instrument" aria-label="Animal audio recorder">
          <div className="hud-corner tl" /><div className="hud-corner tr" /><div className="hud-corner bl" /><div className="hud-corner br" />
          <span className="record-label">BIOACOUSTIC LISTENING</span>

          <div className={`record-control ${isRecording ? 'is-recording' : ''}`}>
            <SonarRings active={isRecording} />
            <button className="record-button" type="button" aria-label={isRecording ? 'Stop recording' : 'Start recording'} aria-pressed={isRecording} onClick={recorder.toggle} disabled={isProcessing}>
              {isRecording ? <Square size={30} fill="currentColor" /> : <Mic size={34} />}
            </button>
          </div>

          <div className="listen-status" aria-live="polite">
            {isRecording ? <><span className="status-led" /> Listening to {activePet?.name || 'your animal'} <span className="record-time">{formatTime(recorder.seconds)}</span></> : isProcessing ? <>Analyzing the acoustic signal<span className="processing-dots"><i /><i /><i /></span></> : isResult ? <><Check size={15} /> Signal interpreted</> : 'Tap to record a vocalization'}
          </div>

          {isRecording && <WaveformBars active />}

          {/* Drag-and-drop audio zone */}
          <AudioDropZone
            onFile={chooseFile}
            disabled={isProcessing || isRecording}
            audioUrl={recorder.audioUrl}
            fileName={uploadedFileName}
          />

          {/* Auto species detection toggle */}
          <AutoSpeciesToggle enabled={autoDetect} onToggle={toggleAutoDetect} />

          {activePet && !autoDetect && (
            <p className="inline-note">Using {activePet.name}'s {activePet.species || 'animal'} profile.</p>
          )}
          {autoDetect && (
            <p className="inline-note">Auto-detecting species from the audio signal.</p>
          )}
          {!activePet && !autoDetect && pets.length === 0 && (
            <p className="inline-note">No pet profile set. Create one in Settings or enable Auto Detection.</p>
          )}

          {recorder.error && !detectionType && <p className="inline-note error-note" role="alert">{recorder.error}</p>}
        </section>

        {/* ─── Detection Overlays ─── */}
        <AnimatePresence>
          {detectionType && (
            <DetectionOverlay
              type={detectionType}
              animalName={detectedAnimal}
              onDismiss={() => { setDetectionType(null); setDetectedAnimal('') }}
            />
          )}
        </AnimatePresence>

        {/* ─── Right: Result / Skeleton ─── */}
        <AnimatePresence mode="wait">
          {isProcessing && (
            <motion.div key="skeleton" className="result-column" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, scale: 0.98 }} transition={{ duration: 0.25 }}>
              <div className="result-card skeleton-result">
                <div className="skeleton-head"><Shimmer className="skeleton-avatar" /><div className="skeleton-copy"><Shimmer className="skeleton-line w-40" /><Shimmer className="skeleton-line w-70" /></div></div>
                <Shimmer className="skeleton-line w-full" />
                <Shimmer className="skeleton-line w-85" />
                <div className="thinking-indicator"><span /><span /><span /></div>
                <Shimmer className="skeleton-audio" />
                <div className="skeleton-metrics"><Shimmer /><Shimmer /><Shimmer /></div>
              </div>
            </motion.div>
          )}

          {isResult && (
            <motion.div key="result" className="result-column" initial={{ opacity: 0, y: 12, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.28 }}>
              <GlassCard className="result-card">
                <div className="result-owner">
                  {(() => {
                    const header = getDisplayHeader()
                    return (
                      <>
                        <span><SpeciesMark species={header.species} size={16} /> {header.name}</span>
                        <small>
                          {header.isDetected
                            ? `Auto-detected species: ${analysis.species}. Probable meaning; not a literal translation.`
                            : 'Probable meaning based on this recording; not a literal translation.'}
                        </small>
                      </>
                    )
                  })()}
                </div>
                {Boolean(analysis.human_speech_removed) && (
                  <div className="human-removed-pill">
                    <span>Human speech was removed before interpretation.</span>
                  </div>
                )}
                <div className="translation-name">Interpretation<span>:</span></div>

                {/* Typewriter translation */}
                <p className="translation-copy">
                  {translationWords.map((word, i) => (
                    <motion.span
                      key={`${word}-${i}`}
                      className="tw-word"
                      initial={{ opacity: 0, filter: 'blur(4px)', y: 4 }}
                      animate={i < visibleWords ? { opacity: 1, filter: 'blur(0px)', y: 0 } : { opacity: 0, filter: 'blur(4px)', y: 4 }}
                      transition={{ duration: 0.14 }}
                    >
                      {word}{' '}
                    </motion.span>
                  ))}
                  {!typewriterDone && <span className="tw-cursor" />}
                </p>

                {/* Voice replay (shown after initial auto-speak completes) */}
                {typewriterDone && (
                  <motion.div className="replay-bar" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
                    <button className="replay-voice-btn" onClick={replayVoice} disabled={speaking}>
                      <Volume2 size={15} />
                      <span>{speaking ? 'Speaking…' : 'Replay Interpretation'}</span>
                    </button>
                    <button className="ghost-icon" aria-label="Replay original recording" onClick={replayOriginal}>
                      <RefreshCw size={15} />
                    </button>
                  </motion.div>
                )}

                <div className="metric-grid">
                  <div className="metric-item"><span>BEHAVIOR</span><b className="behavior-pill">{displayBehavior(analysis)}</b></div>
                  <div className="metric-item"><span>CONFIDENCE</span><div className="metric-value">{Math.round(confidence * 100)}%</div><ConfidenceBar value={confidence} /></div>
                  <div className="metric-item"><span>INTENSITY</span><div className="metric-value">{Math.round((analysis.intensity || 0) * 100)}%</div><ConfidenceBar value={analysis.intensity || 0} color="var(--accent-2)" /></div>
                  <div className="metric-item"><span>URGENCY</span><div className="metric-value amber-text">{Math.round((analysis.urgency || 0) * 100)}%</div><ConfidenceBar value={analysis.urgency || 0} color="var(--accent-3)" /></div>
                </div>

                <div className="result-actions">
                  <button className={`ghost-action ${saved ? 'saved' : ''}`} onClick={() => setSaved((v) => !v)}>{saved ? <Check size={15} /> : <Save size={15} />}{saved ? 'Saved' : 'Save'}</button>
                  <button className="ghost-action" onClick={share}><Share2 size={15} />Share</button>
                </div>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="screen-footnote">
        <AudioLines size={14} /><span>Live interpretation from the AnimalVox API</span><span className="footnote-divider" /><span>BEHAVIORAL CONTEXT MATTERS</span><Clock3 size={13} />
      </div>
    </div>
  )
}
