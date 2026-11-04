import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AudioLines, Check, Clock3, Mic, RefreshCw, Save, Share2, Square, Volume2, Zap } from 'lucide-react'
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
import { translateAudio } from '../../lib/api'
import { preAnalyzeAudio, classifyDetectionResult } from '../../hooks/useAudioAnalyzer'
import { readableSignal, readableSpecies } from '../../lib/labels'

function formatTime(s) {
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

function displayBehavior(result) { return readableSignal(result?.vocalization_type || result?.behaviors?.[0]?.name) }

/**
 * Quick Translate – no pet profile required.
 * Users can record or upload audio clips and get translations directly.
 */
export default function QuickTranslate() {
  const addTranslation = useAppStore((s) => s.addTranslation)
  const voice = useAppStore((s) => s.voice)
  const speed = useAppStore((s) => s.speed)
  const autoDetect = useAppStore((s) => s.autoSpeciesDetection)
  const toggleAutoDetect = useAppStore((s) => s.toggleAutoSpeciesDetection)
  const [selectedSpecies, setSelectedSpecies] = useState('dogs')
  const autoDetectRef = useRef(autoDetect)
  const selectedSpeciesRef = useRef(selectedSpecies)

  useEffect(() => { autoDetectRef.current = autoDetect }, [autoDetect])
  useEffect(() => { selectedSpeciesRef.current = selectedSpecies }, [selectedSpecies])

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

    const preCheck = await preAnalyzeAudio(blob, fileName)
    if (preCheck.type === 'no_animal' || preCheck.type === 'music_detected') {
      setDetectionType(preCheck.type)
      throw new Error(preCheck.reason || 'No animal vocalizations detected.')
    }

    let result
    try {
      result = await translateAudio({
        audio: blob,
        fileName,
        species: autoDetectRef.current ? '' : selectedSpeciesRef.current,
        petId: null,
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
      throw new Error(result.rejection_reason || result.message || 'Could not translate.')
    }

    setAnalysis(result)
    setSaved(false)
    addTranslation({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      createdAt: new Date().toISOString(),
      petId: '',
      petName: result.species ? (result.species.charAt(0).toUpperCase() + result.species.slice(1)) : 'Unknown animal',
      emoji: '',
      translation: result.translation,
      behavior: displayBehavior(result),
      confidence: result.behaviors?.[0]?.score ?? null,
      result,
    })
  }, [addTranslation])

  const recorder = useAudioRecorder(analyzeRecording)
  const isRecording = recorder.status === 'recording'
  const isProcessing = recorder.status === 'processing'
  const isResult = recorder.status === 'result' && analysis
  const confidence = analysis?.behaviors?.[0]?.score || 0
  const translationWords = analysis?.translation?.split(/\s+/) || []

  // Typewriter + auto-speak
  useEffect(() => {
    if (!isResult || !analysis?.translation) return
    if (hasAutoSpokenRef.current) return
    hasAutoSpokenRef.current = true

    const words = analysis.translation.split(/\s+/)
    let idx = 0
    twTimerRef.current = setInterval(() => {
      idx += 1
      setVisibleWords(idx)
      if (idx >= words.length) { clearInterval(twTimerRef.current); setTypewriterDone(true) }
    }, 65)

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
    try { await audioRef.current.play() } catch { /* gesture */ }
  }

  const share = async () => {
    if (!analysis?.translation) return
    try {
      if (navigator.share) await navigator.share({ title: 'Animal vocalization translation', text: analysis.translation })
      else await navigator.clipboard?.writeText(analysis.translation)
    } catch { /* dismissed */ }
  }

  const chooseFile = (file) => {
    if (file) { setUploadedFileName(file.name); recorder.processFile(file) }
  }

  return (
    <div className="record-screen quick-translate-screen">
      <div className="quick-translate-badge"><Zap size={13} /> Quick Translate — no pet profile required</div>

      <div className="record-layout">
        {/* ─── Left: Instrument ─── */}
        <section className="record-instrument" aria-label="Quick translate recorder">
          <div className="hud-corner tl" /><div className="hud-corner tr" /><div className="hud-corner bl" /><div className="hud-corner br" />
          <span className="record-label">BIOACOUSTIC LISTENING</span>

          <div className={`record-control ${isRecording ? 'is-recording' : ''}`}>
            <SonarRings active={isRecording} />
            <button className="record-button" type="button" aria-label={isRecording ? 'Stop' : 'Record'} onClick={recorder.toggle} disabled={isProcessing}>
              {isRecording ? <Square size={30} fill="currentColor" /> : <Mic size={34} />}
            </button>
          </div>

          <div className="listen-status" aria-live="polite">
            {isRecording ? <><span className="status-led" /> Listening… <span className="record-time">{formatTime(recorder.seconds)}</span></> : isProcessing ? <>Analyzing<span className="processing-dots"><i /><i /><i /></span></> : isResult ? <><Check size={15} /> Interpreted</> : 'Record or upload any animal sound'}
          </div>

          {isRecording && <WaveformBars active />}

          <AudioDropZone onFile={chooseFile} disabled={isProcessing || isRecording} audioUrl={recorder.audioUrl} fileName={uploadedFileName} />
          <AutoSpeciesToggle enabled={autoDetect} onToggle={toggleAutoDetect} />
          {autoDetect ? (
            <p className="inline-note">Species will be detected automatically from the audio.</p>
          ) : (
            <div className="manual-species-select" style={{ marginTop: '10px', width: '100%' }}>
              <p className="inline-note" style={{ marginBottom: '8px' }}>Select species mode:</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
                {[
                  { id: 'dogs', label: 'Dog' },
                  { id: 'cats', label: 'Cat' },
                  { id: 'birds', label: 'Bird' },
                  { id: 'whales', label: 'Whale' },
                  { id: 'dolphins', label: 'Dolphin' },
                  { id: 'frogs', label: 'Frog' },
                ].map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setSelectedSpecies(s.id)}
                    style={{
                      padding: '7px 4px',
                      borderRadius: '8px',
                      border: selectedSpecies === s.id ? '1px solid #10B981' : '1px solid rgba(255,255,255,0.08)',
                      background: selectedSpecies === s.id ? 'rgba(16,185,129,0.18)' : 'rgba(255,255,255,0.03)',
                      color: selectedSpecies === s.id ? '#10B981' : '#94A3B8',
                      fontSize: '12px',
                      fontWeight: selectedSpecies === s.id ? '600' : '400',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {recorder.error && !detectionType && <p className="inline-note error-note" role="alert">{recorder.error}</p>}
        </section>

        {/* Detection Overlays */}
        <AnimatePresence>
          {detectionType && (
            <DetectionOverlay
              type={detectionType}
              animalName={detectedAnimal}
              onDismiss={() => { setDetectionType(null); setDetectedAnimal('') }}
            />
          )}
        </AnimatePresence>

        {/* ─── Right: Result ─── */}
        <AnimatePresence mode="wait">
          {isProcessing && (
            <motion.div key="skel" className="result-column" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
              <div className="result-card skeleton-result">
                <div className="skeleton-head"><Shimmer className="skeleton-avatar" /><div className="skeleton-copy"><Shimmer className="skeleton-line w-40" /><Shimmer className="skeleton-line w-70" /></div></div>
                <Shimmer className="skeleton-line w-full" /><Shimmer className="skeleton-line w-85" />
                <div className="thinking-indicator"><span /><span /><span /></div>
                <Shimmer className="skeleton-audio" /><div className="skeleton-metrics"><Shimmer /><Shimmer /><Shimmer /></div>
              </div>
            </motion.div>
          )}

          {isResult && (
            <motion.div key="res" className="result-column" initial={{ opacity: 0, y: 12, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.28 }}>
              <GlassCard className="result-card">
                <div className="result-owner">
                  <span><SpeciesMark species={analysis.species} size={16} /> {readableSpecies(analysis.species, 'Animal')}</span>
                  <small>Probable meaning; not a literal translation.</small>
                </div>
                {Boolean(analysis.human_speech_removed) && (
                  <div className="human-removed-pill">
                    <span>Human speech was removed before interpretation.</span>
                  </div>
                )}
                <div className="translation-name">Interpretation<span>:</span></div>
                <p className="translation-copy">
                  {translationWords.map((word, i) => (
                    <motion.span key={`${word}-${i}`} className="tw-word" initial={{ opacity: 0, filter: 'blur(4px)', y: 4 }} animate={i < visibleWords ? { opacity: 1, filter: 'blur(0px)', y: 0 } : { opacity: 0, filter: 'blur(4px)', y: 4 }} transition={{ duration: 0.14 }}>{word}{' '}</motion.span>
                  ))}
                  {!typewriterDone && <span className="tw-cursor" />}
                </p>

                {typewriterDone && (
                  <motion.div className="replay-bar" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
                    <button className="replay-voice-btn" onClick={replayVoice} disabled={speaking}><Volume2 size={15} /><span>{speaking ? 'Speaking…' : 'Replay'}</span></button>
                    <button className="ghost-icon" onClick={replayOriginal} aria-label="Replay original"><RefreshCw size={15} /></button>
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
        <AudioLines size={14} /><span>Quick translation from the AnimalVox API</span><span className="footnote-divider" /><span>NO PET PROFILE NEEDED</span><Clock3 size={13} />
      </div>
    </div>
  )
}
