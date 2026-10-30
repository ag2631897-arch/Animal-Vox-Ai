import { useCallback, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Radar, Radio, Upload } from 'lucide-react'
import GlassCard from '../ui/GlassCard'
import ConfidenceBar from '../ui/ConfidenceBar'
import SkeletonCard from '../ui/SkeletonCard'
import { useAudioRecorder } from '../../hooks/useAudioRecorder'
import { translateAudio } from '../../lib/api'

export default function DetectionScreen() {
  const [result, setResult] = useState(null)
  const uploadRef = useRef(null)
  const analyze = useCallback(async ({ blob, fileName }) => {
    const response = await translateAudio({ audio: blob, fileName })
    if (response.error) throw new Error(response.rejection_reason || response.message || 'No animal vocalization could be confirmed.')
    setResult(response)
  }, [])
  const recorder = useAudioRecorder(analyze)
  const active = recorder.status === 'recording'
  const loading = recorder.status === 'processing'
  const candidates = result?.auto_detection?.species_detections || (result ? [{ species: result.species, display_name: result.species, emoji: result.species_emoji, confidence: result.behaviors?.[0]?.score || 0 }] : [])
  const selectFile = (event) => { const file = event.target.files?.[0]; if (file) recorder.processFile(file); event.target.value = '' }

  return <div className="detection-screen">
    <div className="screen-intro"><div><span className="eyebrow">SPECIES ROUTING / SINGLE CLIP</span><p>Analyze a recording without selecting a species.</p></div><span className="live-chip"><i /> API CONNECTED</span></div>
    <div className="detection-layout"><div className={`radar-panel ${active ? 'radar-active' : ''}`}><div className="radar-grid"><div className="radar-sweep" /><span className="radar-axis horizontal" /><span className="radar-axis vertical" /><span className="radar-center"><Radio size={17} /></span></div><button className={`radar-action ${active ? 'active' : ''}`} onClick={recorder.toggle} disabled={loading}><Radar size={17} />{active ? 'Stop recording' : loading ? 'Analyzing…' : 'Record a clip'}</button><input ref={uploadRef} type="file" accept="audio/*" hidden onChange={selectFile} /><button className="text-button demo-button" type="button" disabled={active || loading} onClick={() => uploadRef.current?.click()}><Upload size={14} /> Upload audio file</button>{recorder.error && <p className="inline-note" role="alert">{recorder.error}</p>}</div>
      <section className="detection-results"><div className="section-heading compact-heading"><div><span className="eyebrow">SPECIES CANDIDATES</span><h2>{result ? 'Detected signal' : 'Waiting for audio'}</h2></div><span className="source-count">{String(candidates.length).padStart(2, '0')}</span></div>{loading ? <div className="skeleton-stack">{[1, 2].map((item) => <SkeletonCard key={item} compact />)}</div> : candidates.length ? <div className="detection-list">{candidates.map((item, index) => <motion.div key={`${item.species}-${index}`} initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.28, delay: index * 0.08 }}><GlassCard className="detection-card" style={{ '--species-color': '#00C9A7' }}><div className="detection-avatar">{item.emoji || '🐾'}</div><div className="detection-card-content"><strong>{item.display_name || item.species}</strong><p>{index === 0 ? result.translation : 'Alternative species candidate from the routing model.'}</p><ConfidenceBar value={item.confidence} color="#00C9A7" /></div><span className="detection-confidence">{Math.round(item.confidence * 100)}<small>%</small></span></GlassCard></motion.div>)}</div> : <div className="empty-state"><Radar size={24} /><strong>Start a species scan</strong><p>Record or upload an animal vocalization to see supported species candidates.</p></div>}</section></div>
    {result && <div className="retro-status"><span>&gt; ROUTING: {result.auto_detection?.routing_confident ? 'confident' : 'tentative'}</span><span>SPECIES: {result.species}</span><span>PROCESSING: {result.processing_time_ms}ms</span><i className="terminal-cursor">▌</i></div>}
  </div>
}
