import { useCallback, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Radar, Radio, Upload } from 'lucide-react'
import GlassCard from '../ui/GlassCard'
import ConfidenceBar from '../ui/ConfidenceBar'
import SkeletonCard from '../ui/SkeletonCard'
import { SpeciesMark } from '../ui/SpeciesMark'
import { useAudioRecorder } from '../../hooks/useAudioRecorder'
import { translateAudio } from '../../lib/api'
import { readableSpecies } from '../../lib/labels'

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
  const candidates = result?.auto_detection?.species_detections || (result ? [{
    species: result.species,
    display_name: result.species,
    confidence: result.behaviors?.[0]?.score,
  }] : [])
  const selectFile = (event) => {
    const file = event.target.files?.[0]
    if (file) recorder.processFile(file)
    event.target.value = ''
  }

  return (
    <div className="detection-screen">
      <div className="screen-intro"><div><span className="eyebrow">Single recording</span><p>Analyze a recording without first choosing a species.</p></div></div>
      <div className="detection-layout">
        <div className={`radar-panel ${active ? 'radar-active' : ''}`}>
          <div className="radar-grid"><div className="radar-sweep" /><span className="radar-axis horizontal" /><span className="radar-axis vertical" /><span className="radar-center"><Radio size={17} /></span></div>
          <button className={`radar-action ${active ? 'active' : ''}`} type="button" onClick={recorder.toggle} disabled={loading}><Radar size={17} />{active ? 'Stop recording' : loading ? 'Analyzing' : 'Record a clip'}</button>
          <input ref={uploadRef} type="file" accept="audio/*" hidden onChange={selectFile} />
          <button className="text-button demo-button" type="button" disabled={active || loading} onClick={() => uploadRef.current?.click()}><Upload size={14} /> Upload audio file</button>
          {recorder.error && <p className="inline-note error-note" role="alert">{recorder.error}</p>}
        </div>
        <section className="detection-results">
          <div className="section-heading compact-heading"><div><span className="eyebrow">Species candidates</span><h2>{result ? 'Detected signal' : 'Waiting for audio'}</h2></div>{candidates.length > 0 && <span className="source-count">{String(candidates.length).padStart(2, '0')}</span>}</div>
          {loading ? <div className="skeleton-stack">{[1, 2].map((item) => <SkeletonCard key={item} compact />)}</div> : candidates.length ? <div className="detection-list">
            {candidates.map((item, index) => {
              const confidence = typeof item.confidence === 'number' ? item.confidence : null
              const species = readableSpecies(item.display_name || item.species)
              return <motion.div key={`${item.species}-${index}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2, delay: index * 0.06 }}>
                <GlassCard className="detection-card">
                  <div className="detection-avatar"><SpeciesMark species={item.species} size={20} /></div>
                  <div className="detection-card-content"><strong>{species}</strong>{index === 0 && result.translation && <p>{result.translation}</p>}{confidence !== null && <ConfidenceBar value={confidence} />}</div>
                  {confidence !== null && <span className="detection-confidence">{Math.round(confidence * 100)}<small>%</small></span>}
                </GlassCard>
              </motion.div>
            })}
          </div> : <div className="empty-state"><Radar size={24} /><strong>Start a species scan</strong><p>Record or upload an animal vocalization to see supported species candidates.</p></div>}
        </section>
      </div>
    </div>
  )
}
