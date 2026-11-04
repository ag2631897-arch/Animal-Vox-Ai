import { AudioLines, Dna } from 'lucide-react'
import GlassCard from '../ui/GlassCard'
import ConfidenceBar from '../ui/ConfidenceBar'
import { useAppStore } from '../../store/useAppStore'
import { confidenceTier, readableSignal } from '../../lib/labels'

const minimumWinnerConfidence = 0.5

export default function BreedIdentifier() {
  const lastTranslation = useAppStore((state) => state.lastTranslation)
  const breed = lastTranslation?.result?.breed
  const predictions = breed?.top_predictions || []

  if (!lastTranslation) return <div className="breed-screen"><div className="empty-state"><Dna size={24} /><strong>No vocal profile analyzed yet</strong><p>Record or upload a dog vocalization in Translate. A breed estimate appears only when the service returns one.</p></div></div>
  if (!predictions.length) return <div className="breed-screen"><div className="empty-state"><AudioLines size={24} /><strong>No breed estimate for this recording</strong><p>The service did not return an acoustic breed estimate for this vocalization.</p></div></div>

  const [winner, ...alternatives] = predictions
  const confidence = typeof winner.confidence === 'number' ? winner.confidence : null
  const isConfidentEnough = confidence !== null && confidence >= minimumWinnerConfidence
  const group = breed.acoustic_group || winner.group

  return <div className="breed-screen">
    <div className="screen-intro"><div><span className="eyebrow">Acoustic estimate</span><p>Vocal patterns can suggest a group; they cannot confirm breed.</p></div>{breed.fusion_method && <span className="local-badge">{readableSignal(breed.fusion_method, 'Acoustic estimate')}</span>}</div>
    <GlassCard className="breed-result-card">
      {isConfidentEnough ? <>
        <span className="eyebrow">Highest acoustic match</span>
        <div className="breed-winner"><h2>{winner.breed}</h2>{confidence !== null && <span>{Math.round(confidence * 100)}%</span>}</div>
        {confidence !== null && <ConfidenceBar value={confidence} />}
        {group && <p className="breed-clues"><AudioLines size={15} /><span>Acoustic group: {group}.</span></p>}
      </> : <>
        <span className="eyebrow">Acoustic estimate</span>
        <div className="breed-winner"><h2>Not enough signal to name a result</h2></div>
        <p className="breed-clues"><AudioLines size={15} /><span>The strongest returned signal is below the display threshold.</span></p>
      </>}
      {alternatives.length > 0 && <><div className="breed-divider" /><span className="eyebrow">Other returned signals</span><div className="secondary-breeds">{alternatives.map((item) => {
        const score = typeof item.confidence === 'number' ? item.confidence : null
        return <div className="secondary-breed" key={item.breed}><div><strong>{item.breed}</strong>{score !== null && <span>{Math.round(score * 100)}%</span>}</div>{score !== null && <ConfidenceBar value={score} color="var(--text-muted)" />}</div>
      })}</div></>}
      {confidence !== null && <small className="demo-caveat">{confidenceTier(confidence)}. This is an acoustic estimate, not a breed confirmation.</small>}
    </GlassCard>
  </div>
}
