import { AudioLines, Dna } from 'lucide-react'
import GlassCard from '../ui/GlassCard'
import ConfidenceBar from '../ui/ConfidenceBar'
import { useAppStore } from '../../store/useAppStore'

export default function BreedIdentifier() {
  const lastTranslation = useAppStore((state) => state.lastTranslation)
  const breed = lastTranslation?.result?.breed
  const predictions = breed?.top_predictions || []

  if (!lastTranslation) return <div className="breed-screen"><div className="empty-state"><Dna size={24} /><strong>No vocal profile analyzed yet</strong><p>Record or upload a dog vocalization in Translate. The backend will include acoustic breed signals when available.</p></div></div>
  if (!predictions.length) return <div className="breed-screen"><div className="empty-state"><AudioLines size={24} /><strong>No breed estimate for this recording</strong><p>Breed estimation is available only when the backend can derive an acoustic canine profile. Vocalizations alone cannot confirm breed.</p></div></div>

  const [winner, ...alternatives] = predictions
  return <div className="breed-screen">
    <div className="screen-intro"><div><span className="eyebrow">VOCAL SIGNATURE / BACKEND ANALYSIS</span><p>Acoustic estimates are probabilistic and do not identify breed with certainty.</p></div><span className="local-badge">{breed.fusion_method || 'ACOUSTIC PROFILE'}</span></div>
    <GlassCard className="breed-result-card"><span className="eyebrow">MOST LIKELY ACOUSTIC MATCH</span><div className="breed-winner"><h2>{winner.breed}</h2><span>{Math.round(winner.confidence * 100)}%</span></div><ConfidenceBar value={winner.confidence} /><p className="breed-clues"><AudioLines size={15} /> <span>Acoustic group: {breed.acoustic_group || winner.group || 'Unknown'}.</span></p><div className="breed-divider" /><span className="eyebrow">ALSO POSSIBLE</span><div className="secondary-breeds">{alternatives.map((item) => <div className="secondary-breed" key={item.breed}><div><strong>{item.breed}</strong><span>{Math.round(item.confidence * 100)}%</span></div><ConfidenceBar value={item.confidence} color="var(--text-muted)" /></div>)}</div><small className="demo-caveat">A vocal signal provides an acoustic estimate only; it is not a breed confirmation.</small></GlassCard>
  </div>
}
