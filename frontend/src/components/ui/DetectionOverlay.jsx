import { motion } from 'framer-motion'
import { AlertTriangle, HelpCircle, RefreshCw, ShieldAlert, UserX, X } from 'lucide-react'

const configs = {
  no_animal: {
    Icon: HelpCircle,
    title: 'No Animal Detected',
    subtitle: 'We couldn\'t find any animal vocalizations in this audio.',
    description: 'Make sure the recording contains clear animal sounds — barking, meowing, chirping, or other vocalizations. Background noise, music, or silence won\'t produce a translation.',
    color: '#F5A623',
    gradient: 'linear-gradient(135deg, rgba(245,166,35,0.12), rgba(245,166,35,0.03))',
    tips: ['Move the microphone closer to the animal', 'Reduce background noise and music', 'Record for at least 2–3 seconds', 'Ensure your pet is actually vocalizing'],
  },
  human_detected: {
    Icon: UserX,
    title: 'Human Voice Detected',
    subtitle: 'This audio contains human speech mimicking animal sounds.',
    description: 'Our anti-spoofing system identified human vocal patterns instead of genuine animal vocalizations. AnimalVox is designed to interpret real animal sounds, not imitations.',
    color: '#EF4444',
    gradient: 'linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.03))',
    tips: ['Record your actual pet — not an imitation', 'Avoid speaking or humming during recording', 'If your pet is nearby, wait for their sound'],
  },
  unsupported_species: {
    Icon: ShieldAlert,
    title: 'Species Not Supported Yet',
    subtitle: 'The detected animal isn\'t in our translation database.',
    description: 'We\'re continuously expanding our species library. Currently supported: Dogs, Cats, Birds, Dolphins, Whales, and Frogs. More species are on the way.',
    color: '#A78BFA',
    gradient: 'linear-gradient(135deg, rgba(167,139,250,0.12), rgba(167,139,250,0.03))',
    tips: ['Try one of our six supported species', 'Check back — we add new species regularly', 'Contact us to request a specific species'],
  },
}

export default function DetectionOverlay({ type, onDismiss }) {
  const config = configs[type]
  if (!config) return null
  const { Icon } = config

  return (
    <motion.div
      className="detection-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
    >
      <motion.div
        className="detection-card"
        style={{ '--det-color': config.color, '--det-gradient': config.gradient }}
        initial={{ scale: 0.92, y: 24, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        exit={{ scale: 0.92, y: 24, opacity: 0 }}
        transition={{ type: 'spring', damping: 22, stiffness: 260 }}
      >
        <button className="detection-close" onClick={onDismiss} aria-label="Dismiss">
          <X size={18} />
        </button>

        <div className="detection-icon-ring">
          <span className="detection-pulse-ring" />
          <span className="detection-pulse-ring delay" />
          <Icon size={36} strokeWidth={1.6} />
        </div>

        <h2>{config.title}</h2>
        <p className="detection-subtitle">{config.subtitle}</p>
        <p className="detection-description">{config.description}</p>

        <div className="detection-tips">
          <h4><AlertTriangle size={13} /> What to try</h4>
          <ul>
            {config.tips.map((tip) => (
              <li key={tip}>{tip}</li>
            ))}
          </ul>
        </div>

        <button className="detection-retry" onClick={onDismiss}>
          <RefreshCw size={15} /> Try Again
        </button>
      </motion.div>
    </motion.div>
  )
}
