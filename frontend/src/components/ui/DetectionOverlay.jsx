import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, HelpCircle, Music, RefreshCw, ShieldAlert, UserX, X } from 'lucide-react'

const configs = {
  no_animal: {
    Icon: HelpCircle,
    title: 'No animal detected',
    subtitle: 'We could not find an animal vocalization in this audio.',
    description: 'Use a clear recording of an animal vocalization. Background noise, sound effects, music, and silence cannot be interpreted.',
    tone: 'warning',
    tips: ['Move the microphone closer to the animal', 'Reduce background noise and music', 'Record for at least two seconds'],
  },
  human_detected: {
    Icon: UserX,
    title: 'Human voice detected',
    subtitle: 'This audio contains human speech or an imitation.',
    description: 'The recording contains human vocal patterns rather than a clear animal vocalization.',
    tone: 'danger',
    tips: ['Record your pet directly', 'Avoid speaking or humming during recording', 'Wait for a clear animal sound'],
  },
  unsupported_species: {
    Icon: ShieldAlert,
    title: 'Species not supported',
    subtitle: 'The detected animal is not currently supported.',
    description: 'This recording cannot be interpreted with the species currently supported by AnimalVox.',
    tone: 'neutral',
    tips: ['Try a recording from a supported species', 'Use a clearer, isolated recording', 'Check back as support expands'],
  },
  music_detected: {
    Icon: Music,
    title: 'Music cannot be interpreted',
    subtitle: 'Music, songs, and background tracks cannot be translated.',
    description: 'AnimalVox works with isolated animal vocalizations, not songs, background music, instrumentals, or media tracks.',
    tone: 'warning',
    tips: ['Record your pet in a quiet environment', 'Upload an isolated animal sound', 'Turn off nearby speakers and playback'],
  },
}

export default function DetectionOverlay({ type, onDismiss, animalName }) {
  const dialogRef = useRef(null)
  const closeButtonRef = useRef(null)
  const effectiveType = typeof type === 'object' ? type?.type : type
  const effectiveAnimal = animalName || (typeof type === 'object' ? type?.animalName : '')
  const config = configs[effectiveType]

  useEffect(() => {
    const previousFocus = document.activeElement
    closeButtonRef.current?.focus()
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onDismiss()
      }
      if (event.key !== 'Tab') return
      const focusable = dialogRef.current?.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled])')
      if (!focusable?.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      previousFocus?.focus?.()
    }
  }, [onDismiss])

  if (!config) return null
  const { Icon } = config
  const subtitle = effectiveType === 'unsupported_species' && effectiveAnimal
    ? `We detected a ${effectiveAnimal} vocalization, which is not currently supported.`
    : config.subtitle

  return (
    <motion.div className="signal-dialog-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
      <motion.div
        className="signal-dialog"
        data-tone={config.tone}
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="detection-dialog-title"
        aria-describedby="detection-dialog-description"
        initial={{ scale: 0.98, y: 16, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        exit={{ scale: 0.98, y: 16, opacity: 0 }}
        transition={{ duration: 0.2, ease: [0.2, 0, 0, 1] }}
      >
        <button ref={closeButtonRef} className="signal-dialog-close" type="button" onClick={onDismiss} aria-label="Close dialog"><X size={18} /></button>
        <div className="signal-dialog-icon"><Icon size={24} strokeWidth={1.5} /></div>
        <h2 id="detection-dialog-title">{config.title}</h2>
        <p className="signal-dialog-subtitle">{subtitle}</p>
        <p id="detection-dialog-description" className="signal-dialog-description">{config.description}</p>
        <div className="signal-dialog-tips">
          <h3><AlertTriangle size={16} /> What to try</h3>
          <ul>{config.tips.map((tip) => <li key={tip}>{tip}</li>)}</ul>
        </div>
        <footer className="signal-dialog-actions">
          <button className="primary-button signal-dialog-retry" type="button" onClick={onDismiss}><RefreshCw size={16} /> Try again</button>
          <button className="ghost-action signal-dialog-dismiss" type="button" onClick={onDismiss}>Close</button>
        </footer>
      </motion.div>
    </motion.div>
  )
}
