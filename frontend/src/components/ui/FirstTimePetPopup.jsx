import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, PawPrint, Sparkles } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'

const speciesOptions = [
  { key: 'Dog', emoji: '🐕' },
  { key: 'Cat', emoji: '🐈' },
  { key: 'Bird', emoji: '🐦' },
  { key: 'Other', emoji: '🐾' },
]

export default function FirstTimePetPopup() {
  const hasCompleted = useAppStore((s) => s.hasCompletedOnboarding)
  const setOnboarded = useAppStore((s) => s.setOnboarded)
  const savePet = useAppStore((s) => s.savePet)
  const setActivePet = useAppStore((s) => s.setActivePet)

  const [step, setStep] = useState(0) // 0 = welcome, 1 = form
  const [name, setName] = useState('')
  const [species, setSpecies] = useState('Dog')
  const [breed, setBreed] = useState('')

  if (hasCompleted) return null

  const handleCreate = () => {
    if (!name.trim()) return
    const id = `${name.trim().toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`
    const emoji = speciesOptions.find((o) => o.key === species)?.emoji || '🐾'
    savePet({ id, name: name.trim(), species, breed, age: '', gender: 'Unknown', emoji })
    setActivePet(id)
    setOnboarded()
  }

  const skip = () => setOnboarded()

  return (
    <motion.div
      className="onboarding-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <motion.div
        className="onboarding-card"
        initial={{ scale: 0.88, y: 40, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        transition={{ type: 'spring', damping: 20, stiffness: 200, delay: 0.1 }}
      >
        <AnimatePresence mode="wait">
          {step === 0 ? (
            <motion.div
              key="welcome"
              className="onboarding-step"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, x: -40 }}
              transition={{ duration: 0.2 }}
            >
              <div className="onboarding-icon-group">
                <Sparkles size={26} className="onboarding-sparkle" />
                <div className="onboarding-paw-ring">
                  <PawPrint size={36} />
                </div>
              </div>
              <h2>Welcome to AnimalVox</h2>
              <p>
                Let's set up a profile for your pet. This helps personalize translations
                and track their behavioral patterns over time.
              </p>
              <div className="onboarding-actions">
                <button className="primary-button" onClick={() => setStep(1)}>
                  Add My Pet <ArrowRight size={16} />
                </button>
                <button className="text-button" onClick={skip}>
                  Skip for now
                </button>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="form"
              className="onboarding-step"
              initial={{ opacity: 0, x: 40 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <h2>Your Pet's Profile</h2>
              <div className="onboarding-fields">
                <label>
                  Pet Name
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Rio, Luna, Buddy"
                    autoFocus
                  />
                </label>

                <label>
                  Species
                  <div className="species-chip-row">
                    {speciesOptions.map((opt) => (
                      <button
                        key={opt.key}
                        type="button"
                        className={`species-chip ${species === opt.key ? 'active' : ''}`}
                        onClick={() => setSpecies(opt.key)}
                      >
                        {opt.emoji} {opt.key}
                      </button>
                    ))}
                  </div>
                </label>

                <label>
                  Breed <span className="optional-tag">(optional)</span>
                  <input
                    value={breed}
                    onChange={(e) => setBreed(e.target.value)}
                    placeholder="e.g. German Shepherd, Siamese"
                  />
                </label>
              </div>
              <div className="onboarding-actions">
                <button
                  className="primary-button"
                  onClick={handleCreate}
                  disabled={!name.trim()}
                >
                  Create Profile <ArrowRight size={16} />
                </button>
                <button className="text-button" onClick={() => setStep(0)}>
                  Back
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  )
}
