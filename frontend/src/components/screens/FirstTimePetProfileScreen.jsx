import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, Check, PawPrint, Sparkles, UserCheck } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'

const speciesCards = [
  { id: 'Dog', label: 'Dog', image: '/images/animals/dog.png', hint: 'Bark, whine, growl' },
  { id: 'Cat', label: 'Cat', image: '/images/animals/cat.png', hint: 'Meow, purr, hiss' },
  { id: 'Bird', label: 'Songbird', image: '/images/animals/bird.png', hint: 'Chirp, trill, alarm call' },
  { id: 'Dolphin', label: 'Dolphin', image: '/images/animals/dolphin.png', hint: 'Whistle, click train' },
  { id: 'Whale', label: 'Whale', image: '/images/animals/whale.png', hint: 'Moan, song phrase' },
  { id: 'Frog', label: 'Frog', image: '/images/animals/frog.png', hint: 'Croak, chorus pulse' },
  { id: 'Other', label: 'Other', hint: 'Wild / exotic animal' },
]

export default function FirstTimePetProfileScreen() {
  const savePet = useAppStore((s) => s.savePet)
  const setActivePet = useAppStore((s) => s.setActivePet)
  const setMustCompletePetProfile = useAppStore((s) => s.setMustCompletePetProfile)
  const setOnboarded = useAppStore((s) => s.setOnboarded)
  const authUser = useAppStore((s) => s.authUser)

  const [name, setName] = useState('')
  const [species, setSpecies] = useState('Dog')
  const [breed, setBreed] = useState('')
  const [age, setAge] = useState('')
  const [gender, setGender] = useState('Unknown')
  const [notes, setNotes] = useState('')
  const [photo, setPhoto] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handlePhotoUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => setPhoto(String(reader.result))
    reader.readAsDataURL(file)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (name.trim().length < 2) return

    setSubmitting(true)
    const petId = `${name.trim().toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`
    const emoji = ''

    const newPet = {
      id: petId,
      name: name.trim(),
      species,
      breed: breed.trim(),
      age: age.trim(),
      gender,
      notes: notes.trim(),
      photo: photo || '',
      emoji,
      createdAt: new Date().toISOString(),
    }

    savePet(newPet)
    setActivePet(petId)
    setOnboarded()
    setMustCompletePetProfile(false)
  }

  return (
    <div className="first-time-onboarding-view">
      <div className="first-time-backdrop-glow" />

      <motion.div
        className="first-time-container"
        initial={{ opacity: 0, y: 24, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.35, ease: 'easeOut' }}
      >
        <header className="first-time-header">
          <div className="first-time-badge">
            <UserCheck size={13} />
            <span>Google Authentication · First-Time Setup</span>
          </div>
          <h1>Welcome to AnimalVox{authUser?.name ? `, ${authUser.name.split(' ')[0]}` : ''}!</h1>
          <p>
            Please set up your primary animal profile to calibrate bioacoustic feature extraction, vocal tract biometrics, and baseline ethology models.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="first-time-form">
          {/* Step 1: Name and Species */}
          <div className="ft-section">
            <label className="ft-label">
              <span>Companion / Animal Name *</span>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Rio, Mochi, Echo, Luna"
                className="ft-input"
                autoFocus
              />
            </label>
          </div>

          <div className="ft-section">
            <span className="ft-label-text">Select Animal Species *</span>
            <div className="ft-species-grid">
              {speciesCards.map((card) => {
                const isSelected = species === card.id
                return (
                  <button
                    key={card.id}
                    type="button"
                    onClick={() => setSpecies(card.id)}
                    className={`ft-species-card ${isSelected ? 'is-selected' : ''}`}
                  >
                    <div className="ft-species-img-box">
                      {card.image ? (
                        <img src={card.image} alt={card.label} className="ft-species-thumb" />
                      ) : (
                        <span className="ft-species-emoji"><PawPrint size={20} /></span>
                      )}
                    </div>
                    <strong className="ft-species-title">{card.label}</strong>
                    <small className="ft-species-hint">{card.hint}</small>
                    {isSelected && (
                      <span className="ft-selected-check">
                        <Check size={12} />
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Step 2: Breed, Age & Gender */}
          <div className="ft-row-3">
            <label className="ft-label">
              <span>Breed / Subspecies</span>
              <input
                type="text"
                value={breed}
                onChange={(e) => setBreed(e.target.value)}
                placeholder={
                  species === 'Dog'
                    ? 'e.g. Golden Retriever, Husky'
                    : species === 'Cat'
                    ? 'e.g. Siamese, Domestic Shorthair'
                    : species === 'Bird'
                    ? 'e.g. Cockatiel, Canary'
                    : species === 'Whale'
                    ? 'e.g. Blue Whale, Humpback'
                    : species === 'Dolphin'
                    ? 'e.g. Bottlenose Dolphin'
                    : species === 'Frog'
                    ? 'e.g. Pacific Treefrog'
                    : 'e.g. Red Fox'
                }
                className="ft-input"
              />
            </label>

            <label className="ft-label">
              <span>Age / Life Stage</span>
              <input
                type="text"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="e.g. 2 years, Juvenile, Adult"
                className="ft-input"
              />
            </label>

            <label className="ft-label">
              <span>Gender</span>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="ft-input ft-select"
              >
                <option value="Unknown">Unknown / Not applicable</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </label>
          </div>

          {/* Step 3: Habits or Vocal Notes */}
          <div className="ft-section">
            <label className="ft-label">
              <span>Acoustic Habits & Behavioral Notes (Optional)</span>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g. Barks at doorbell, chatters at birds by window, frequent low moans"
                className="ft-input"
              />
            </label>
          </div>

          {/* Step 4: Optional Photo Upload */}
          <div className="ft-section ft-photo-row">
            <div className="ft-photo-preview">
              {photo ? (
                <img src={photo} alt="Preview" className="ft-avatar-img" />
              ) : (
                <div className="ft-avatar-placeholder">
                  <PawPrint size={22} />
                </div>
              )}
            </div>
            <div className="ft-photo-upload-meta">
              <span className="ft-photo-title">Companion Avatar (Optional)</span>
              <label className="ft-photo-btn">
                <span>Choose photo</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handlePhotoUpload}
                  style={{ display: 'none' }}
                />
              </label>
            </div>
          </div>

          {/* Submit Action */}
          <div className="ft-actions">
            <button
              type="submit"
              disabled={name.trim().length < 2 || submitting}
              className="ft-submit-btn"
            >
              <span>{submitting ? 'Creating Profile…' : 'Save Profile & Enter AnimalVox'}</span>
              <ArrowRight size={17} />
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  )
}
