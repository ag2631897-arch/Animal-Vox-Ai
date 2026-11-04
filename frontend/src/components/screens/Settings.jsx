import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, ImagePlus, Plus, Save, Speaker, Trash2, X } from 'lucide-react'
import { Shimmer } from '../ui/SkeletonCard'
import { useSkeletonLoader } from '../../hooks/useSkeletonLoader'
import { createSpeechUtterance } from '../../hooks/speech'
import { useAppStore, themePresets } from '../../store/useAppStore'
import { PetAvatar } from '../ui/SpeciesMark'

const colorFields = [['Background', 'bg'], ['Surface', 'surface'], ['Accent', 'accent'], ['Accent 2', 'accent2'], ['Text', 'text']]
const voices = ['Nova', 'Ember', 'Echo', 'Aria', 'Default']
const blankPet = { id: '', name: '', species: 'Dog', breed: '', age: '', gender: 'Unknown', emoji: '' }

function readImage(file, onLoaded) {
  if (!file || !file.type.startsWith('image/')) return
  const reader = new FileReader()
  reader.onload = () => onLoaded(String(reader.result))
  reader.readAsDataURL(file)
}

function readSavedAccount() {
  try { return JSON.parse(localStorage.getItem('animalvox-account') || '{}') } catch { return {} }
}

function SettingsSkeleton({ account }) {
  return <motion.div className={`settings-loading ${account ? 'account-loading' : ''}`} key={account ? 'account-skeleton' : 'pets-skeleton'} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
    {account ? <><Shimmer /><Shimmer /><Shimmer /></> : <><Shimmer className="w-40" /><div className="loading-row"><Shimmer /><Shimmer /><Shimmer /></div><div className="loading-row"><Shimmer /><Shimmer /><Shimmer /></div><Shimmer /></>}
  </motion.div>
}

export default function Settings() {
  const [tab, setTab] = useState('My Pets')
  const [form, setForm] = useState(null)
  const [photo, setPhoto] = useState('')
  const [savedAccount] = useState(readSavedAccount)
  const [accountAvatar, setAccountAvatar] = useState(savedAccount.accountAvatar || '')
  const [accountName, setAccountName] = useState(savedAccount.accountName || '')
  const [accountEmail, setAccountEmail] = useState(savedAccount.accountEmail || '')
  const [notice, setNotice] = useState('')
  const tabLoading = useSkeletonLoader(550, true, tab)
  const fileRef = useRef(null)
  const accountFileRef = useRef(null)
  const pets = useAppStore((state) => state.pets)
  const savePet = useAppStore((state) => state.savePet)
  const deletePet = useAppStore((state) => state.deletePet)
  const theme = useAppStore((state) => state.theme)
  const setThemeColor = useAppStore((state) => state.setThemeColor)
  const setThemePreset = useAppStore((state) => state.setThemePreset)
  const presetName = useAppStore((state) => state.presetName)
  const voice = useAppStore((state) => state.voice)
  const setVoice = useAppStore((state) => state.setVoice)
  const speed = useAppStore((state) => state.speed)
  const setSpeed = useAppStore((state) => state.setSpeed)
  const authUser = useAppStore((state) => state.authUser)
  const activePet = useAppStore((state) => state.pets.find((pet) => pet.id === state.activePetId))

  const [hexDrafts, setHexDrafts] = useState(theme)
  useEffect(() => setHexDrafts(theme), [theme])
  useEffect(() => {
    setAccountName((current) => current || authUser?.name || '')
    setAccountEmail((current) => current || authUser?.email || '')
  }, [authUser])

  const notify = (message) => {
    setNotice(message)
    window.setTimeout(() => setNotice(''), 2200)
  }
  const startForm = (pet = blankPet) => {
    setForm({ ...pet })
    setPhoto(pet.photo || '')
  }
  const save = (event) => {
    event.preventDefault()
    if (!form?.name.trim() || form.name.trim().length < 2) { notify('Pet name must have at least two characters.'); return }
    savePet({ ...form, photo, emoji: '' })
    setForm(null)
    notify('Pet profile saved on this device.')
  }
  const remove = (pet) => {
    if (window.confirm(`Remove ${pet.name}'s profile from this device?`)) deletePet(pet.id)
  }
  const previewVoice = (voiceName = voice) => {
    if (!('speechSynthesis' in window)) {
      notify('Speech preview is unavailable in this browser.')
      return
    }
    window.speechSynthesis.cancel()
    const utterance = createSpeechUtterance(`Hello. This is your ${voiceName} voice preview.`, voiceName, speed)
    window.speechSynthesis.speak(utterance)
  }
  const saveTheme = () => notify(`${presetName} saved on this device.`)
  const saveAccount = () => {
    try { localStorage.setItem('animalvox-account', JSON.stringify({ accountName, accountEmail, accountAvatar })) } catch { /* Private browsing may block local storage. */ }
    notify('Account details saved on this device.')
  }

  return <div className="settings-screen">
    <div className="settings-tabs" role="tablist" aria-label="Settings sections">
      <button className={tab === 'My Pets' ? 'active' : ''} onClick={() => setTab('My Pets')} role="tab" aria-selected={tab === 'My Pets'}>My Pets</button>
      <button className={tab === 'My Account' ? 'active' : ''} onClick={() => setTab('My Account')} role="tab" aria-selected={tab === 'My Account'}>My Account</button>
    </div>
    {notice && <div className="settings-notice" role="status"><Check size={14} />{notice}</div>}

    <AnimatePresence mode="wait">
      {tabLoading ? <SettingsSkeleton account={tab === 'My Account'} /> : tab === 'My Pets' ? <motion.section className="settings-panel" key="pets" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
        <div className="settings-section-head"><div><span className="eyebrow">PROFILE LIBRARY</span><h2>My Pets</h2></div><button className="ghost-action" onClick={() => startForm()}><Plus size={15} />Add Pet</button></div>
        <div className="pet-list">{pets.map((pet) => <div className="pet-profile-row" key={pet.id}>
          <PetAvatar pet={pet} className="pet-profile-avatar" size={18} />
          <div className="pet-profile-info"><strong>{pet.name}</strong><span>{pet.species} · {pet.breed || 'Breed not set'}</span><small>{pet.age || 'Age not set'} · {pet.gender || 'Gender not set'}</small></div>
          <div className="pet-profile-actions"><button onClick={() => startForm(pet)}>Edit</button><button onClick={() => remove(pet)} aria-label={`Delete ${pet.name}`}><Trash2 size={14} />Delete</button></div>
        </div>)}</div>
        <AnimatePresence>{form && <motion.form className="pet-form" onSubmit={save} initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.24 }}>
          <div className="form-title"><h3>{form.id ? `Edit ${form.name}` : 'New pet profile'}</h3><button type="button" className="ghost-icon" onClick={() => setForm(null)} aria-label="Close pet form"><X size={16} /></button></div>
          <div className="form-grid">
            <label>Pet Name<input required minLength={2} value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="e.g. Rio" /></label>
            <label>Species<select value={form.species} onChange={(event) => setForm({ ...form, species: event.target.value })}><option>Dog</option><option>Cat</option><option>Bird</option><option>Dolphin</option><option>Whale</option><option>Frog</option><option>Other</option></select></label>
            <label>Breed<input value={form.breed} onChange={(event) => setForm({ ...form, breed: event.target.value })} placeholder="Breed" /></label>
            <label>Age<input value={form.age} onChange={(event) => setForm({ ...form, age: event.target.value })} placeholder="e.g. 3 years" /></label>
            <label>Gender<select value={form.gender} onChange={(event) => setForm({ ...form, gender: event.target.value })}><option>Unknown</option><option>Male</option><option>Female</option></select></label>
          </div>
          <div className="photo-upload-wrap">
            <button type="button" className="photo-drop" onClick={() => fileRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); readImage(event.dataTransfer.files[0], setPhoto) }}>
              {photo ? <><img src={photo} alt="Pet preview" /><span>Photo ready</span></> : <><ImagePlus size={17} /><span>Drop photo or click to upload</span></>}
            </button>
            {photo && <button type="button" className="photo-remove" onClick={() => setPhoto('')} aria-label="Remove pet photo"><X size={14} /></button>}
          </div>
          <input type="file" accept="image/*" hidden ref={fileRef} onChange={(event) => readImage(event.target.files?.[0], setPhoto)} />
          <button className="primary-button form-save" type="submit">Save Pet</button>
        </motion.form>}</AnimatePresence>
        <p className="local-storage-note">Pet profiles stay in this browser for this demo.</p>
      </motion.section> : <motion.section className="settings-panel account-panel" key="account" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
        <section className="account-profile">
          <div className="settings-section-head">
            <div>
              <span className="eyebrow">AUTHENTICATION & SECURITY</span>
              <h2>My Account</h2>
            </div>
            {authUser?.provider && (
              <span style={{
                fontSize: '0.74rem',
                padding: '4px 10px',
                borderRadius: 20,
                background: authUser.provider === 'google' ? 'rgba(66, 133, 244, 0.15)' : 'rgba(0, 201, 167, 0.15)',
                color: authUser.provider === 'google' ? '#8AB4F8' : '#00C9A7',
                border: `1px solid ${authUser.provider === 'google' ? 'rgba(66, 133, 244, 0.3)' : 'rgba(0, 201, 167, 0.3)'}`,
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
              }}>
                {authUser.provider === 'google' ? 'Google Account' : 'NeonDB Account'}
              </span>
            )}
          </div>
          <div className="account-form">
            <button type="button" className="account-avatar" onClick={() => accountFileRef.current?.click()} aria-label="Upload account avatar">{accountAvatar ? <img src={accountAvatar} alt="Account avatar" /> : (accountName ? accountName.charAt(0).toUpperCase() : 'A')}<span><ImagePlus size={13} /></span></button>
            <input ref={accountFileRef} type="file" accept="image/*" hidden onChange={(event) => readImage(event.target.files?.[0], setAccountAvatar)} />
            <label>Your name<input value={accountName} onChange={(event) => setAccountName(event.target.value)} /></label>
            <label>Email<input type="email" value={accountEmail} onChange={(event) => setAccountEmail(event.target.value)} /></label>
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 18, alignItems: 'center' }}>
            <button className="ghost-action account-save" onClick={saveAccount}><Save size={14} />Save profile</button>
          </div>
        </section>

        <section className="theme-section">
          <div className="settings-section-head"><div><span className="eyebrow">COLOR SYSTEM</span><h2>Custom Theme</h2></div><span className="theme-current">{presetName}</span></div>
          <div className="theme-presets">{themePresets.map((preset) => <button key={preset.name} className={`theme-preset ${presetName === preset.name ? 'selected' : ''}`} onClick={() => setThemePreset(preset)}><span style={{ '--swatch-bg': preset.colors.bg, '--swatch-surface': preset.colors.surface, '--swatch-accent': preset.colors.accent }}><i /><i /><i /></span><small>{preset.name}</small></button>)}</div>
          <div className="color-controls">{colorFields.map(([label, key]) => <label className="color-row" key={key}><span>{label}</span><input type="color" value={theme[key]} onChange={(event) => setThemeColor(key, event.target.value)} aria-label={`${label} color`} /><input className="color-hex" value={hexDrafts[key] || theme[key].toUpperCase()} onChange={(event) => { const value = event.target.value; setHexDrafts((current) => ({ ...current, [key]: value })); if (/^#[0-9a-f]{6}$/i.test(value)) setThemeColor(key, value) }} onBlur={() => setHexDrafts((current) => ({ ...current, [key]: theme[key].toUpperCase() }))} aria-label={`${label} hex value`} /></label>)}</div>
          <div className="theme-preview" style={{ '--preview-bg': theme.bg, '--preview-surface': theme.surface, '--preview-accent': theme.accent, '--preview-text': theme.text }}><div className="preview-heading"><span className="preview-brand">AnimalVox</span><span>Field preview</span></div><div className="preview-signal"><span /><i /><b /></div><div className="preview-result"><strong>{activePet?.name || 'Your pet'}:</strong><p>Interpretation preview</p><small>Behavioral context</small></div></div>
          <button className="primary-button save-theme" onClick={saveTheme}><Save size={15} />Save as My Theme</button>
        </section>

        <section className="voice-section">
          <div className="settings-section-head"><div><span className="eyebrow">SPEECH OUTPUT</span><h2>Voice Settings</h2></div></div>
          <div className="voice-options">{voices.map((option) => <div className={`voice-option ${voice === option ? 'selected' : ''}`} key={option}><button className="voice-name" onClick={() => setVoice(option)} aria-pressed={voice === option}>{option}</button><button className="voice-preview" onClick={() => { setVoice(option); previewVoice(option) }} aria-label={`Preview ${option} voice`}><Speaker size={13} /></button></div>)}</div>
          <label className="speed-control">Speech speed <span>{speed.toFixed(2)}×</span><input type="range" min="0.75" max="1.5" step="0.25" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} /><div className="speed-ticks" aria-hidden="true"><span>0.75</span><span>1.0</span><span>1.25</span><span>1.5</span></div></label>
        </section>
      </motion.section>}
    </AnimatePresence>
  </div>
}
