import { useState } from 'react'
import { ChevronDown, PawPrint, Plus } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../../store/useAppStore'

export default function PetSelector() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const pets = useAppStore((state) => state.pets)
  const activePetId = useAppStore((state) => state.activePetId)
  const setActivePet = useAppStore((state) => state.setActivePet)
  const active = pets.find((pet) => pet.id === activePetId) || pets[0]
  return <div className="pet-selector-wrap">
    <button className="pet-selector" type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
      <span className="pet-avatar-small">{active?.emoji || <PawPrint size={15} />}</span>
      <span>{active?.name || 'Choose a pet'}</span><ChevronDown size={14} className={open ? 'chevron-open' : ''} />
    </button>
    {open && <>
      <button className="dropdown-scrim" aria-label="Close pet menu" onClick={() => setOpen(false)} />
      <div className="pet-dropdown" role="menu">
        {pets.map((pet) => <button type="button" role="menuitem" key={pet.id} className={`pet-option ${pet.id === activePetId ? 'selected' : ''}`} onClick={() => { setActivePet(pet.id); setOpen(false) }}><span className="pet-avatar-small">{pet.emoji}</span><span><strong>{pet.name}</strong><small>{pet.species} · {pet.breed}</small></span></button>)}
        <button type="button" className="pet-option add-pet-option" onClick={() => { setOpen(false); navigate('/app/settings') }}><span className="pet-avatar-small"><Plus size={15} /></span><span><strong>Add Pet</strong><small>Manage your profiles</small></span></button>
      </div>
    </>}
  </div>
}
