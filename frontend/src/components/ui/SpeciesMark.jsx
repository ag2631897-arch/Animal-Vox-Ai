import { AudioLines, PawPrint } from 'lucide-react'

export function SpeciesMark({ species, className = '', size = 18 }) {
  const label = String(species || '').toLowerCase()
  const Icon = label ? AudioLines : PawPrint
  return <Icon className={`species-mark ${className}`} size={size} strokeWidth={1.6} aria-hidden="true" />
}

export function PetAvatar({ pet, className = '', size = 18 }) {
  const name = pet?.name || ''
  return (
    <span className={`pet-avatar ${className}`} aria-label={name ? `${name} profile` : 'Pet profile'}>
      {pet?.photo ? <img src={pet.photo} alt="" /> : <SpeciesMark species={pet?.species} size={size} />}
    </span>
  )
}
