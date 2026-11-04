import { useEffect, useState } from 'react'
import { ChevronDown, LogOut, UserRound } from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import Sidebar from './Sidebar'
import PetSelector from './PetSelector'
import AmbientField from '../ui/AmbientField'
import FirstTimePetPopup from '../ui/FirstTimePetPopup'
import FirstTimePetProfileScreen from '../screens/FirstTimePetProfileScreen'
import { useAppStore } from '../../store/useAppStore'

const titles = {
  record: 'Translate',
  quick: 'Quick Translate',
  detect: 'Animal Detection',
  health: 'Pet Health',
  breed: 'Breed Identifier',
  emotions: 'Emotion Timeline',
  history: 'Translation History',
  settings: 'Settings',
}
const pathToScreen = {
  '/app': 'record',
  '/app/': 'record',
  '/app/quick': 'quick',
  '/app/detect': 'detect',
  '/app/health': 'health',
  '/app/breed': 'breed',
  '/app/emotions': 'emotions',
  '/app/history': 'history',
  '/app/settings': 'settings',
}

const hidePetSelector = new Set(['settings', 'quick'])
const ambientByScreen = {
  record: 'full', quick: 'full', detect: 'full', health: 'low', emotions: 'low', history: 'minimal', settings: 'minimal', breed: 'minimal',
}

export default function AppShell({ children }) {
  const location = useLocation()
  const navigate = useNavigate()
  const theme = useAppStore((s) => s.theme)
  const setActiveScreen = useAppStore((s) => s.setActiveScreen)
  const logout = useAppStore((s) => s.logout)
  const authUser = useAppStore((s) => s.authUser)
  const pets = useAppStore((s) => s.pets)
  const mustCompletePetProfile = useAppStore((s) => s.mustCompletePetProfile)
  const [accountOpen, setAccountOpen] = useState(false)
  const screen = pathToScreen[location.pathname] || 'record'

  useEffect(() => {
    setActiveScreen(screen)
    document.title = `${titles[screen]} · AnimalVox`
  }, [screen, setActiveScreen])

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const style = {
    '--bg': theme.bg,
    '--bg-2': `color-mix(in srgb, ${theme.bg} 82%, #162318)`,
    '--surface': theme.surface,
    '--surface-2': `color-mix(in srgb, ${theme.surface} 78%, white 5%)`,
    '--accent': theme.accent,
    '--accent-2': theme.accent2,
    '--text': theme.text,
    '--accent-dim': `${theme.accent}40`,
    '--accent-glow': `0 0 20px ${theme.accent}60`,
  }

  // When someone logs in with Google for the first time (or is a new user with 0 pets):
  // Show ONLY the pet profile making page first, without sidebar or surrounding tabs.
  const isFirstTimeProfileRequired = Boolean(mustCompletePetProfile || (authUser?.provider === 'google' && pets.length === 0))

  if (isFirstTimeProfileRequired) {
    return (
      <div className="app-shell onboarding-only-shell" style={style}>
        <FirstTimePetProfileScreen />
      </div>
    )
  }

  return (
    <div className="app-shell" style={style} data-ambient={ambientByScreen[screen] || 'minimal'}>
      <AmbientField intensity={ambientByScreen[screen] || 'minimal'} />
      <Sidebar />
      <div className="app-main">
        <header className="app-topbar">
          <h1>{titles[screen]}</h1>
          <div className="topbar-actions">
            {!hidePetSelector.has(screen) && <PetSelector />}
            <div className="account-menu-wrap">
              <button type="button" className="account-menu-trigger" aria-expanded={accountOpen} aria-haspopup="menu" onClick={() => setAccountOpen((open) => !open)}>
                <span className="topbar-user-avatar">{authUser?.avatar_url ? <img src={authUser.avatar_url} alt="" /> : <UserRound size={16} />}</span>
                <span className="topbar-user-name">{authUser?.name?.split(' ')[0] || 'Account'}</span>
                <ChevronDown size={16} aria-hidden="true" />
              </button>
              {accountOpen && <>
                <button type="button" className="account-menu-scrim" aria-label="Close account menu" onClick={() => setAccountOpen(false)} />
                <div className="account-menu" role="menu">
                  <Link to="/app/settings" role="menuitem" onClick={() => setAccountOpen(false)}>Account settings</Link>
                  <button type="button" role="menuitem" onClick={handleLogout}><LogOut size={16} /> Log out</button>
                </div>
              </>}
            </div>
            <Link className="back-to-site" to="/landing">View site</Link>
          </div>
        </header>
        <main className="screen-content" key={location.pathname}>{children}</main>
      </div>

      {/* First-time pet profile onboarding popup */}
      <FirstTimePetPopup />
    </div>
  )
}
