import { useEffect } from 'react'
import { ArrowUpRight, LogOut } from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import Sidebar from './Sidebar'
import PetSelector from './PetSelector'
import FirstTimePetPopup from '../ui/FirstTimePetPopup'
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

export default function AppShell({ children }) {
  const location = useLocation()
  const navigate = useNavigate()
  const theme = useAppStore((s) => s.theme)
  const setActiveScreen = useAppStore((s) => s.setActiveScreen)
  const logout = useAppStore((s) => s.logout)
  const authUser = useAppStore((s) => s.authUser)
  const screen = pathToScreen[location.pathname] || 'record'

  useEffect(() => {
    setActiveScreen(screen)
    document.title = `${titles[screen]} · AnimalVox`
  }, [screen, setActiveScreen])

  const handleLogout = () => {
    logout()
    navigate('/landing')
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

  return (
    <div className="app-shell" style={style}>
      <Sidebar />
      <div className="app-main">
        <header className="app-topbar">
          <div>
            <span className="eyebrow">ANIMALVOX / FIELD NOTES</span>
            <h1>{titles[screen]}</h1>
          </div>
          <div className="topbar-actions">
            {!hidePetSelector.has(screen) && <PetSelector />}
            {authUser && (
              <button className="topbar-user" onClick={handleLogout} title="Sign out">
                <span className="topbar-user-name">{authUser.name?.charAt(0) || 'U'}</span>
                <LogOut size={14} />
              </button>
            )}
            <Link className="back-to-site" to="/landing">Site <ArrowUpRight size={14} /></Link>
          </div>
        </header>
        <main className="screen-content" key={location.pathname}>{children}</main>
      </div>

      {/* First-time pet profile onboarding popup */}
      <FirstTimePetPopup />
    </div>
  )
}
