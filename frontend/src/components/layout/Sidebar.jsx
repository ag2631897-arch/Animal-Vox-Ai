import { AudioLines, Dna, HeartPulse, History, PawPrint, Radar, Settings2, Smile, Zap } from 'lucide-react'
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/app', label: 'Translate', icon: AudioLines, end: true },
  { to: '/app/quick', label: 'Quick Translate', icon: Zap },
  { to: '/app/detect', label: 'Detect', icon: Radar },
  { to: '/app/health', label: 'Health', icon: HeartPulse },
  { to: '/app/breed', label: 'Breed', icon: Dna },
  { to: '/app/emotions', label: 'Emotions', icon: Smile },
  { to: '/app/history', label: 'History', icon: History },
  { to: '/app/settings', label: 'Settings', icon: Settings2 },
]

function Item({ item, mobile = false }) {
  const Icon = item.icon
  return (
    <NavLink
      to={item.to}
      end={item.end}
      className={({ isActive }) => `nav-item ${isActive ? 'active' : ''} ${mobile ? 'mobile-nav-item' : ''}`}
      aria-label={item.label}
    >
      <Icon size={19} strokeWidth={1.8} />
      <span className="nav-label">{item.label}</span>
      <span className="nav-tooltip">{item.label}</span>
    </NavLink>
  )
}

export default function Sidebar() {
  return (
    <>
      <aside className="sidebar" aria-label="Main navigation">
        <NavLink to="/landing" className="sidebar-logo" aria-label="AnimalVox home">
          <PawPrint size={22} />
          <span className="logo-wave"><AudioLines size={15} /></span>
        </NavLink>
        <nav className="sidebar-nav">
          {links.slice(0, 7).map((item) => <Item key={item.to} item={item} />)}
        </nav>
        <div className="sidebar-bottom">
          <Item item={links[7]} />
        </div>
      </aside>
      <nav className="mobile-tabs" aria-label="Mobile navigation">
        {links.map((item) => <Item key={item.to} item={item} mobile />)}
      </nav>
    </>
  )
}
