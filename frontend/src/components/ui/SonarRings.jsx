export default function SonarRings({ active = false }) {
  return <div className={`sonar-rings ${active ? 'sonar-active' : ''}`} aria-hidden="true"><i /><i /><i /></div>
}
