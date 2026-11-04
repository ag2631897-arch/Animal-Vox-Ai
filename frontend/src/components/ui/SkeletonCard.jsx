export function Shimmer({ className = '' }) { return <div className={`shimmer ${className}`} aria-hidden="true" /> }

export default function SkeletonCard({ compact = false }) {
  return <div className={`skeleton-card ${compact ? 'skeleton-compact' : ''}`} aria-label="Loading content">
    <div className="skeleton-head"><Shimmer className="skeleton-avatar" /><div className="skeleton-copy"><Shimmer className="skeleton-line w-40" /><Shimmer className="skeleton-line w-70" /></div></div>
    <Shimmer className="skeleton-line w-full" /><Shimmer className="skeleton-line w-85" /><Shimmer className="skeleton-line w-60" />
  </div>
}
