import { useEffect, useState } from 'react'

export function useSkeletonLoader(duration = 1100, enabled = true, resetKey = '') {
  const [loading, setLoading] = useState(enabled)
  useEffect(() => {
    if (!enabled) { setLoading(false); return undefined }
    setLoading(true)
    const timer = window.setTimeout(() => setLoading(false), duration)
    return () => window.clearTimeout(timer)
  }, [duration, enabled, resetKey])
  return loading
}
