import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileAudio, Pause, Play, Trash2, Upload } from 'lucide-react'

function fmtSize(bytes) {
  if (!bytes || bytes <= 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

function fmtDuration(s) {
  if (!s || !Number.isFinite(s)) return '—'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}

export default function AudioDropZone({
  onFile,
  disabled,
  audioUrl,
  fileName,
  fileSize: externalFileSize = 0,
  onClear,
}) {
  const [dragging, setDragging] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const [localFileSize, setLocalFileSize] = useState(0)
  const audioElRef = useRef(null)
  const inputRef = useRef(null)
  const rafRef = useRef(null)

  const displaySize = externalFileSize || localFileSize

  // Create / destroy the playback audio element when URL changes
  useEffect(() => {
    if (!audioUrl) {
      setPlaying(false)
      setProgress(0)
      setDuration(0)
      return
    }
    const el = new Audio(audioUrl)
    audioElRef.current = el
    el.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(el.duration)) setDuration(el.duration)
    })
    el.addEventListener('ended', () => { setPlaying(false); setProgress(0) })
    return () => {
      cancelAnimationFrame(rafRef.current)
      el.pause()
      el.src = ''
      audioElRef.current = null
    }
  }, [audioUrl])

  // Animation frame ticker for progress bar
  useEffect(() => {
    if (!playing) { cancelAnimationFrame(rafRef.current); return }
    const tick = () => {
      const el = audioElRef.current
      if (el && el.duration) setProgress(el.currentTime / el.duration)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing])

  const togglePlay = () => {
    const el = audioElRef.current
    if (!el) return
    if (playing) { el.pause(); setPlaying(false) }
    else { el.play().catch(() => {}); setPlaying(true) }
  }

  const handleClear = () => {
    if (audioElRef.current) {
      audioElRef.current.pause()
      audioElRef.current.currentTime = 0
    }
    setPlaying(false)
    setProgress(0)
    setDuration(0)
    setLocalFileSize(0)
    onClear?.()
  }

  // Drag handlers
  const onDragOver = useCallback((e) => { e.preventDefault(); e.stopPropagation() }, [])
  const onDragEnter = useCallback((e) => { e.preventDefault(); setDragging(true) }, [])
  const onDragLeave = useCallback((e) => { e.preventDefault(); setDragging(false) }, [])
  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer?.files?.[0]
    if (file && (file.type.startsWith('audio/') || /\.(wav|mp3|flac|ogg|webm|m4a)$/i.test(file.name))) {
      setLocalFileSize(file.size)
      onFile?.(file)
    }
  }, [onFile])

  const handleInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setLocalFileSize(file.size)
      onFile?.(file)
    }
    e.target.value = ''
  }

  return (
    <div
      className={`audio-dropzone ${dragging ? 'dragging' : ''} ${audioUrl ? 'has-file' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDrop={disabled ? undefined : onDrop}
    >
      <AnimatePresence mode="wait">
        {audioUrl ? (
          <motion.div
            key="preview"
            className="audio-file-preview"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
          >
            <div className="file-info-row">
              <FileAudio size={18} />
              <div className="file-meta">
                <strong>{fileName || 'Recording'}</strong>
                <small>
                  {duration > 0 ? fmtDuration(duration) : '—'}
                  {displaySize > 0 && <> · {fmtSize(displaySize)}</>}
                </small>
              </div>
            </div>

            <div className="file-playback">
              <button type="button" className="playback-toggle" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>
                {playing ? <Pause size={14} /> : <Play size={14} fill="currentColor" />}
              </button>
              <div className="playback-track">
                <div className="playback-filled" style={{ width: `${progress * 100}%` }} />
              </div>
              <span className="playback-stamp">
                {playing ? fmtDuration(progress * duration) : fmtDuration(duration)}
              </span>
            </div>

            <div className="file-actions-bar">
              <button
                type="button"
                className="file-action-btn file-replace"
                onClick={() => inputRef.current?.click()}
                disabled={disabled}
                aria-label="Upload another audio file"
              >
                <Upload size={13} />
                <span>Upload another</span>
              </button>
              {onClear && (
                <button
                  type="button"
                  className="file-action-btn file-remove"
                  onClick={handleClear}
                  disabled={disabled}
                  aria-label="Remove audio clip"
                >
                  <Trash2 size={13} />
                  <span>Remove</span>
                </button>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.button
            key="empty"
            className="dropzone-cta"
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Upload size={18} className="dropzone-icon" />
            <span className="dropzone-main">Drag & drop an audio file here</span>
            <span className="dropzone-sub">or click to browse · WAV, MP3, FLAC, OGG, WebM</span>
          </motion.button>
        )}
      </AnimatePresence>

      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.flac,.ogg,.webm,.m4a"
        hidden
        onChange={handleInputChange}
      />
    </div>
  )
}
