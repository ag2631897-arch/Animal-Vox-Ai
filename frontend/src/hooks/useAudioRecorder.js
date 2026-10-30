import { useCallback, useEffect, useRef, useState } from 'react'

export function useAudioRecorder(onComplete) {
  const [status, setStatus] = useState('idle')
  const [seconds, setSeconds] = useState(0)
  const [audioUrl, setAudioUrl] = useState('')
  const [error, setError] = useState('')
  const recorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)

  const finish = useCallback(async (blob, url = '', fileName = 'recording.webm') => {
    if (!(blob instanceof Blob) || blob.size === 0) {
      setError('No audio was captured. Please record again or choose an audio file.')
      setStatus('idle')
      return
    }

    setStatus('processing')
    try {
      await onComplete?.({ blob, url, fileName, durationSeconds: seconds })
      setStatus('result')
    } catch (err) {
      setError(err?.message || 'The recording could not be analyzed. Please try again.')
      setStatus('idle')
    }
  }, [onComplete, seconds])

  const stop = useCallback(() => {
    window.clearInterval(timerRef.current)
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }, [])

  const start = useCallback(async () => {
    setError('')
    setAudioUrl('')
    setSeconds(0)
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError('Microphone access is unavailable in this browser. Choose an audio file instead.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find((type) => MediaRecorder.isTypeSupported?.(type))
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      recorderRef.current = recorder
      chunksRef.current = []
      recorder.ondataavailable = (event) => { if (event.data?.size) chunksRef.current.push(event.data) }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setAudioUrl(url)
        stream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        finish(blob, url, `recording.${blob.type.includes('mp4') ? 'mp4' : 'webm'}`)
      }
      recorder.start()
      setStatus('recording')
      timerRef.current = window.setInterval(() => setSeconds((value) => value + 1), 1000)
    } catch (err) {
      setError(err?.name === 'NotAllowedError'
        ? 'Microphone permission was declined. Choose an audio file instead.'
        : 'Audio input could not start. Choose an audio file instead.')
    }
  }, [finish])

  const processFile = useCallback(async (file) => {
    if (!(file instanceof Blob) || file.size === 0) {
      setError('Choose a non-empty audio file.')
      return
    }
    window.clearInterval(timerRef.current)
    setError('')
    const url = URL.createObjectURL(file)
    setAudioUrl(url)
    setSeconds(0)
    await finish(file, url, file.name || 'upload.wav')
  }, [finish])

  const toggle = useCallback(() => {
    if (status === 'recording') stop()
    else if (status !== 'processing') start()
  }, [start, status, stop])

  useEffect(() => () => {
    window.clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    if (audioUrl) URL.revokeObjectURL(audioUrl)
  }, [audioUrl])

  return { status, seconds, audioUrl, error, toggle, stop, processFile }
}
