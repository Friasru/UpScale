import { useEffect, useRef, useState, type ChangeEvent, type FormEvent, type KeyboardEvent } from 'react'
import {
  ACCEPTED_IMAGE_TYPES,
  MAX_ATTACHMENTS,
  imagesFromClipboard,
  readClipboardImages,
  readImageFile,
  toDataUrl,
  toSupportedImage,
} from '../lib/images'
import { describeDataTransfer, inspectClipboard, pasteLog } from '../lib/pasteDebug'
import type { ImageAttachment } from '../types/chat'

interface ComposerProps {
  onSend: (content: string, attachments: ImageAttachment[]) => void
  disabled: boolean
}

export function Composer({ onSend, disabled }: ComposerProps) {
  const [text, setText] = useState('')
  const [attachments, setAttachments] = useState<ImageAttachment[]>([])
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const formRef = useRef<HTMLFormElement>(null)
  const pasteSeen = useRef(false)

  const canSend = !disabled && (text.trim().length > 0 || attachments.length > 0)

  function submit() {
    if (!canSend) return
    onSend(text.trim(), attachments)
    setText('')
    setAttachments([])
    setError(null)
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    submit()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  async function addFiles(files: File[]) {
    const room = MAX_ATTACHMENTS - attachments.length
    const results = await Promise.allSettled(
      files.slice(0, room).map((f) => toSupportedImage(f).then(readImageFile)),
    )

    const added = results.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : []))
    const errors = results.flatMap((r) => (r.status === 'rejected' ? [(r.reason as Error).message] : []))
    if (files.length > room) errors.push(`You can attach up to ${MAX_ATTACHMENTS} images per message.`)

    // Skip images already attached, so a paste event delivered twice can't duplicate them.
    setAttachments((prev) => {
      const unique = added.filter(
        (a, i) => !prev.some((p) => p.data === a.data) && added.findIndex((b) => b.data === a.data) === i,
      )
      return [...prev, ...unique].slice(0, MAX_ATTACHMENTS)
    })
    setError(errors.length ? errors.join(' ') : null)
    pasteLog('addFiles done', { added: added.length, errors })
  }

  function handleFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? [])
    event.target.value = '' // allow re-selecting the same file
    void addFiles(files)
  }

  async function pasteFromClipboardApi(reason: string) {
    pasteLog(`falling back to navigator.clipboard.read() (${reason})`)
    try {
      const files = await readClipboardImages()
      pasteLog('clipboard.read() images', files.map((f) => ({ name: f.name, type: f.type, size: f.size })))
      if (files.length > 0) await addFiles(files)
    } catch (err) {
      pasteLog('clipboard.read() failed', String(err))
    }
  }

  function handlePaste(event: ClipboardEvent) {
    pasteSeen.current = true
    const target = event.target as HTMLElement | null
    pasteLog('paste event', {
      target: target?.tagName,
      inComposer: !!target && !!formRef.current?.contains(target),
      ...describeDataTransfer(event.clipboardData),
    })
    // Leave pastes into other text fields alone.
    const editable = target?.closest('input, textarea, [contenteditable="true"]')
    if (editable && !formRef.current?.contains(editable)) return
    const data = event.clipboardData
    if (!data) return

    const files = imagesFromClipboard(data)
    pasteLog('images from paste event', files.map((f) => ({ name: f.name, type: f.type, size: f.size })))
    if (files.length > 0) {
      event.preventDefault()
      void addFiles(files)
    } else if (data.types.length === 0) {
      // WebKitGTK can deliver an empty clipboardData for image-only clipboards.
      void pasteFromClipboardApi('paste event had no data')
    }
    // Otherwise it's text: keep the default paste.
  }

  function handleGlobalKeyDown(event: globalThis.KeyboardEvent) {
    if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 'v') return
    if (event.shiftKey) {
      // TEMPORARY: Ctrl+Shift+V dumps the async clipboard contents instead of pasting.
      event.preventDefault()
      pasteLog('Ctrl+Shift+V keydown: running clipboard diagnostic')
      void inspectClipboard()
      return
    }
    pasteSeen.current = false
    pasteLog('Ctrl+V keydown', { target: (event.target as HTMLElement | null)?.tagName })
    setTimeout(() => {
      if (!pasteSeen.current) void pasteFromClipboardApi('no paste event after Ctrl+V')
    }, 100)
  }

  // Latest handlers, so the document listeners below never see stale state.
  const handlers = useRef({ handlePaste, handleGlobalKeyDown })
  useEffect(() => {
    handlers.current = { handlePaste, handleGlobalKeyDown }
  })

  // Listen on the document rather than the form so pasting works wherever focus is
  // (e.g. after returning to the window from the screenshot tool).
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => handlers.current.handlePaste(e)
    const onKeyDown = (e: globalThis.KeyboardEvent) => handlers.current.handleGlobalKeyDown(e)
    document.addEventListener('paste', onPaste)
    document.addEventListener('keydown', onKeyDown)
    pasteLog('listeners attached', { userAgent: navigator.userAgent })
    return () => {
      document.removeEventListener('paste', onPaste)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  return (
    <form ref={formRef} className="composer" onSubmit={handleSubmit}>
      {attachments.length > 0 && (
        <div className="composer-previews">
          {attachments.map((image, i) => (
            <div key={i} className="preview">
              <img src={toDataUrl(image)} alt={image.name} />
              <button
                type="button"
                className="preview-remove"
                aria-label={`Remove ${image.name}`}
                onClick={() => setAttachments((prev) => prev.filter((_, j) => j !== i))}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      {error && <p className="composer-error">{error}</p>}
      <div className="composer-row">
        <button
          type="button"
          className="icon-button"
          aria-label="Attach screenshot"
          title="Attach screenshot"
          onClick={() => fileInputRef.current?.click()}
          disabled={attachments.length >= MAX_ATTACHMENTS}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <path
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M21 12.5l-8.2 8.2a5.3 5.3 0 01-7.5-7.5l8.9-8.9a3.5 3.5 0 015 5l-8.9 8.9a1.8 1.8 0 01-2.5-2.5l8.2-8.2"
            />
          </svg>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_IMAGE_TYPES.join(',')}
          multiple
          hidden
          onChange={handleFiles}
        />
        <textarea
          className="composer-input"
          placeholder="Ask about a market or a chart…"
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button type="submit" className="send-button" disabled={!canSend} aria-label="Send">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 19V5M5 12l7-7 7 7"
            />
          </svg>
        </button>
      </div>
      <p className="composer-hint">UpScale explains evidence and risk. It does not give financial advice.</p>
    </form>
  )
}
