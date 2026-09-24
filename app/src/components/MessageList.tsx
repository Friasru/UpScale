import { useEffect, useRef } from 'react'
import { toDataUrl } from '../lib/images'
import type { UiMessage } from '../types/chat'

interface MessageListProps {
  messages: UiMessage[]
  isSending: boolean
}

export function MessageList({ messages, isSending }: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isSending])

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <h1>What are you looking at?</h1>
        <p>Ask about a market or attach a chart screenshot.</p>
      </div>
    )
  }

  return (
    <div className="messages" role="log" aria-live="polite">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`message message-${message.role}${message.error ? ' message-error' : ''}`}
        >
          {message.attachments.length > 0 && (
            <div className="message-images">
              {message.attachments.map((image, i) => (
                <a key={i} href={toDataUrl(image)} target="_blank" rel="noreferrer">
                  <img src={toDataUrl(image)} alt={image.name} />
                </a>
              ))}
            </div>
          )}
          {message.content && <p className="message-text">{message.content}</p>}
        </div>
      ))}
      {isSending && (
        <div className="message message-assistant typing" aria-label="UpScale is thinking">
          <span />
          <span />
          <span />
        </div>
      )}
      <div ref={endRef} />
    </div>
  )
}
