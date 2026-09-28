import { useCallback, useRef, useState } from 'react'
import { sendChat } from '../lib/api'
import type { ChatMessage, ImageAttachment, ScoutContext, UiMessage } from '../types/chat'
import type { AssetRef } from '../types/scout'

export interface SendOptions {
  /** The exact token this turn is about (Scout's Analyze): sent as is, never re-searched. */
  asset?: AssetRef
  /** UI-only: where the request came from, shown under the user's message. */
  scout?: ScoutContext
}

const newId = () => crypto.randomUUID()

export function useChat() {
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(
    async (content: string, attachments: ImageAttachment[], options: SendOptions = {}) => {
      const userMessage: UiMessage = { id: newId(), role: 'user', content, attachments, scout: options.scout }
      // Error bubbles are UI-only and never sent back to the backend.
      const history: ChatMessage[] = [...messages, userMessage]
        .filter((m) => !m.error)
        .map(({ role, content, attachments }) => ({ role, content, attachments }))

      setMessages((prev) => [...prev, userMessage])
      setIsSending(true)
      const controller = new AbortController()
      abortRef.current = controller

      try {
        const { message: reply, analysis } = await sendChat(history, controller.signal, options.asset)
        setMessages((prev) => [...prev, { ...reply, analysis: analysis ?? null, id: newId() }])
      } catch (err) {
        if (controller.signal.aborted) return
        const detail = err instanceof Error ? err.message : String(err)
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: 'assistant',
            content: `Couldn't reach the UpScale backend (${detail}). Is it running?`,
            attachments: [],
            error: true,
          },
        ])
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null
          setIsSending(false)
        }
      }
    },
    [messages],
  )

  const newChat = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setIsSending(false)
    setMessages([])
  }, [])

  return { messages, isSending, send, newChat }
}
