import { useCallback, useRef, useState } from 'react'
import { sendChat } from '../lib/api'
import type { ChatMessage, ImageAttachment, UiMessage } from '../types/chat'

const newId = () => crypto.randomUUID()

export function useChat() {
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(
    async (content: string, attachments: ImageAttachment[]) => {
      const userMessage: UiMessage = { id: newId(), role: 'user', content, attachments }
      // Error bubbles are UI-only and never sent back to the backend.
      const history: ChatMessage[] = [...messages, userMessage]
        .filter((m) => !m.error)
        .map(({ role, content, attachments }) => ({ role, content, attachments }))

      setMessages((prev) => [...prev, userMessage])
      setIsSending(true)
      const controller = new AbortController()
      abortRef.current = controller

      try {
        const reply = await sendChat(history, controller.signal)
        setMessages((prev) => [...prev, { ...reply, id: newId() }])
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
