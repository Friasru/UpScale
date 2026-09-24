import type { ChatMessage, ChatResponse } from '../types/chat'

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export async function sendChat(messages: ChatMessage[], signal?: AbortSignal): Promise<ChatMessage> {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal,
  })
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`)
  }
  const body = (await response.json()) as ChatResponse
  return body.message
}
