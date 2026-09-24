// Mirrors backend/upscale/schemas.py
export type Role = 'user' | 'assistant'

export type ImageMediaType = 'image/png' | 'image/jpeg' | 'image/webp' | 'image/gif'

export interface ImageAttachment {
  name: string
  media_type: ImageMediaType
  /** Base64-encoded bytes, no `data:` prefix. */
  data: string
}

export interface ChatMessage {
  role: Role
  content: string
  attachments: ImageAttachment[]
}

export interface ChatResponse {
  message: ChatMessage
}

/** A message as shown in the UI. */
export interface UiMessage extends ChatMessage {
  id: string
  error?: boolean
}
