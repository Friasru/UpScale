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

export type AgentName =
  | 'vision'
  | 'technical_analysis'
  | 'market'
  | 'news_sentiment'
  | 'opportunity'
  | 'risk'
  | 'education'

export type Level = 'low' | 'medium' | 'high'

export interface AgentResult {
  agent: AgentName
  status: 'ok' | 'error'
  mock: boolean
  summary: string
  /** Agent-specific structured output; see lib/decision.ts for the opportunity shape. */
  findings: Record<string, unknown>
  evidence: string[]
  error: string | null
}

/** The parts of the backend `Analysis` the UI reads. The backend sends more fields. */
export interface Analysis {
  mock: boolean
  summary: string
  assets: string[]
  agents_used: AgentName[]
  agent_results: AgentResult[]
  uncertainty: { level: Level; notes: string[] }
  disclaimer: string
}

export interface ChatResponse {
  message: ChatMessage
  analysis?: Analysis | null
}

/** A message as shown in the UI. */
export interface UiMessage extends ChatMessage {
  id: string
  error?: boolean
  /** Structured analysis behind an assistant reply. UI-only; never sent back. */
  analysis?: Analysis | null
}
