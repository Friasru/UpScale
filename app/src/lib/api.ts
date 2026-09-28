import type { ChatMessage, ChatResponse } from '../types/chat'
import type { AssetRef, ScoutQuery, ScoutView } from '../types/scout'

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

/** `asset`: the exact token the new turn is about (Scout's Analyze); resolved as is. */
export async function sendChat(
  messages: ChatMessage[],
  signal?: AbortSignal,
  asset?: AssetRef,
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(asset ? { messages, asset } : { messages }),
    signal,
  })
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`)
  }
  return (await response.json()) as ChatResponse
}

function scoutParams(query: ScoutQuery): string {
  const params = new URLSearchParams({ limit: String(query.limit) })
  if (query.chain) params.set('chain', query.chain)
  if (query.stage) params.set('stage', query.stage)
  if (query.minLiquidity !== null) params.set('min_liquidity', String(query.minLiquidity))
  return params.toString()
}

async function scoutRequest(path: string, method: 'GET' | 'POST', query: ScoutQuery): Promise<ScoutView> {
  const response = await fetch(`${API_URL}${path}?${scoutParams(query)}`, { method })
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`)
  }
  return (await response.json()) as ScoutView
}

/** The current Growth Scout ranking (the backend scans once if it has none yet). */
export function fetchScout(query: ScoutQuery): Promise<ScoutView> {
  return scoutRequest('/scout', 'GET', query)
}

/** Re-run Scout (the backend shares a running scan and skips one that just finished). */
export function refreshScout(query: ScoutQuery): Promise<ScoutView> {
  return scoutRequest('/scout/refresh', 'POST', query)
}
