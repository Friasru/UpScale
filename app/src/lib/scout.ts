// Presentation helpers for Scout. Formatting only: every value comes from the backend.
import type { ScoutContext } from '../types/chat'
import type { AssetRef, ScoutCard, Stage } from '../types/scout'

/** The order Scout groups stages in (the backend's stage names, unchanged). */
export const STAGE_ORDER: Stage[] = [
  'ACCELERATING',
  'EARLY',
  'NEW',
  'STEADY',
  'CROWDED',
  'FADING',
  'INSUFFICIENT_DATA',
]

/** "INSUFFICIENT_DATA" -> "Insufficient data" (display only; the data keeps the stage). */
export function stageLabel(stage: Stage): string {
  const words = stage.toLowerCase().replace('_', ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

const compact = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumSignificantDigits: 2,
})

/** Two significant digits: $420K, $1.2M, $950. No fake precision. */
export function money(value: number | null): string {
  if (value === null) return 'n/a'
  return `$${compact.format(value)}`
}

/** Pool age: 35m, 5h, 12d. */
export function age(hours: number | null): string {
  if (hours === null) return 'unknown'
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))}m`
  if (hours < 48) return `${Math.round(hours)}h`
  return `${Math.round(hours / 24)}d`
}

/** Market cap when the backend trusts it, otherwise FDV (labeled as such). */
export function capacity(card: ScoutCard): { label: string; value: string } {
  if (card.market.market_cap_usd !== null) return { label: 'Market cap', value: money(card.market.market_cap_usd) }
  if (card.market.fdv_usd !== null) return { label: 'FDV', value: money(card.market.fdv_usd) }
  return { label: 'Market cap', value: 'n/a' }
}

function minutesAgo(iso: string, now: number): number {
  return Math.max(0, (now - Date.parse(iso)) / 60_000)
}

function ago(minutes: number): string {
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${Math.round(minutes)}m ago`
  return `${Math.round(minutes / 60)}h ago`
}

/** Freshness of a card's market evidence. Stale carried data is never shown as fresh. */
export function freshness(card: ScoutCard, now: number): { label: string; stale: boolean } {
  const minutes = minutesAgo(card.freshness.observed_at, now)
  const old = minutes < 1 ? 'under 1m old' : `${Math.round(minutes)}m old`
  if (card.freshness.status === 'STALE_CARRIED') {
    return { label: `Stale market data · ${old}`, stale: true }
  }
  return { label: `Market data ${old}`, stale: false }
}

export function updatedLabel(iso: string | null, now: number): string | null {
  return iso === null ? null : `Updated ${ago(minutesAgo(iso, now))}`
}

export function groupByStage(cards: ScoutCard[]): { stage: Stage; cards: ScoutCard[] }[] {
  return STAGE_ORDER.map((stage) => ({ stage, cards: cards.filter((c) => c.stage === stage) })).filter(
    (group) => group.cards.length > 0,
  )
}

/** The Analyze request for a Scout candidate: the exact chain + contract / mint travels as a
 *  structured `asset` (the backend never re-searches by ticker); the text names the token for
 *  the conversation and later follow-ups. */
export function analyzeRequest(card: ScoutCard): {
  content: string
  options: { asset: AssetRef; scout: ScoutContext }
} {
  const name = card.symbol ?? card.address
  return {
    content: `Analyze ${name}: ${card.address} on ${card.chain_label}`,
    options: { asset: card.analyze, scout: { symbol: name, momentum: card.scout_momentum } },
  }
}
