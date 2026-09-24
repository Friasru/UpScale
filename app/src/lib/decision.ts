// Compact BUY / SELL / WAIT view built from the Opportunity agent's structured findings
// (backend/upscale/services/opportunity.py `OpportunityAssessment`). Never parses message text.
import type { Analysis } from '../types/chat'

export type Action = 'buy' | 'sell' | 'wait'

interface Zone {
  lower: number
  upper: number
}

interface Trigger {
  action: 'buy' | 'sell'
  condition: string
  price: number
  basis: string
  confirmed: boolean
  live_price_beyond: boolean | null
}

interface Invalidation {
  condition: string
  price: number
  timeframe: string
}

interface Factor {
  reason: string
}

/** The subset of `OpportunityAssessment` the compact view reads. */
export interface OpportunityFindings {
  asset: string | null
  timeframe: string | null
  requested_timeframe: string | null
  action: Action
  confidence: string
  summary: string
  blocking_factors: Factor[]
  bullish_trigger: Trigger | null
  bearish_trigger: Trigger | null
  entry_zone: Zone | null
  entry_basis: string | null
  invalidation: Invalidation | null
  risk_level: string
  uncertainty_level: string
  missing_evidence: string[]
  sell_meaning: string
}

export interface DecisionRow {
  label: string
  value: string
  /** The full condition from the agent, for a tooltip / screen readers. */
  detail?: string
  note?: string
}

export interface DecisionView {
  action: Action
  /** Collapsed card: only the triggers that exist, compact (label + price). */
  triggers: DecisionRow[]
  risk: string
  /** "See more": everything else, from the same findings. */
  asset: string | null
  timeframe: string | null
  requestedTimeframe: string | null
  reason: string
  details: DecisionRow[]
  confidence: string
  note: string | null
}

const ACTIONS: readonly string[] = ['buy', 'sell', 'wait']

/** Same format as backend `upscale.formatting.usd`. */
export function usd(value: number): string {
  if (Math.abs(value) >= 1 || value === 0) {
    return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }
  const decimals = 3 - Math.floor(Math.log10(Math.abs(value)))
  return `$${value.toFixed(decimals).replace(/0+$/, '')}`
}

function zone(z: Zone): string {
  return usd(z.lower) === usd(z.upper) ? `~${usd(z.lower)}` : `~${usd(z.lower)}–${usd(z.upper)}`
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null
}

/** The opportunity findings of a real, successful Opportunity run, else null. */
export function opportunityFindings(analysis: Analysis | null | undefined): OpportunityFindings | null {
  const result = analysis?.agent_results.find((r) => r.agent === 'opportunity')
  if (!result || result.status !== 'ok' || result.mock) return null
  const f = result.findings
  if (!isRecord(f) || typeof f.action !== 'string' || !ACTIONS.includes(f.action)) return null
  return f as unknown as OpportunityFindings
}

function triggerRow(t: Trigger): DecisionRow {
  const buy = t.action === 'buy'
  let value: string
  if (!t.confirmed) value = `${buy ? 'above' : 'below'} ~${usd(t.price)}`
  else if (t.basis === 'last_close') value = `confirmed close ~${usd(t.price)}`
  else value = `closed ${buy ? 'above' : 'below'} ~${usd(t.price)}`
  return { label: buy ? 'Buy trigger' : 'Sell trigger', value, detail: t.condition }
}

/** Full trigger condition for "See more", with the live-price caveat. */
function triggerDetail(t: Trigger, tf: string | null): DecisionRow {
  const row: DecisionRow = { label: t.action === 'buy' ? 'Buy trigger' : 'Sell trigger', value: t.condition }
  if (!t.confirmed && t.live_price_beyond) {
    row.note = `Live price is already past it; the ${tf ?? 'candle'} close decides`
  }
  return row
}

function reason(f: OpportunityFindings): string {
  if (f.action === 'wait' && f.blocking_factors?.length) {
    return (
      f.blocking_factors
        .slice(0, 2)
        .map((b) => cap(b.reason))
        .join('. ') + '.'
    )
  }
  return f.summary
}

export function decisionView(f: OpportunityFindings): DecisionView {
  const tf = f.timeframe
  const present = [f.bullish_trigger, f.bearish_trigger].filter((t): t is Trigger => t != null)
  const details = present.map((t) => triggerDetail(t, tf))
  if (f.entry_zone && f.action !== 'wait') {
    details.push({
      label: f.action === 'buy' ? 'Entry zone' : 'Exit zone',
      value: zone(f.entry_zone),
      note: f.entry_basis ? cap(f.entry_basis) : undefined,
    })
  }
  if (f.invalidation) {
    details.push({
      label: 'Invalidation',
      value: `${f.invalidation.timeframe} close ${f.action === 'sell' ? 'above' : 'below'} ~${usd(f.invalidation.price)}`,
      detail: f.invalidation.condition,
    })
  }
  return {
    action: f.action,
    triggers: present.map(triggerRow),
    risk: f.risk_level ? cap(f.risk_level) : 'Unavailable',
    asset: f.asset,
    timeframe: tf,
    requestedTimeframe:
      f.requested_timeframe && f.requested_timeframe !== tf ? f.requested_timeframe : null,
    reason: reason(f),
    details,
    confidence: f.confidence ? cap(f.confidence) : 'Low',
    note: f.action === 'sell' ? f.sell_meaning || null : null,
  }
}
