// Mirrors backend/upscale/scout_api.py (ScoutView). The UI renders this as is: Growth
// Scout's ranking, stages, reasons and safety come from the backend, never re-derived here.

export type Stage =
  | 'NEW'
  | 'EARLY'
  | 'ACCELERATING'
  | 'STEADY'
  | 'CROWDED'
  | 'FADING'
  | 'INSUFFICIENT_DATA'

export type ChainFilter = 'solana' | 'ethereum' | 'base' | 'bsc'
export type StageFilter = 'ACCELERATING' | 'EARLY' | 'NEW'
export type ScoutLimit = 10 | 20

/** The exact identity Analyze receives (chain + contract / mint). */
export interface AssetRef {
  chain: string
  address: string
  symbol: string | null
  name: string | null
  pool_address: string | null
  source: 'scout'
}

export interface ScoutFlag {
  code: string
  label: string
  severity: string
}

export interface ScoutRow {
  label: string
  value: string
}

export interface ScoutSection {
  title: string
  rows: ScoutRow[]
}

export interface ScoutCard {
  rank: number
  canonical_id: string
  chain: string
  chain_label: string
  address: string
  symbol: string | null
  name: string | null
  stage: Stage
  stage_reasons: string[]
  /** Discovery ranking 0..100. Not a trade confidence, probability or expected return. */
  scout_momentum: number
  market: {
    market_cap_usd: number | null
    fdv_usd: number | null
    liquidity_usd: number | null
    age_hours: number | null
  }
  safety: { status: string; label: string; flags: ScoutFlag[] }
  freshness: {
    status: 'CURRENT' | 'STALE_CARRIED'
    observed_at: string
    snapshot_age_minutes: number
  }
  reasons: string[]
  details: ScoutSection[]
  analyze: AssetRef
}

export interface ScoutFilters {
  chain: ChainFilter | null
  stage: StageFilter | null
  min_liquidity_usd: number | null
}

export interface ScoutView {
  status: 'ok' | 'empty' | 'unavailable'
  computed_at: string | null
  mode: string | null
  limit: number
  filters: ScoutFilters
  ranked: number
  matching: number
  candidates: ScoutCard[]
  warnings: string[]
  refreshing: boolean
  error: string | null
  disclaimer: string
}

export interface ScoutQuery {
  limit: ScoutLimit
  chain: ChainFilter | null
  stage: StageFilter | null
  minLiquidity: number | null
}
