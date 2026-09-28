import { useEffect, useState, type ChangeEvent } from 'react'
import { useScout } from '../hooks/useScout'
import { groupByStage, updatedLabel } from '../lib/scout'
import type { ChainFilter, ScoutCard as Card, ScoutLimit, StageFilter } from '../types/scout'
import { ScoutCard } from './ScoutCard'

interface ScoutScreenProps {
  onAnalyze: (card: Card) => void
}

const CHAINS: { value: ChainFilter | ''; label: string }[] = [
  { value: '', label: 'All chains' },
  { value: 'solana', label: 'Solana' },
  { value: 'ethereum', label: 'Ethereum' },
  { value: 'base', label: 'Base' },
  { value: 'bsc', label: 'BSC' },
]
const STAGES: { value: StageFilter | ''; label: string }[] = [
  { value: '', label: 'All stages' },
  { value: 'ACCELERATING', label: 'Accelerating' },
  { value: 'EARLY', label: 'Early' },
  { value: 'NEW', label: 'New' },
]
const LIQUIDITY: { value: string; label: string }[] = [
  { value: '', label: 'Any liquidity' },
  { value: '10000', label: '$10K+ liquidity' },
  { value: '50000', label: '$50K+ liquidity' },
  { value: '100000', label: '$100K+ liquidity' },
]

/** "What new or early cryptocurrencies are showing the strongest real activity right
 *  now?" Growth Scout's top candidates, grouped by stage. */
export function ScoutScreen({ onAnalyze }: ScoutScreenProps) {
  const { query, setQuery, view, loading, refreshing, error, refresh } = useScout()
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 30_000) // relative times only
    return () => window.clearInterval(timer)
  }, [])

  const select =
    <K extends 'chain' | 'stage'>(key: K) =>
    (event: ChangeEvent<HTMLSelectElement>) =>
      setQuery((q) => ({ ...q, [key]: event.target.value || null }))

  const busy = refreshing || view?.refreshing === true
  const updated = updatedLabel(view?.computed_at ?? null, now)
  const groups = groupByStage(view?.candidates ?? [])

  return (
    <div className="scout">
      <div className="scout-toolbar">
        <div className="scout-filters">
          <select
            aria-label="Number of candidates"
            value={query.limit}
            onChange={(e) => setQuery((q) => ({ ...q, limit: Number(e.target.value) as ScoutLimit }))}
          >
            <option value={10}>Top 10</option>
            <option value={20}>Top 20</option>
          </select>
          <select aria-label="Chain" value={query.chain ?? ''} onChange={select('chain')}>
            {CHAINS.map((c) => (
              <option key={c.label} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
          <select aria-label="Stage" value={query.stage ?? ''} onChange={select('stage')}>
            {STAGES.map((s) => (
              <option key={s.label} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
          <select
            aria-label="Minimum liquidity"
            value={query.minLiquidity === null ? '' : String(query.minLiquidity)}
            onChange={(e) =>
              setQuery((q) => ({ ...q, minLiquidity: e.target.value ? Number(e.target.value) : null }))
            }
          >
            {LIQUIDITY.map((l) => (
              <option key={l.label} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
        <div className="scout-refresh">
          {updated && <span className="decision-muted">{updated}</span>}
          <button type="button" className="button-ghost" onClick={() => void refresh()} disabled={busy}>
            {busy ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <p className="scout-notice scout-error" role="alert">
          {view ? `Couldn't refresh Scout (${error}). Showing the last results.` : `Couldn't reach Scout (${error}).`}
        </p>
      )}
      {view?.error && (
        <p className="scout-notice scout-error" role="alert">
          {view.error}
        </p>
      )}
      {view?.warnings.map((w) => (
        <p key={w} className="scout-notice">
          {w}
        </p>
      ))}

      {!view && loading && (
        <div className="empty-state" aria-busy="true">
          <p>Scanning markets…</p>
        </div>
      )}

      {view?.status === 'empty' && (
        <div className="empty-state">
          <h1>No candidates right now</h1>
          <p>
            {view.ranked > 0
              ? 'Nothing matches these filters.'
              : 'Scout found nothing with enough real activity. Try Refresh later.'}
          </p>
        </div>
      )}

      {groups.map((group) => (
        <section key={group.stage} className="scout-group" aria-label={group.stage}>
          <h2 className="scout-group-title">{group.stage.replace('_', ' ')}</h2>
          {group.cards.map((card) => (
            <ScoutCard key={card.canonical_id} card={card} now={now} onAnalyze={onAnalyze} />
          ))}
        </section>
      ))}

      {view && view.candidates.length > 0 && <p className="decision-footnote scout-disclaimer">{view.disclaimer}</p>}
    </div>
  )
}
