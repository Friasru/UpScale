import { useId, useState } from 'react'
import { age, capacity, freshness, money, stageLabel } from '../lib/scout'
import type { ScoutCard as Card } from '../types/scout'

interface ScoutCardProps {
  card: Card
  now: number
  onAnalyze: (card: Card) => void
}

/** One Scout candidate: what it is, why it surfaced, what to watch out for. Advanced
 *  evidence stays behind "See more". Scout Momentum ranks discovery; it is not a trade
 *  confidence (Analyze gives BUY / SELL / WAIT). */
export function ScoutCard({ card, now, onAnalyze }: ScoutCardProps) {
  const [expanded, setExpanded] = useState(false)
  const panelId = useId()
  const cap = capacity(card)
  const fresh = freshness(card, now)
  const title = card.symbol ?? `${card.address.slice(0, 4)}…${card.address.slice(-4)}`

  return (
    <article className="scout-card" aria-label={`${title} on ${card.chain_label}`}>
      <header className="scout-card-head">
        <div>
          <p className="scout-symbol">
            {title}
            {card.name && <span className="decision-muted"> {card.name}</span>}
          </p>
          <p className="scout-sub">
            <span className={`scout-stage scout-stage-${card.stage.toLowerCase()}`} data-stage={card.stage}>
              {stageLabel(card.stage)}
            </span>
            <span className="decision-muted"> · {card.chain_label}</span>
          </p>
        </div>
        <div className="scout-momentum" title="Discovery ranking, not a trade confidence">
          <span className="scout-momentum-value">{card.scout_momentum}</span>
          <span className="scout-momentum-label">Scout Momentum</span>
        </div>
      </header>

      <dl className="decision-rows scout-facts">
        <div className="decision-row">
          <dt>{cap.label}</dt>
          <dd>{cap.value}</dd>
        </div>
        <div className="decision-row">
          <dt>Liquidity</dt>
          <dd>{money(card.market.liquidity_usd)}</dd>
        </div>
        <div className="decision-row">
          <dt>Age</dt>
          <dd>{age(card.market.age_hours)}</dd>
        </div>
      </dl>

      {card.reasons.length > 0 && (
        <ul className="scout-reasons">
          {card.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}

      <div className="scout-status">
        <span className="scout-safety">{card.safety.label}</span>
        {card.safety.flags.map((flag) => (
          <span key={flag.label} className={`scout-flag scout-flag-${flag.severity}`}>
            {flag.label}
          </span>
        ))}
        <span className={fresh.stale ? 'scout-fresh scout-stale' : 'scout-fresh'}>{fresh.label}</span>
      </div>

      <div className="scout-actions">
        <button type="button" className="send-button scout-analyze" onClick={() => onAnalyze(card)}>
          Analyze →
        </button>
        <button
          type="button"
          className="button-ghost decision-toggle"
          aria-expanded={expanded}
          aria-controls={panelId}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? 'See less' : 'See more'}
        </button>
      </div>

      {expanded && (
        <div id={panelId} className="scout-more" role="region" aria-label={`${title} details`}>
          {card.details.map((section) => (
            <section key={section.title}>
              <h3 className="scout-more-title">{section.title}</h3>
              <dl className="decision-rows">
                {section.rows.map((row, i) => (
                  <div className="decision-row" key={`${row.label}-${i}`}>
                    <dt>{row.label}</dt>
                    <dd className="scout-more-value">{row.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      )}
    </article>
  )
}
