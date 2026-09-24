import { useId, useState } from 'react'
import { decisionView, type DecisionRow, type OpportunityFindings } from '../lib/decision'

interface DecisionCardProps {
  findings: OpportunityFindings
  /** The complete existing text report (disclaimer included), shown under "See more". */
  fullText: string
}

function Rows({ rows }: { rows: DecisionRow[] }) {
  return (
    <dl className="decision-rows">
      {rows.map((row) => (
        <div className="decision-row" key={row.label} title={row.detail}>
          <dt>{row.label}</dt>
          <dd>
            {row.value}
            {row.note && <span className="decision-muted"> · {row.note}</span>}
          </dd>
        </div>
      ))}
    </dl>
  )
}

export function DecisionCard({ findings, fullText }: DecisionCardProps) {
  const [expanded, setExpanded] = useState(false)
  const panelId = useId()
  const view = decisionView(findings)
  const subject = [view.asset ?? 'No asset', view.timeframe].filter(Boolean).join(' ')

  // Collapsed: action, the triggers that exist, risk. Nothing else — speed first.
  return (
    <div className="decision-wrap">
      <section className={`decision decision-${view.action}`} aria-label="Decision">
        <p className="decision-action">{view.action.toUpperCase()}</p>
        {view.triggers.length > 0 && <Rows rows={view.triggers} />}
        <dl className="decision-rows decision-risk">
          <div className="decision-row">
            <dt>Risk</dt>
            <dd>{view.risk}</dd>
          </div>
        </dl>
      </section>

      <button
        type="button"
        className="button-ghost decision-toggle"
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? 'See less' : 'See more'}
      </button>

      {expanded && (
        <div id={panelId} className="decision-full" aria-label="Full analysis" role="region">
          <p className="decision-subject">
            {subject}
            {view.timeframe === null && view.asset !== null && (
              <span className="decision-muted"> · timeframe unavailable</span>
            )}
            {view.requestedTimeframe && (
              <span className="decision-muted"> ({view.requestedTimeframe} requested)</span>
            )}
          </p>
          <p className="decision-reason">{view.reason}</p>
          <Rows rows={[...view.details, { label: 'Confidence', value: view.confidence }]} />
          {view.note && <p className="decision-footnote">{view.note}</p>}
          <p className="message-text">{fullText}</p>
        </div>
      )}
    </div>
  )
}
