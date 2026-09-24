import { act, fireEvent, render, renderHook, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useChat } from '../hooks/useChat'
import type { OpportunityFindings } from '../lib/decision'
import type { AgentResult, Analysis, UiMessage } from '../types/chat'
import { MessageList } from './MessageList'

const DISCLAIMER =
  'BUY / SELL / WAIT reads are rule-based decision support from the evidence shown, not guarantees or financial advice.'
const SELL_MEANING =
  'SELL means reduce or exit a long position. It is not a signal to open a short position.'

// Shaped like backend OpportunityAssessment.model_dump(mode="json").
function findings(overrides: Partial<OpportunityFindings> = {}): Record<string, unknown> {
  return {
    asset: 'BTC',
    timeframe: '5m',
    requested_timeframe: null,
    action: 'wait',
    confidence: 'medium',
    setup: null,
    confirmed: false,
    summary: 'BTC 5m: the 5m trend is mixed; bullish and bearish evidence are roughly balanced.',
    bullish_score: 2,
    bearish_score: 2,
    bullish_evidence: [],
    bearish_evidence: [],
    blocking_factors: [
      { id: 'mixed_trend', kind: 'blocker', applies_to: 'both', reason: 'the 5m trend is mixed', source: 'technical_analysis' },
      { id: 'weak', kind: 'blocker', applies_to: 'both', reason: 'volume is weak (0.5x the 20-candle average)', source: 'technical_analysis' },
    ],
    cautions: [],
    bullish_trigger: {
      action: 'buy',
      condition: '5m close above ~$84,515.00 (upper bound of the resistance zone ~$84,400.00–$84,515.00)',
      price: 84515,
      basis: 'resistance_zone_upper',
      zone: { lower: 84400, upper: 84515 },
      confirmed: false,
      live_price_beyond: false,
    },
    bearish_trigger: {
      action: 'sell',
      condition: '5m close below ~$84,263.00 (lower bound of the support zone ~$84,263.00–$84,300.00)',
      price: 84263,
      basis: 'support_zone_lower',
      zone: { lower: 84263, upper: 84300 },
      confirmed: false,
      live_price_beyond: false,
    },
    entry_zone: null,
    entry_basis: null,
    invalidation: null,
    other_invalidations: [],
    risk_level: 'medium',
    uncertainty_level: 'medium',
    missing_evidence: [],
    last_close: 84350,
    live_price: 84360,
    context: [],
    inputs: [],
    sell_meaning: SELL_MEANING,
    rules: 'Deterministic rules …',
    ...overrides,
  }
}

function result(agent: AgentResult['agent'], extra: Partial<AgentResult> = {}): AgentResult {
  return { agent, status: 'ok', mock: false, summary: `${agent} summary`, findings: {}, evidence: [], error: null, ...extra }
}

function analysis(opportunity: Record<string, unknown> | null, extra: AgentResult[] = []): Analysis {
  const results = [...extra, result('technical_analysis'), result('risk')]
  if (opportunity) results.push(result('opportunity', { findings: opportunity, summary: 'WAIT\n…' }))
  return {
    mock: false,
    summary: 'Combined analysis of BTC from 4 agent(s).',
    assets: ['BTC'],
    agents_used: results.map((r) => r.agent),
    agent_results: results,
    uncertainty: { level: 'medium', notes: [] },
    disclaimer: DISCLAIMER,
  }
}

const FULL_TEXT = `WAIT\nBTC 5m: …\n\nAgents:\n• Technical: detailed evidence line\n\n${DISCLAIMER}`

function reply(a: Analysis | null, content = FULL_TEXT): UiMessage {
  return { id: crypto.randomUUID(), role: 'assistant', content, attachments: [], analysis: a }
}

const question: UiMessage = { id: 'q', role: 'user', content: 'Should I buy BTC?', attachments: [] }

function renderReply(message: UiMessage, before: UiMessage[] = [question]) {
  render(<MessageList messages={[...before, message]} isSending={false} />)
  return screen.getByRole('region', { name: 'Decision' })
}

function rowValue(root: HTMLElement, label: string): string | null {
  const dt = within(root).queryByText(label, { selector: 'dt' })
  return dt?.nextElementSibling?.textContent ?? null
}

/** Every "label: value" line of the collapsed card, in order. */
function collapsedLines(card: HTMLElement): string[] {
  return [...card.querySelectorAll('.decision-row')].map(
    (r) => `${r.querySelector('dt')!.textContent}: ${r.querySelector('dd')!.textContent}`,
  )
}

function openSeeMore(): HTMLElement {
  const toggle = screen.getByRole('button', { name: 'See more' })
  fireEvent.click(toggle)
  return document.getElementById(toggle.getAttribute('aria-controls')!)!
}

const BUY = findings({
  action: 'buy',
  confidence: 'high',
  summary: 'BTC 1h is in an uptrend, closed above the zone ~$84,000.00–$84,200.00, MACD is above its signal line.',
  blocking_factors: [],
  bullish_trigger: {
    action: 'buy',
    condition: 'confirmed: 1h close $84,350.00 above ~$84,200.00',
    price: 84200,
    basis: 'resistance_zone_upper',
    confirmed: true,
    live_price_beyond: null,
  },
  bearish_trigger: null,
  entry_zone: { lower: 84000, upper: 84200 },
  entry_basis: 'retest of the broken zone',
  invalidation: {
    condition: '1h close back below ~$84,000.00 (lower bound of the broken zone)',
    price: 84000,
    timeframe: '1h',
  },
  timeframe: '1h',
  risk_level: 'low',
})

const SELL = findings({
  action: 'sell',
  confidence: 'low',
  summary: 'ETH 4h is in a downtrend, MACD is below its signal line.',
  asset: 'ETH',
  timeframe: '4h',
  blocking_factors: [],
  bullish_trigger: null,
  bearish_trigger: {
    action: 'sell',
    condition: 'confirmed: 4h close $3,100.00 in a downtrend with MACD below its signal line',
    price: 3100,
    basis: 'last_close',
    confirmed: true,
    live_price_beyond: null,
  },
  entry_zone: { lower: 3150, upper: 3150 },
  entry_basis: 'retest of the broken zone, to exit longs',
  invalidation: { condition: '4h close above ~$3,300.00', price: 3300, timeframe: '4h' },
  risk_level: 'medium',
})

/** Nothing but action, triggers and risk may appear while collapsed. */
function expectOnlyEssentials(card: HTMLElement, action: string, lines: string[]) {
  expect(card.querySelector('.decision-action')?.textContent).toBe(action)
  expect(collapsedLines(card)).toEqual(lines)
  expect(card.textContent).toBe(action + lines.map((l) => l.replace(': ', '')).join(''))
  expect(screen.queryByRole('region', { name: 'Full analysis' })).toBeNull()
  expect(screen.queryByText(DISCLAIMER)).toBeNull()
  expect(screen.queryByText(SELL_MEANING)).toBeNull()
  expect(screen.queryByText(/Confidence|Entry zone|Exit zone|Invalidation|BTC 5m|ETH 4h/)).toBeNull()
}

describe('collapsed decision card', () => {
  it('WAIT: action, both triggers and risk only', () => {
    const card = renderReply(reply(analysis(findings())))
    expect(card.className).toContain('decision-wait')
    expectOnlyEssentials(card, 'WAIT', [
      'Buy trigger: above ~$84,515.00',
      'Sell trigger: below ~$84,263.00',
      'Risk: Medium',
    ])
  })

  it('BUY: action, confirmed buy trigger and risk only; missing sell trigger is omitted', () => {
    const card = renderReply(reply(analysis(BUY)))
    expect(card.className).toContain('decision-buy')
    expectOnlyEssentials(card, 'BUY', ['Buy trigger: closed above ~$84,200.00', 'Risk: Low'])
    expect(rowValue(card, 'Sell trigger')).toBeNull()
  })

  it('SELL: action, sell trigger and risk only; long-only note is not shown', () => {
    const card = renderReply(reply(analysis(SELL)))
    expect(card.className).toContain('decision-sell')
    expectOnlyEssentials(card, 'SELL', ['Sell trigger: confirmed close ~$3,100.00', 'Risk: Medium'])
    expect(rowValue(card, 'Buy trigger')).toBeNull()
  })

  it('omits both trigger rows when neither is available (missing evidence)', () => {
    const missing = findings({
      timeframe: null,
      confidence: 'low',
      summary: 'BTC: there is no live technical analysis for BTC.',
      blocking_factors: [
        { id: 'no_technical', kind: 'blocker', applies_to: 'both', reason: 'there is no live technical analysis for BTC', source: 'opportunity' } as never,
      ],
      bullish_trigger: null,
      bearish_trigger: null,
      risk_level: 'unavailable',
      missing_evidence: ['Live technical analysis for this asset', 'Risk review'],
    })
    const card = renderReply(reply(analysis(missing)))
    expectOnlyEssentials(card, 'WAIT', ['Risk: Unavailable'])
  })
})

describe('See more', () => {
  it('reveals the full WAIT analysis locally, without another request', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    renderReply(reply(analysis(findings({ requested_timeframe: '3m' }))))
    expect(screen.queryByText(/detailed evidence line/)).toBeNull()

    const toggle = screen.getByRole('button', { name: 'See more' })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    const full = openSeeMore()
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(toggle.textContent).toBe('See less')

    expect(within(full).getByText('BTC 5m')).toBeTruthy()
    expect(within(full).getByText('(3m requested)')).toBeTruthy()
    expect(
      within(full).getByText('The 5m trend is mixed. Volume is weak (0.5x the 20-candle average).'),
    ).toBeTruthy()
    expect(rowValue(full, 'Buy trigger')).toContain('5m close above ~$84,515.00 (upper bound')
    expect(rowValue(full, 'Sell trigger')).toContain('5m close below ~$84,263.00 (lower bound')
    expect(rowValue(full, 'Confidence')).toBe('Medium')
    // The whole existing report, disclaimer included, verbatim.
    expect(full.querySelector('.message-text')?.textContent).toBe(FULL_TEXT)
    expect(within(full).getByText(/detailed evidence line/)).toBeTruthy()

    fireEvent.click(toggle)
    expect(screen.queryByText(/detailed evidence line/)).toBeNull()
    expect(fetchSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()
  })

  it('BUY: entry zone, invalidation and confidence move here', () => {
    renderReply(reply(analysis(BUY)))
    const full = openSeeMore()
    expect(within(full).getByText('BTC 1h')).toBeTruthy()
    expect(within(full).getByText(/is in an uptrend/)).toBeTruthy()
    expect(rowValue(full, 'Entry zone')).toBe('~$84,000.00–$84,200.00 · Retest of the broken zone')
    expect(rowValue(full, 'Invalidation')).toBe('1h close below ~$84,000.00')
    expect(within(full).getByText('Invalidation').parentElement?.title).toContain('broken zone')
    expect(rowValue(full, 'Confidence')).toBe('High')
    expect(within(full).queryByText(SELL_MEANING)).toBeNull()
  })

  it('SELL: exit zone, invalidation above and the long-only clarification move here', () => {
    renderReply(reply(analysis(SELL)))
    const full = openSeeMore()
    expect(within(full).getByText('ETH 4h')).toBeTruthy()
    expect(rowValue(full, 'Exit zone')).toBe('~$3,150.00 · Retest of the broken zone, to exit longs')
    expect(rowValue(full, 'Invalidation')).toBe('4h close above ~$3,300.00')
    expect(rowValue(full, 'Confidence')).toBe('Low')
    expect(within(full).getByText(SELL_MEANING)).toBeTruthy()
  })

  it('notes there when the live price is already past an unconfirmed trigger', () => {
    const f = findings()
    ;(f.bullish_trigger as Record<string, unknown>).live_price_beyond = true
    const card = renderReply(reply(analysis(f)))
    expect(rowValue(card, 'Buy trigger')).toBe('above ~$84,515.00')
    const full = openSeeMore()
    expect(rowValue(full, 'Buy trigger')).toMatch(/· Live price is already past it; the 5m close decides$/)
  })

  it('shows "No asset" and the reason when no asset was identified', () => {
    const none = findings({
      asset: null,
      timeframe: null,
      blocking_factors: [],
      bullish_trigger: null,
      bearish_trigger: null,
      summary: 'No specific cryptocurrency was identified, so there is nothing to decide on.',
    })
    renderReply(reply(analysis(none)))
    const full = openSeeMore()
    expect(within(full).getByText('No asset')).toBeTruthy()
    expect(within(full).getByText(/No specific cryptocurrency/)).toBeTruthy()
  })
})

describe('screenshot decision response', () => {
  it('uses the same compact card and keeps the user screenshot', () => {
    const shot: UiMessage = {
      id: 's',
      role: 'user',
      content: '',
      attachments: [{ name: 'chart.png', media_type: 'image/png', data: 'iVBORw0KGgo=' }],
    }
    const a = analysis(findings({ timeframe: '15m' }), [
      result('vision', { findings: { charts: [{ normalized_timeframe: '15m' }] } }),
    ])
    const card = renderReply(reply(a), [shot])
    expect(collapsedLines(card)).toEqual([
      'Buy trigger: above ~$84,515.00',
      'Sell trigger: below ~$84,263.00',
      'Risk: Medium',
    ])
    expect(within(card).queryByText(/15m/)).toBeNull()
    expect(screen.getByRole('img', { name: 'chart.png' })).toBeTruthy()
    expect(within(openSeeMore()).getByText('BTC 15m')).toBeTruthy()
  })
})

describe('non-decision responses', () => {
  it.each([
    ['no analysis', null],
    [
      'an education-only analysis (e.g. "What is RSI?")',
      {
        ...analysis(null),
        assets: [],
        agents_used: ['education'],
        agent_results: [result('education', { summary: 'RSI is a momentum oscillator.' })],
        disclaimer: 'General educational explanation, not financial advice.',
      },
    ],
    ['analysis without an opportunity result', analysis(null)],
    ['a failed opportunity result', { ...analysis(null), agent_results: [result('opportunity', { status: 'error', error: 'boom' })] }],
    ['a mock opportunity result', { ...analysis(null), agent_results: [result('opportunity', { mock: true, findings: findings() })] }],
  ])('renders %s as plain text', (_, a) => {
    const text = 'RSI is a momentum oscillator.'
    render(<MessageList messages={[question, reply(a as Analysis | null, text)]} isSending={false} />)
    expect(screen.queryByRole('region', { name: 'Decision' })).toBeNull()
    expect(screen.queryByRole('button', { name: /See more/ })).toBeNull()
    expect(screen.getByText(text).className).toBe('message-text')
  })
})

describe('useChat', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('keeps the structured analysis on the reply but never sends it back in history', async () => {
    const a = analysis(findings())
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({ message: { role: 'assistant', content: FULL_TEXT, attachments: [] }, analysis: a }),
      ),
    )
    vi.stubGlobal('fetch', fetchMock)
    const { result: hook } = renderHook(() => useChat())

    await act(() => hook.current.send('Should I buy BTC?', []))
    expect(hook.current.messages[1].analysis).toEqual(a)

    await act(() => hook.current.send('And ETH?', []))
    const body = JSON.parse((fetchMock.mock.calls[1] as unknown as [string, RequestInit])[1].body as string)
    expect(body.messages).toHaveLength(3)
    for (const m of body.messages) expect(Object.keys(m).sort()).toEqual(['attachments', 'content', 'role'])
  })
})
