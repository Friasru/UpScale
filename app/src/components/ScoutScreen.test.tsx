// Scout in the app, driven through fetch with a REAL backend view: the fixture is
// `build_view(...)` output from backend/upscale/scout_api.py (see test_scout_api.py).
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import live from '../test/fixtures/chat-btc-5m-live.json'
import scoutFixture from '../test/fixtures/scout-view.json'
import type { ScoutView } from '../types/scout'
import { ScoutScreen } from './ScoutScreen'

const fixture = scoutFixture as ScoutView
const OBSERVED = Date.parse(fixture.candidates[0].freshness.observed_at)

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

type Handler = (url: string, init?: RequestInit) => Response | Promise<Response>

function serve(handler: Handler) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => handler(String(input), init))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function card(symbol: string): HTMLElement {
  const article = screen.getAllByRole('article').find((a) => within(a).queryByText(symbol, { exact: true }))
  if (!article) throw new Error(`no card for ${symbol}`)
  return article
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(OBSERVED + 3 * 60_000) // three minutes after the scan observed the markets
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('Scout screen', () => {
  it('renders the ranked candidates grouped by the backend stage', async () => {
    serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })

    const groups = screen.getAllByRole('region').filter((r) => r.tagName === 'SECTION')
    expect(groups.map((g) => g.getAttribute('aria-label'))).toEqual(['ACCELERATING', 'STEADY', 'CROWDED', 'FADING'])
    expect(screen.getAllByRole('article')).toHaveLength(fixture.candidates.length)
    // The stage is the backend's own, never renamed in the data.
    const a = card('A')
    expect(within(a).getByText('Accelerating').getAttribute('data-stage')).toBe('ACCELERATING')
    expect(within(card('D')).getByText('Fading').getAttribute('data-stage')).toBe('FADING')
    // Growth Scout's own surfaced reasons.
    for (const reason of fixture.candidates[0].reasons) expect(within(a).getByText(reason)).toBeTruthy()
    expect(screen.getByText(/Updated 3m ago/)).toBeTruthy()
  })

  it('shows market cap when trustworthy, FDV otherwise, without fake precision', async () => {
    serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    const a = card('A')
    expect(within(a).getByText('Market cap')).toBeTruthy()
    expect(within(a).getByText('$400K')).toBeTruthy()
    const b = card('B') // the backend reported no market cap: FDV, labeled as FDV
    expect(within(b).queryByText('Market cap')).toBeNull()
    expect(within(b).getByText('FDV')).toBeTruthy()
    expect(within(b).getByText('$900K')).toBeTruthy()
    expect(within(a).getByText('$80K')).toBeTruthy() // liquidity, 2 significant digits
  })

  it('labels stale carried market data and never presents it as fresh', async () => {
    serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    expect(within(card('B')).getByText('Stale market data · 3m old')).toBeTruthy()
    expect(within(card('B')).getByText('Stale market data')).toBeTruthy() // the flag
    expect(within(card('A')).getByText('Market data 3m old')).toBeTruthy()
    expect(within(card('A')).queryByText(/Stale/)).toBeNull()
  })

  it('shows the safety status and serious flags, never "safe"', async () => {
    serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    expect(within(card('A')).getByText('Safety checks complete')).toBeTruthy()
    expect(within(card('B')).getByText('Insufficient safety data')).toBeTruthy()
    expect(within(card('C')).getByText('Thin-market pump')).toBeTruthy()
    const text = document.body.textContent ?? ''
    expect(text).not.toMatch(/\bSafe\b|\bLegit\b/)
  })

  it('shows advanced evidence only under See more', async () => {
    serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    const a = card('A')
    expect(within(a).queryByText('Canonical id')).toBeNull()
    fireEvent.click(within(a).getByRole('button', { name: 'See more' }))
    const more = within(a).getByRole('region', { name: 'A details' })
    for (const title of ['Identity', 'Market', 'Social', 'Scoring', 'Safety', 'Freshness']) {
      expect(within(more).getByText(title)).toBeTruthy()
    }
    expect(within(more).getByText(fixture.candidates[0].canonical_id)).toBeTruthy()
    expect(within(more).getByText(/^unavailable \(X credits are exhausted/)).toBeTruthy() // not zero
    expect(within(more).getByText('Stage adjustment')).toBeTruthy()
    fireEvent.click(within(a).getByRole('button', { name: 'See less' }))
    expect(within(a).queryByText('Canonical id')).toBeNull()
  })

  it('calls it Scout Momentum, never confidence or probability', async () => {
    serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    expect(within(card('A')).getByText('Scout Momentum')).toBeTruthy()
    expect(within(card('A')).getByText('95')).toBeTruthy()
    // Only the disclaimer mentions these words, to say Scout Momentum is none of them.
    const disclaimer = screen.getByText(/is not a probability of profit/)
    expect(disclaimer.textContent).toContain('not a probability of profit, an expected return or a BUY confidence')
    const text = (document.body.textContent ?? '').replace(disclaimer.textContent ?? '', '').toLowerCase()
    for (const word of ['confidence', 'probability', 'win rate', 'chance', 'success rate']) {
      expect(text).not.toContain(word)
    }
  })

  it('shows friendly warnings instead of failing', async () => {
    serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    expect(screen.getByText(/Some social sources are unavailable: X \(credits exhausted\)/)).toBeTruthy()
    expect(screen.getByText(/partially unavailable/)).toBeTruthy()
    expect(screen.queryByText(/Scout failed/)).toBeNull()
  })

  it('filters by re-querying the backend ranking, never re-scoring', async () => {
    const fetchMock = serve(() => json(fixture))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    fireEvent.change(screen.getByLabelText('Chain'), { target: { value: 'solana' } })
    fireEvent.change(screen.getByLabelText('Stage'), { target: { value: 'ACCELERATING' } })
    fireEvent.change(screen.getByLabelText('Minimum liquidity'), { target: { value: '50000' } })
    fireEvent.change(screen.getByLabelText('Number of candidates'), { target: { value: '20' } })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(5))
    const last = new URL(String(fetchMock.mock.calls[4][0]))
    expect(last.pathname).toBe('/scout')
    expect(Object.fromEntries(last.searchParams)).toEqual({
      limit: '20',
      chain: 'solana',
      stage: 'ACCELERATING',
      min_liquidity: '50000',
    })
  })

  it('shows the empty state when nothing matches', async () => {
    serve(() => json({ ...fixture, status: 'empty', candidates: [], matching: 0 }))
    render(<ScoutScreen onAnalyze={() => {}} />)
    expect(await screen.findByText('No candidates right now')).toBeTruthy()
    expect(screen.getByText('Nothing matches these filters.')).toBeTruthy()
  })
})

describe('Scout refresh', () => {
  it('keeps the results while refreshing and ignores double clicks', async () => {
    let release: (r: Response) => void = () => {}
    const fetchMock = serve((_url, init) =>
      init?.method === 'POST' ? new Promise<Response>((resolve) => (release = resolve)) : json(fixture),
    )
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    const button = screen.getByRole('button', { name: 'Refresh' })
    fireEvent.click(button)
    fireEvent.click(button)
    const busy = await screen.findByRole('button', { name: 'Refreshing…' })
    expect((busy as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('A', { exact: true })).toBeTruthy() // still on screen
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1)
    await act(async () => release(json(fixture)))
    expect(await screen.findByRole('button', { name: 'Refresh' })).toBeTruthy()
  })

  it('keeps the last results when a refresh fails', async () => {
    serve((_url, init) => (init?.method === 'POST' ? json({ detail: 'boom' }, 500) : json(fixture)))
    render(<ScoutScreen onAnalyze={() => {}} />)
    await screen.findByText('A', { exact: true })
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    expect(await screen.findByText(/Couldn't refresh Scout \(Backend returned 500\)/)).toBeTruthy()
    expect(screen.getByText('A', { exact: true })).toBeTruthy()
  })

  it('reports a backend it cannot reach', async () => {
    serve(() => {
      throw new TypeError('Failed to fetch')
    })
    render(<ScoutScreen onAnalyze={() => {}} />)
    expect(await screen.findByText("Couldn't reach Scout (Failed to fetch).")).toBeTruthy()
  })
})

describe('Analyze from Scout', () => {
  it('sends the exact canonical identity and shows the unchanged decision card', async () => {
    const fetchMock = serve((url) => (url.includes('/scout') ? json(fixture) : json(live)))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Scout' }))
    await screen.findByText('A', { exact: true })
    fireEvent.click(within(card('A')).getByRole('button', { name: 'Analyze →' }))

    await screen.findByRole('region', { name: 'Decision' }) // the existing compact card
    const [, init] = fetchMock.mock.calls.find(([u]) => String(u).endsWith('/chat'))!
    const body = JSON.parse(String(init?.body))
    const a = fixture.candidates[0]
    expect(body.asset).toEqual(a.analyze) // chain + mint, exactly as Scout knows it
    expect(body.asset).toMatchObject({ chain: 'solana', address: a.address })
    expect(body.messages.at(-1).content).toBe(`Analyze A: ${a.address} on Solana`)
    // Scout Momentum and the decision stay separate concepts.
    expect(screen.getByText('From Scout · Scout Momentum 95')).toBeTruthy()
    const decision = screen.getByRole('region', { name: 'Decision' })
    expect(decision.textContent).not.toContain('Scout')
    expect(decision.textContent).toContain('WAIT')
  })

  it('never sends Scout data with ordinary chat turns', async () => {
    const fetchMock = serve(() => json(live))
    render(<App />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'BTC 5m?' } })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    await screen.findByRole('region', { name: 'Decision' })
    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body))
    expect(body).not.toHaveProperty('asset')
    expect(Object.keys(body.messages[0])).toEqual(['role', 'content', 'attachments'])
  })
})
