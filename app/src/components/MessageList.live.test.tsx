// Regression test on a REAL /chat response ("Should I buy BTC right now on the 5m?"), captured
// from the running backend and saved verbatim, driven through fetch → sendChat → useChat.
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useChat } from '../hooks/useChat'
import live from '../test/fixtures/chat-btc-5m-live.json'
import { MessageList } from './MessageList'

function Chat({ question }: { question: string }) {
  const chat = useChat()
  return (
    <>
      <button onClick={() => chat.send(question, [])}>Ask</button>
      <MessageList messages={chat.messages} isSending={chat.isSending} />
    </>
  )
}

async function ask(question: string) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(live), { status: 200, headers: { 'Content-Type': 'application/json' } })),
  )
  render(<Chat question={question} />)
  fireEvent.click(screen.getByRole('button', { name: 'Ask' }))
  await screen.findByRole('region', { name: 'Decision' })
}

// A line of the full text report that the compact card never shows.
const REPORT_LINE = 'Combined analysis of BTC from 5 agent(s).'

afterEach(() => vi.unstubAllGlobals())

describe('live /chat response', () => {
  it('has the analysis shape the parser reads', () => {
    const opp = live.analysis.agent_results.find((r) => r.agent === 'opportunity')
    expect(opp).toMatchObject({ status: 'ok', mock: false, findings: { action: 'wait' } })
    expect(live.message.content).toContain(REPORT_LINE)
  })

  it('renders the compact card, not the full report, until See more', async () => {
    await ask('Should I buy BTC right now on the 5m?')

    expect(screen.getByRole('region', { name: 'Decision' }).textContent).toContain('WAIT')
    expect(screen.getByText('Buy trigger')).toBeTruthy()
    expect(screen.getByText('Sell trigger')).toBeTruthy()
    expect(screen.queryByText(REPORT_LINE, { exact: false })).toBeNull()

    const toggle = screen.getByRole('button', { name: 'See more' })
    fireEvent.click(toggle)
    expect(screen.getByRole('region', { name: 'Full analysis' }).textContent).toContain(REPORT_LINE)

    fireEvent.click(screen.getByRole('button', { name: 'See less' }))
    expect(screen.queryByText(REPORT_LINE, { exact: false })).toBeNull()
    expect(screen.getByRole('button', { name: 'See more' })).toBeTruthy()
  })
})
