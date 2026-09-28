import { useState } from 'react'
import { Composer } from './components/Composer'
import { Header, type AppView } from './components/Header'
import { MessageList } from './components/MessageList'
import { ScoutScreen } from './components/ScoutScreen'
import { useChat } from './hooks/useChat'
import { analyzeRequest } from './lib/scout'
import type { ScoutCard } from './types/scout'

export default function App() {
  const { messages, isSending, send, newChat } = useChat()
  const [view, setView] = useState<AppView>('chat')

  function analyze(card: ScoutCard) {
    const { content, options } = analyzeRequest(card)
    setView('chat')
    void send(content, [], options)
  }

  return (
    <div className="app">
      <Header view={view} onNavigate={setView} onNewChat={newChat} canReset={messages.length > 0} />
      {/* Chat stays mounted, so switching views never loses the conversation. */}
      <main className="chat" hidden={view !== 'chat'}>
        <MessageList messages={messages} isSending={isSending} />
      </main>
      <div className="composer-wrap" hidden={view !== 'chat'}>
        <Composer onSend={send} disabled={isSending} />
      </div>
      {view === 'scout' && (
        <main className="chat">
          <ScoutScreen onAnalyze={analyze} />
        </main>
      )}
    </div>
  )
}
