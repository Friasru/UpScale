import { Composer } from './components/Composer'
import { Header } from './components/Header'
import { MessageList } from './components/MessageList'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, isSending, send, newChat } = useChat()

  return (
    <div className="app">
      <Header onNewChat={newChat} canReset={messages.length > 0} />
      <main className="chat">
        <MessageList messages={messages} isSending={isSending} />
      </main>
      <div className="composer-wrap">
        <Composer onSend={send} disabled={isSending} />
      </div>
    </div>
  )
}
