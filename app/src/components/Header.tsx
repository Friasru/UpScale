export type AppView = 'chat' | 'scout'

interface HeaderProps {
  view: AppView
  onNavigate: (view: AppView) => void
  onNewChat: () => void
  canReset: boolean
}

export function Header({ view, onNavigate, onNewChat, canReset }: HeaderProps) {
  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          ↗
        </span>
        <span className="brand-name">UpScale</span>
        <nav className="nav" aria-label="Views">
          {(['chat', 'scout'] as const).map((v) => (
            <button
              key={v}
              type="button"
              className={view === v ? 'nav-item nav-item-active' : 'nav-item'}
              aria-current={view === v ? 'page' : undefined}
              onClick={() => onNavigate(v)}
            >
              {v === 'chat' ? 'Chat' : 'Scout'}
            </button>
          ))}
        </nav>
      </div>
      {view === 'chat' && (
        <button type="button" className="button-ghost" onClick={onNewChat} disabled={!canReset}>
          + New chat
        </button>
      )}
    </header>
  )
}
