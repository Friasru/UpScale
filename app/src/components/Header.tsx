interface HeaderProps {
  onNewChat: () => void
  canReset: boolean
}

export function Header({ onNewChat, canReset }: HeaderProps) {
  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          ↗
        </span>
        <span className="brand-name">UpScale</span>
      </div>
      <button type="button" className="button-ghost" onClick={onNewChat} disabled={!canReset}>
        + New chat
      </button>
    </header>
  )
}
