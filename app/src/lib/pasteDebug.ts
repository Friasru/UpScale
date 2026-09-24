// TEMPORARY paste diagnostics. Remove this file (and its imports) once clipboard
// paste is confirmed working in the desktop app.
//
// Logs go to the WebView devtools console and, in `tauri dev` builds, to the terminal
// through tauri-plugin-log (the webview console isn't visible there otherwise).

export const PASTE_DEBUG = true

interface TauriInternals {
  invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>
}

export function pasteLog(message: string, detail?: unknown) {
  if (!PASTE_DEBUG) return
  const line = detail === undefined ? `[paste] ${message}` : `[paste] ${message} ${JSON.stringify(detail)}`
  console.info(line)
  const tauri = (window as unknown as { __TAURI_INTERNALS__?: TauriInternals }).__TAURI_INTERNALS__
  // level 3 = Info. The log plugin is only registered in debug builds, so ignore failures.
  tauri?.invoke('plugin:log|log', { level: 3, message: line }).catch(() => {})
}

/** A JSON-friendly snapshot of what a clipboard DataTransfer actually contains. */
export function describeDataTransfer(data: DataTransfer | null) {
  if (!data) return { clipboardData: null }
  return {
    types: Array.from(data.types),
    items: Array.from(data.items).map((item) => ({ kind: item.kind, type: item.type })),
    files: Array.from(data.files).map((f) => ({ name: f.name, type: f.type, size: f.size })),
    textLength: data.getData('text/plain').length,
  }
}

/**
 * Logs exactly what navigator.clipboard.read() returns, without converting anything.
 * Triggered by Ctrl+Shift+V in the Composer.
 */
export async function inspectClipboard() {
  pasteLog('diag: start', {
    hasClipboard: typeof navigator.clipboard !== 'undefined',
    hasRead: typeof navigator.clipboard?.read === 'function',
    hasReadText: typeof navigator.clipboard?.readText === 'function',
    isSecureContext: window.isSecureContext,
  })
  if (typeof navigator.clipboard?.read !== 'function') return

  const started = performance.now()
  let items: ClipboardItems
  try {
    items = await navigator.clipboard.read()
  } catch (err) {
    const name = err instanceof Error ? err.name : typeof err
    pasteLog('diag: clipboard.read() failed', { name, error: String(err), ms: Math.round(performance.now() - started) })
    return
  }
  pasteLog('diag: clipboard.read() succeeded', { itemCount: items.length, ms: Math.round(performance.now() - started) })

  for (const [i, item] of items.entries()) {
    pasteLog(`diag: item ${i} types`, { types: Array.from(item.types), presentationStyle: item.presentationStyle })
    for (const type of item.types) {
      try {
        const blob = await item.getType(type)
        pasteLog(`diag: item ${i} getType(${type}) ok`, { blobType: blob.type, size: blob.size })
      } catch (err) {
        pasteLog(`diag: item ${i} getType(${type}) failed`, String(err))
      }
    }
  }
  pasteLog('diag: done')
}
