import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchScout, refreshScout } from '../lib/api'
import type { ScoutQuery, ScoutView } from '../types/scout'

export const DEFAULT_QUERY: ScoutQuery = { limit: 10, chain: null, stage: null, minLiquidity: null }

/** Scout's ranking: loaded once, re-queried when filters change (cheap: the backend
 *  filters its latest ranking), re-run only on an explicit Refresh. No polling. */
function keyOf(query: ScoutQuery): string {
  return JSON.stringify(query)
}

/** Scout's ranking: loaded once, re-queried when filters change (cheap: the backend
 *  filters its latest ranking), re-run only on an explicit Refresh. No polling. */
export function useScout() {
  const [query, setQuery] = useState<ScoutQuery>(DEFAULT_QUERY)
  const [view, setView] = useState<ScoutView | null>(null)
  // The query whose request last settled (loaded or failed): loading while it differs.
  const [settled, setSettled] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const refreshingRef = useRef(false)
  const latest = useRef(0)

  useEffect(() => {
    // Filters and top-N re-query the backend's latest ranking; a newer query (or a
    // refresh) supersedes this one.
    const request = ++latest.current
    fetchScout(query)
      .then((result) => {
        if (request !== latest.current) return
        setView(result)
        setError(null)
      })
      .catch((err: unknown) => {
        if (request === latest.current) setError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (request === latest.current) setSettled(keyOf(query))
      })
  }, [query])

  const refresh = useCallback(async () => {
    if (refreshingRef.current) return // no duplicate refreshes
    refreshingRef.current = true
    setRefreshing(true)
    const request = ++latest.current
    try {
      const result = await refreshScout(query)
      if (request === latest.current) {
        setView(result) // existing results stay on screen until these arrive
        setError(null)
      }
    } catch (err) {
      if (request === latest.current) setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (request === latest.current) setSettled(keyOf(query))
      refreshingRef.current = false
      setRefreshing(false)
    }
  }, [query])

  const loading = settled !== keyOf(query)
  return { query, setQuery, view, loading, refreshing, error, refresh }
}
