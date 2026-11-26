export type MarketId = 'jp' | 'us'

export type SymbolMatch = { symbol: string; name: string }

export type AnalysisRequest = {
  start_date: string
  end_date: string
  strategy: string
  market: MarketId
  tickers: string[]
}

export type AnalysisConfig = {
  defaults: AnalysisRequest
  strategies: Record<string, string>
  markets: Record<MarketId, {
    label: string
    currency: string
    benchmark_ticker: string
    benchmark_label: string
    default_tickers: string[]
  }>
  max_tickers: number
  estimation_window: number
  holding_period: number
  transaction_cost_bps: number
  maximum_date: string
}

export type PerformanceRow = {
  label: string
  cagr: number | null
  volatility: number | null
  sharpe: number | null
  max_drawdown: number | null
}

export type AnalysisResponse = {
  requested: AnalysisRequest
  benchmark_label: string
  benchmark_ticker: string
  currency: string
  evaluation: { start_date: string; end_date: string; observations: number }
  performance: PerformanceRow[]
  cumulative_returns: { date: string; strategy: number; benchmark: number }[]
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch (error) {
    if (init?.signal?.aborted) throw error
    throw new Error('分析APIに接続できません。サーバーの起動を確認してください。')
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail
    if (typeof detail === 'string') throw new Error(detail)
    if (response.status === 422) throw new Error('銘柄、期間、戦略の入力を確認してください。')
    throw new Error('分析APIから応答を取得できませんでした。再度お試しください。')
  }

  return response.json() as Promise<T>
}

export function fetchConfig(signal: AbortSignal) {
  return requestJson<AnalysisConfig>('/api/config', { signal })
}

export function searchSymbols(market: MarketId, query: string, signal: AbortSignal) {
  const params = new URLSearchParams({ market, q: query })
  return requestJson<SymbolMatch[]>(`/api/symbols?${params}`, { signal })
}

export function analyze(request: AnalysisRequest) {
  return requestJson<AnalysisResponse>('/api/analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
}
