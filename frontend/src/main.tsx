import { lazy, StrictMode, Suspense, useEffect, useState, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import {
  analyze,
  fetchConfig,
  type AnalysisConfig,
  type AnalysisRequest,
  type AnalysisResponse,
  type MarketId,
  type PerformanceRow,
} from './api'
import SymbolPicker from './SymbolPicker'
import './styles.css'

const ReturnChart = lazy(() => import('./ReturnChart'))
const percent = new Intl.NumberFormat('ja-JP', { style: 'percent', minimumFractionDigits: 2 })
const decimal = new Intl.NumberFormat('ja-JP', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})
const metricColumns = [
  { key: 'period_return', label: '期間リターン' },
  { key: 'cagr', label: '年率リターン' },
  { key: 'volatility', label: '年率ボラティリティ' },
  { key: 'sharpe', label: 'シャープレシオ' },
  { key: 'max_drawdown', label: '最大ドローダウン' },
] as const

type AnalysisForm = Omit<AnalysisRequest, 'tickers'> & { tickersByMarket: Record<MarketId, string[]> }

function formatMetric(row: PerformanceRow, key: (typeof metricColumns)[number]['key']) {
  const value = row[key]
  if (value === null) return '—'
  return (key === 'sharpe' ? decimal : percent).format(value)
}

function Result({ result }: { result: AnalysisResponse }) {
  const { evaluation, performance, cumulative_returns: points, requested } = result
  const strategyLabel = performance[0]?.label ?? result.requested.strategy
  const entries = Object.entries(result.entry_dates).sort((left, right) => left[1].localeCompare(right[1]))
  const portfolioDates = new Map<string, string[]>()
  for (const [symbol, day] of entries) {
    const group = portfolioDates.get(day) ?? []
    group.push(symbol)
    portfolioDates.set(day, group)
  }
  let assetCount = 0

  return (
    <>
      <p className="muted">指定期間 {requested.start_date} 〜 {requested.end_date}</p>
      <p className="evaluation">
        共通評価期間{' '}
        <strong>
          {evaluation.start_date} 〜 {evaluation.end_date}
        </strong>
        <span>{evaluation.observations}営業日</span>
      </p>
      {(!result.annualized_metrics_available || result.applied_strategy !== requested.strategy || result.unavailable_tickers.length > 0) && (
        <div className="analysis-notice">
          {result.applied_strategy !== requested.strategy && (
            <p>選んだ戦略の学習履歴が足りないため、Equal Weightの結果を参考表示しています。</p>
          )}
          {evaluation.observations < result.minimum_annualized_observations && (
            <p>共通評価が{result.minimum_annualized_observations}営業日未満のため、年率指標とシャープレシオは表示しません。期間リターンと最大ドローダウンはこの期間内の参考値です。</p>
          )}
          {result.skipped_sessions > 0 && (
            <p>一部の株価が欠けていた{result.skipped_sessions}営業日は評価日から除き、年率指標は表示しません。</p>
          )}
          {result.unavailable_tickers.length > 0 && (
            <p>指定期間に組み入れに必要な株価がそろわなかった銘柄：{result.unavailable_tickers.join('、')}</p>
          )}
        </div>
      )}
      {portfolioDates.size > 1 && (
        <div className="allocation-history">
          <h3>銘柄の組み入れ</h3>
          <ul>
            {[...portfolioDates].map(([day, symbols], index) => {
              assetCount += symbols.length
              return <li key={day}>{day} {index > 0 && `${symbols.join('、')}を追加し、`}{assetCount}銘柄・各{percent.format(1 / assetCount)}</li>
            })}
          </ul>
          <p className="muted">新しい銘柄は最初の終値を確認した後、次の評価日に組み入れます。</p>
        </div>
      )}
      <div className="table-scroll" tabIndex={0} aria-label="成績の比較表">
        <table>
          <thead>
            <tr>
              <th scope="col">比較対象</th>
              {metricColumns.map(({ key, label }) => (
                <th scope="col" key={key}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {performance.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                {metricColumns.map(({ key }) => (
                  <td key={key}>{formatMetric(row, key)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>累積リターン</h3>
      <Suspense
        fallback={
          <p className="chart" role="status">
            グラフを読み込んでいます…
          </p>
        }
      >
        <ReturnChart
          points={points}
          strategyLabel={strategyLabel}
          benchmarkLabel={result.benchmark_label}
        />
      </Suspense>
      <p className="muted">
        戦略は売買コスト控除後、比較先は連動ETF（{result.benchmark_ticker}）のリターンです。
        いずれも{result.currency}ベースで計算しています。
      </p>
    </>
  )
}

function App() {
  const [config, setConfig] = useState<AnalysisConfig | null>(null)
  const [form, setForm] = useState<AnalysisForm | null>(null)
  const [configError, setConfigError] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<AnalysisResponse | null>(null)
  const market = config && form ? config.markets[form.market] : null

  useEffect(() => {
    const controller = new AbortController()
    fetchConfig(controller.signal)
      .then((data) => {
        const { tickers, ...defaults } = data.defaults
        setConfig(data)
        setForm({
          ...defaults,
          tickersByMarket: {
            jp: data.markets.jp.default_tickers,
            us: data.markets.us.default_tickers,
            [defaults.market]: tickers,
          },
        })
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setConfigError(errorMessage(reason))
      })
    return () => controller.abort()
  }, [])

  function updateForm(changes: Partial<AnalysisForm>) {
    setForm((current) => current && { ...current, ...changes })
    setResult(null)
    setError('')
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!config || !form || busy) return
    setResult(null)
    setError('')
    if (form.start_date > form.end_date) {
      setError('終了日は開始日以降にしてください。')
      return
    }

    const { tickersByMarket, ...request } = form
    const tickers = tickersByMarket[request.market]
    if (tickers.length === 0 || tickers.length > config.max_tickers) {
      setError(`銘柄を1〜${config.max_tickers}件選択してください。`)
      return
    }

    setBusy(true)
    try {
      setResult(await analyze({ ...request, tickers }))
    } catch (reason) {
      setError(errorMessage(reason))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          QuantAlloc
        </div>
        <span className="version">v0.30</span>
      </header>
      <main>
        <div className="intro">
          <h1>ポートフォリオ分析</h1>
          <p>日本株・米国株で配分戦略を検証し、各市場のベンチマークと比較します。</p>
        </div>
        <div className="workspace">
          <section className="panel settings" aria-labelledby="settings-title">
            <h2 id="settings-title">分析条件</h2>
            {configError && (
              <div className="config-error">
                <p className="error" role="alert">
                  {configError}
                </p>
                <button className="primary" onClick={() => window.location.reload()}>
                  再読み込み
                </button>
              </div>
            )}
            {!config && !configError && <p role="status">設定を読み込んでいます…</p>}
            {config && form && market && (
              <>
                <form onSubmit={submit}>
                  <fieldset disabled={busy}>
                    <label htmlFor="market">市場</label>
                    <select
                      id="market"
                      value={form.market}
                      onChange={(event) => updateForm({ market: event.target.value as MarketId })}
                    >
                      {Object.entries(config.markets).map(([id, item]) => (
                        <option key={id} value={id}>
                          {item.label}
                        </option>
                      ))}
                    </select>
                    <SymbolPicker
                      key={form.market}
                      market={form.market}
                      selected={form.tickersByMarket[form.market]}
                      limit={config.max_tickers}
                      onChange={(symbols) => updateForm({
                        tickersByMarket: { ...form.tickersByMarket, [form.market]: symbols },
                      })}
                    />
                    <label htmlFor="strategy">戦略</label>
                    <select
                      id="strategy"
                      value={form.strategy}
                      onChange={(event) => updateForm({ strategy: event.target.value })}
                    >
                      {Object.entries(config.strategies).map(([id, label]) => (
                        <option key={id} value={id}>
                          {label}
                        </option>
                      ))}
                    </select>
                    <label htmlFor="start">評価開始日</label>
                    <input
                      id="start"
                      type="date"
                      required
                      max={config.maximum_date}
                      value={form.start_date}
                      onInput={(event) => updateForm({ start_date: event.currentTarget.value })}
                    />
                    <label htmlFor="end">評価終了日</label>
                    <input
                      id="end"
                      type="date"
                      required
                      max={config.maximum_date}
                      value={form.end_date}
                      onInput={(event) => updateForm({ end_date: event.currentTarget.value })}
                    />
                    <p className="field-note">日次リターンの評価は両端を含みます。学習が必要な戦略は開始日より前の株価も取得します。</p>
                    <button className="primary" type="submit">
                      {busy ? '分析中…' : '分析を実行'}
                    </button>
                  </fieldset>
                </form>
                <dl className="assumptions">
                  <div>
                    <dt>現地通貨</dt>
                    <dd>{market.currency}</dd>
                  </div>
                  <div>
                    <dt>ベンチマーク</dt>
                    <dd>{market.benchmark_label} ({market.benchmark_ticker})</dd>
                  </div>
                  <div>
                    <dt>学習期間</dt>
                    <dd>{form.strategy === 'equal_weight' ? '不要' : `${config.estimation_window}営業日`}</dd>
                  </div>
                  <div>
                    <dt>リバランス間隔</dt>
                    <dd>{config.holding_period}営業日</dd>
                  </div>
                  <div>
                    <dt>売買コスト</dt>
                    <dd>{config.transaction_cost_bps} bps</dd>
                  </div>
                </dl>
              </>
            )}
          </section>
          <section className="panel results" aria-labelledby="results-title" aria-busy={busy}>
            <h2 id="results-title">{market ? `${market.benchmark_label}との比較` : '分析結果'}</h2>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
            {busy && (
              <p className="empty" role="status">
                株価を取得してバックテストを実行しています…
              </p>
            )}
            {!result && !busy && !error && (
              <p className="empty">分析を実行すると、成績の比較と累積リターンを表示します。</p>
            )}
            {result && <Result result={result} />}
          </section>
        </div>
      </main>
    </>
  )
}

function errorMessage(reason: unknown) {
  return reason instanceof Error ? reason.message : '分析を完了できませんでした。'
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
