import { lazy, StrictMode, Suspense, useEffect, useState, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import {
  analyze,
  fetchConfig,
  type AnalysisConfig,
  type AnalysisRequest,
  type AnalysisResponse,
  type PerformanceRow,
} from './api'
import './styles.css'

const ReturnChart = lazy(() => import('./ReturnChart'))
const percent = new Intl.NumberFormat('ja-JP', { style: 'percent', minimumFractionDigits: 2 })
const decimal = new Intl.NumberFormat('ja-JP', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})
const metricColumns = [
  { key: 'cagr', label: '年率リターン' },
  { key: 'volatility', label: '年率ボラティリティ' },
  { key: 'sharpe', label: 'シャープレシオ' },
  { key: 'max_drawdown', label: '最大ドローダウン' },
] as const

function formatMetric(row: PerformanceRow, key: (typeof metricColumns)[number]['key']) {
  const value = row[key]
  if (value === null) return '算出不可'
  return (key === 'sharpe' ? decimal : percent).format(value)
}

function Result({ result }: { result: AnalysisResponse }) {
  const { evaluation, performance, cumulative_returns: points } = result
  const strategyLabel = performance[0]?.label ?? result.requested.strategy

  return (
    <>
      <p className="evaluation">
        共通評価期間{' '}
        <strong>
          {evaluation.start_date} 〜 {evaluation.end_date}
        </strong>
        <span>{evaluation.observations}営業日</span>
      </p>
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
        <ReturnChart points={points} strategyLabel={strategyLabel} />
      </Suspense>
      <p className="muted">
        戦略の成績は売買コスト控除後です。TOPIXは連動ETFの取得リターンを使用しています。
      </p>
    </>
  )
}

function App() {
  const [config, setConfig] = useState<AnalysisConfig | null>(null)
  const [form, setForm] = useState<AnalysisRequest | null>(null)
  const [configError, setConfigError] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<AnalysisResponse | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    fetchConfig(controller.signal)
      .then((data) => {
        setConfig(data)
        setForm(data.defaults)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setConfigError(errorMessage(reason))
      })
    return () => controller.abort()
  }, [])

  function updateField(key: keyof AnalysisRequest, value: string) {
    setForm((current) => current && { ...current, [key]: value })
    setResult(null)
    setError('')
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!form || busy) return
    setResult(null)
    setError('')
    if (form.start_date > form.end_date) {
      setError('終了日は開始日以降にしてください。')
      return
    }

    setBusy(true)
    try {
      setResult(await analyze(form))
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
        <span className="version">v0.27</span>
      </header>
      <main>
        <div className="intro">
          <h1>ポートフォリオ分析</h1>
          <p>日本株の配分戦略を、同じ評価期間のTOPIXと比較します。</p>
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
            {config && form && (
              <>
                <form onSubmit={submit}>
                  <fieldset disabled={busy}>
                    <label htmlFor="strategy">戦略</label>
                    <select
                      id="strategy"
                      value={form.strategy}
                      onChange={(event) => updateField('strategy', event.target.value)}
                    >
                      {Object.entries(config.strategies).map(([id, label]) => (
                        <option key={id} value={id}>
                          {label}
                        </option>
                      ))}
                    </select>
                    <label htmlFor="start">取得開始日</label>
                    <input
                      id="start"
                      type="date"
                      required
                      max={config.maximum_date}
                      value={form.start_date}
                      onChange={(event) => updateField('start_date', event.target.value)}
                    />
                    <label htmlFor="end">取得終了日</label>
                    <input
                      id="end"
                      type="date"
                      required
                      max={config.maximum_date}
                      value={form.end_date}
                      onChange={(event) => updateField('end_date', event.target.value)}
                    />
                    <p className="field-note">開始日・終了日ともに取得期間に含みます。</p>
                    <button className="primary" type="submit">
                      {busy ? '分析中…' : '分析を実行'}
                    </button>
                  </fieldset>
                </form>
                <dl className="assumptions">
                  <div>
                    <dt>対象</dt>
                    <dd>日本株 {config.asset_count}銘柄</dd>
                  </div>
                  <div>
                    <dt>ベンチマーク</dt>
                    <dd>{config.benchmark_ticker}</dd>
                  </div>
                  <div>
                    <dt>学習期間</dt>
                    <dd>{config.estimation_window}営業日</dd>
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
            <h2 id="results-title">TOPIXとの比較</h2>
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
