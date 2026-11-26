import { useEffect, useRef, useState } from 'react'
import { searchSymbols, type MarketId, type SymbolMatch } from './api'

type SymbolPickerProps = {
  market: MarketId
  selected: string[]
  limit: number
  onChange: (symbols: string[]) => void
}

type SearchResult =
  | { matches: SymbolMatch[]; error?: never }
  | { error: string; matches?: never }

export default function SymbolPicker({ market, selected, limit, onChange }: SymbolPickerProps) {
  const [input, setInput] = useState('')
  const [result, setResult] = useState<SearchResult | null>(null)
  const searchInput = useRef<HTMLInputElement>(null)
  const query = input.trim()
  const canSearch = query.length >= 2
  const atLimit = selected.length >= limit

  useEffect(() => {
    if (!canSearch || atLimit) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      searchSymbols(market, query, controller.signal)
        .then((matches) => {
          if (!controller.signal.aborted) setResult({ matches })
        })
        .catch((reason: unknown) => {
          if (controller.signal.aborted) return
          setResult({ error: reason instanceof Error ? reason.message : '銘柄検索に失敗しました。' })
        })
    }, 300)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [market, query, canSearch, atLimit])

  function add(symbol: string) {
    if (atLimit || selected.includes(symbol)) return
    onChange([...selected, symbol])
    setInput('')
    setResult(null)
    searchInput.current?.focus()
  }

  return (
    <div className="symbol-picker">
      <label htmlFor="symbol-query">銘柄を検索</label>
      <input
        ref={searchInput}
        type="search"
        id="symbol-query"
        value={input}
        maxLength={80}
        autoComplete="off"
        disabled={atLimit}
        aria-describedby="symbol-note"
        aria-controls="symbol-matches"
        placeholder={market === 'jp' ? '例 トヨタ、7203' : '例 Apple、AAPL'}
        onChange={(event) => {
          if (event.target.value.trim() !== query) setResult(null)
          setInput(event.target.value)
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.nativeEvent.isComposing) event.preventDefault()
        }}
      />
      <p className="field-note" id="symbol-note">
        {atLimit ? `上限の${limit}銘柄を選択中です。` : '社名または銘柄コードを2文字以上入力してください。'}
      </p>
      <div id="symbol-matches" aria-busy={canSearch && !atLimit && !result}>
        {canSearch && !atLimit && (
          result?.matches?.length ? (
            <ul className="symbol-matches" aria-label="検索結果">
              {result.matches.map(({ symbol, name }) => (
                <li key={symbol}>
                  <button
                    type="button"
                    className="symbol-option"
                    disabled={selected.includes(symbol)}
                    onClick={() => add(symbol)}
                  >
                    <strong>{symbol}</strong>
                    <span>{name}</span>
                    {selected.includes(symbol) && <small>追加済み</small>}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="field-note symbol-status" role={result?.error ? 'alert' : 'status'}>
              {result?.error ?? (result ? '一致する銘柄がありません。' : '検索中…')}
            </p>
          )
        )}
      </div>
      <div className="symbol-selection">
        <div className="symbol-selection-header">
          <span>選択中 <strong>{selected.length}/{limit}</strong></span>
          {selected.length > 0 && (
            <button type="button" className="text-button" onClick={() => onChange([])}>
              すべて解除
            </button>
          )}
        </div>
        {selected.length > 0 && (
          <ul className="symbol-chips" aria-label="選択中の銘柄">
            {selected.map((symbol) => (
              <li key={symbol}>
                <button
                  type="button"
                  aria-label={`${symbol} を削除`}
                  onClick={() => onChange(selected.filter((item) => item !== symbol))}
                >
                  {symbol} <span aria-hidden="true">×</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
