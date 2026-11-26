import {
  CategoryScale,
  Chart,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
} from 'chart.js'
import { useEffect, useRef } from 'react'
import type { AnalysisResponse } from './api'

Chart.register(
  CategoryScale,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
)

type ReturnChartProps = {
  points: AnalysisResponse['cumulative_returns']
  strategyLabel: string
}

export default function ReturnChart({ points, strategyLabel }: ReturnChartProps) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvas.current) return

    const chart = new Chart(canvas.current, {
      type: 'line',
      data: {
        labels: points.map((point) => point.date),
        datasets: [
          {
            label: strategyLabel,
            data: points.map((point) => point.strategy * 100),
            borderColor: '#46ddba',
            backgroundColor: '#46ddba',
          },
          {
            label: 'TOPIX',
            data: points.map((point) => point.benchmark * 100),
            borderColor: '#a9afff',
            backgroundColor: '#a9afff',
            borderDash: [6, 4],
          },
        ],
      },
      options: {
        animation: false,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        elements: { line: { borderWidth: 2 }, point: { radius: 0, hoverRadius: 3 } },
        plugins: {
          legend: { labels: { color: '#c5d3d8', boxWidth: 22 }, align: 'start' },
          tooltip: {
            callbacks: {
              label: (item) => `${item.dataset.label}: ${item.formattedValue}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#98aab3', maxTicksLimit: 6, maxRotation: 0 },
          },
          y: {
            grid: { color: '#26383f' },
            ticks: { color: '#98aab3', callback: (value) => `${value}%` },
          },
        },
      },
    })

    return () => chart.destroy()
  }, [points, strategyLabel])

  return (
    <div className="chart">
      <canvas ref={canvas} role="img" aria-label={`${strategyLabel} とTOPIXの累積リターン`} />
    </div>
  )
}
