"use client"

import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

type HistPoint = { epoch: number; accuracyPct: number; loss: number }
type Session = {
  progress?: number
  accuracy?: number
  loss?: number
  epoch?: number
  totalEpochs?: number
  estimatedTime?: string
  elapsedSeconds?: number
  etaSeconds?: number
  history?: HistPoint[]
}

const fmtSec = (s?: number) => {
  if (!s || Number.isNaN(Number(s))) return "~0s"
  const sec = Math.max(0, Math.floor(Number(s)))
  const m = Math.floor(sec / 60)
  const r = sec % 60
  return m > 0 ? `~${m}m ${r}s` : `~${r}s`
}

/** Lightweight responsive line chart that uses numeric height */
function SimpleLineChart({ data, height = 220 }: { data: HistPoint[]; height?: number }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-[220px] flex items-center justify-center text-sm text-muted-foreground">
        No chart data
      </div>
    )
  }

  const padding = 24
  const width = 900
  const innerW = width - padding * 2
  const innerH = height - padding * 2

  const xs = data.map((_, i) => padding + (i / (data.length - 1 || 1)) * innerW)
  const maxY = 100
  const ys = data.map((d) => padding + (1 - Math.min(maxY, Math.max(0, d.accuracyPct)) / maxY) * innerH)

  const linePath = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${ys[i].toFixed(2)}`).join(" ")

  return (
    <div className="w-full overflow-hidden">
      {/* NOTE: height is numeric (px) to avoid the 'Expected length' error */}
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} preserveAspectRatio="xMidYMid meet">
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
          const y = padding + t * innerH
          return <line key={i} x1={padding} x2={padding + innerW} y1={y} y2={y} stroke="#1f2937" strokeDasharray="3 6" strokeWidth={1} />
        })}

        {/* x ticks */}
        {xs.map((x, i) => (
          <line key={`tick-${i}`} x1={x} x2={x} y1={padding + innerH} y2={padding + innerH + 6} stroke="#333" strokeWidth={1} />
        ))}

        {/* accuracy line */}
        <path d={linePath} fill="none" stroke="#ff6b6b" strokeWidth={2.4} strokeLinecap="round" strokeLinejoin="round" />

        {/* points */}
        {xs.map((x, i) => (
          <circle key={`c-${i}`} cx={x} cy={ys[i]} r={2.6} fill="#ff6b6b" />
        ))}

        {/* left axis labels */}
        {[0, 50, 100].map((val, i) => {
          const y = padding + (1 - val / 100) * innerH
          return (
            <text key={`yl-${i}`} x={6} y={y + 4} fill="#9ca3af" fontSize={11}>
              {val}
            </text>
          )
        })}
      </svg>
    </div>
  )
}

export function TrainingProgress({ session }: { session: Session }) {
  const history = session.history ?? []

  return (
    <Card className="border-border">
      <CardHeader>
        <CardTitle>Training Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">Epoch {session.epoch ?? 0} of {session.totalEpochs ?? 0}</p>

        <div className="mt-3 w-full h-2 rounded bg-muted overflow-hidden">
          <div className="h-2 bg-primary transition-all" style={{ width: `${Math.min(100, Math.max(0, session.progress ?? 0))}%` }} />
        </div>
        <div className="mt-2 text-right text-sm text-muted-foreground">{Math.round(session.progress ?? 0)}%</div>

        <div className="mt-4 grid grid-cols-1 sm:grid-cols-4 gap-3">
          <div className="p-3 rounded-lg bg-card">
            <div className="text-xs text-muted-foreground">Accuracy</div>
            <div className="text-xl font-semibold">{(session.accuracy ?? 0).toFixed(1)}%</div>
          </div>
          <div className="p-3 rounded-lg bg-card">
            <div className="text-xs text-muted-foreground">Loss</div>
            <div className="text-xl font-semibold">{(session.loss ?? 0).toFixed(3)}</div>
          </div>
          <div className="p-3 rounded-lg bg-card">
            <div className="text-xs text-muted-foreground">Elapsed Time</div>
            <div className="text-xl font-semibold">{fmtSec(session.elapsedSeconds)}</div>
          </div>
          <div className="p-3 rounded-lg bg-card">
            <div className="text-xs text-muted-foreground">Est. Remaining</div>
            <div className="text-xl font-semibold">{fmtSec(session.etaSeconds)}</div>
          </div>
        </div>

        <div className="mt-6">
          <div className="mb-2 text-sm text-muted-foreground">Training Metrics</div>
          <div className="rounded-lg bg-card p-4">
            <SimpleLineChart data={history} />
            <div className="mt-2 text-xs text-muted-foreground">Accuracy over epochs (percent)</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
