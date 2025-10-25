"use client"

import React, { useState } from "react"
import { AppLayout } from "@/components/app-layout"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { DatasetManager } from "@/components/dataset-manager"
import { ModelConfig } from "@/components/model-config"
import { TrainingProgress } from "@/components/training-progress"
import { ModelPerformance } from "@/components/model-performance"
import { CheckCircle2, Pause, Square } from "lucide-react"
import { trainModels } from "@/lib/api-client"

interface TrainingSession {
  id: string
  name: string
  status: "idle" | "training" | "completed" | "failed"
  progress: number
  accuracy: number
  loss: number
  epoch: number
  totalEpochs: number
  startTime: string
  estimatedTime: string
  metrics: {
    precision: number // fraction 0..1
    recall: number // fraction 0..1
    f1Score: number // fraction 0..1
    auc: number // fraction 0..1
  }
  // optional UI helpers
  elapsedSeconds?: number
  etaSeconds?: number
  history?: { epoch: number; accuracyPct: number; loss: number }[]
}

export default function TrainingPage() {
  const [trainingSession, setTrainingSession] = useState<TrainingSession | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [datasetSelected, setDatasetSelected] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleDatasetSelected = (payload?: any) => {
    if (!payload) {
      setSelectedFile(null)
      setDatasetSelected(false)
      return
    }
    if (payload instanceof File) {
      setSelectedFile(payload)
      setDatasetSelected(true)
      console.log("Dataset file selected:", payload.name)
      return
    }
    if (typeof payload === "boolean") {
      setDatasetSelected(Boolean(payload))
      return
    }
    if (payload?.target?.files?.[0]) {
      setSelectedFile(payload.target.files[0])
      setDatasetSelected(true)
    }
  }

  const toFraction = (v: any) => {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return 0
    const n = Number(v)
    if (n >= 0 && n <= 1) return n
    if (n > 1 && n <= 100) return n / 100
    if (n > 100) return 1
    return 0
  }

  const handleStartTraining = async (config: any) => {
    setIsTraining(true)
    const totalEpochs = config?.epochs ?? 50
    const initial: TrainingSession = {
      id: "train-" + Date.now(),
      name: config?.modelName || "Custom Threat Detector",
      status: "training",
      progress: 0,
      accuracy: 0,
      loss: 1,
      epoch: 0,
      totalEpochs,
      startTime: new Date().toISOString(),
      estimatedTime: "~calculating",
      metrics: { precision: 0, recall: 0, f1Score: 0, auc: 0 },
      history: [],
    }
    setTrainingSession(initial)

    const startTs = Date.now()
    let elapsedTimer: number | null = window.setInterval(() => {
      setTrainingSession((prev) =>
        prev
          ? {
              ...prev,
              elapsedSeconds: (Date.now() - startTs) / 1000,
            }
          : prev,
      )
    }, 500)

    try {
      const res = await trainModels({ file: selectedFile ?? undefined, epochs: totalEpochs })
      if (!res.success) throw new Error(res.error || "Training failed")
      const data = res.data || {}
      console.log("trainModels response:", data)

      const epoch_metrics: Array<{ epoch?: number; accuracy?: number; loss?: number }> = Array.isArray(
        data.epoch_metrics,
      )
        ? data.epoch_metrics
        : []

      const lr = data.linear_regression ?? {}
      const ifBlock = data.isolation_forest ?? {}

      // compute final fractions (0..1) — keep as fractions so ModelPerformance can format
      const precisionFrac = toFraction(lr["precision_binary"] ?? lr["precision"] ?? ifBlock["precision_binary"] ?? 0)
      const recallFrac = toFraction(lr["recall_binary"] ?? lr["recall"] ?? ifBlock["recall_binary"] ?? 0)
      const f1Frac = precisionFrac + recallFrac > 0 ? (2 * precisionFrac * recallFrac) / (precisionFrac + recallFrac) : 0
      const aucFrac = toFraction(lr["auc"] ?? ifBlock["auc"] ?? 0)

      if (epoch_metrics.length > 0) {
        const frameMs = config?.uiFrameMs ?? 120
        const uiDurationMs = Math.max(300, epoch_metrics.length * frameMs)
        const replayStart = Date.now()
        let idx = 0
        setTrainingSession((prev) => (prev ? { ...prev, history: [] } : prev))

        const replayTimer = window.setInterval(() => {
          const elapsedSec = (Date.now() - startTs) / 1000
          const elapsedReplay = Date.now() - replayStart
          const remainingMs = Math.max(0, uiDurationMs - elapsedReplay)
          const etaSec = remainingMs / 1000

          if (idx >= epoch_metrics.length) {
            clearInterval(replayTimer)
            const final = epoch_metrics[epoch_metrics.length - 1] || {}

            setTrainingSession((prev) =>
              prev
                ? {
                    ...prev,
                    status: "completed",
                    progress: 100,
                    accuracy: Math.round(((final.accuracy ?? 0) * 100) * 100) / 100, // stored as percent for chart/box
                    loss: Math.round(((final.loss ?? 0) * 1000)) / 1000,
                    epoch: epoch_metrics.length,
                    elapsedSeconds: elapsedSec,
                    etaSeconds: 0,
                    estimatedTime: "~0s",
                    // store metrics AS FRACTIONS 0..1 — ModelPerformance will render percent
                    metrics: {
                      precision: precisionFrac,
                      recall: recallFrac,
                      f1Score: f1Frac,
                      auc: aucFrac,
                    },
                    history: epoch_metrics.map((e) => ({
                      epoch: e.epoch ?? 0,
                      accuracyPct: Math.round(((e.accuracy ?? 0) * 100) * 100) / 100, // percent 0..100
                      loss: Math.round(((e.loss ?? 0) * 1000)) / 1000,
                    })),
                  }
                : prev,
            )

            if (elapsedTimer) {
              clearInterval(elapsedTimer)
              elapsedTimer = null
            }
            setIsTraining(false)
            return
          }

          const e = epoch_metrics[idx]
          const epochIndex = idx + 1
          const progressPct = Math.round((epochIndex / epoch_metrics.length) * 10000) / 100
          const accuracyPct = Math.round(((e.accuracy ?? 0) * 100) * 100) / 100
          const lossRounded = Math.round(((e.loss ?? 0) * 1000)) / 1000

          setTrainingSession((prev) =>
            prev
              ? {
                  ...prev,
                  progress: progressPct,
                  accuracy: accuracyPct,
                  loss: lossRounded,
                  epoch: epochIndex,
                  elapsedSeconds: elapsedSec,
                  etaSeconds: etaSec,
                  estimatedTime: etaSec <= 0.5 ? "~0s" : etaSec >= 60 ? `~${Math.floor(etaSec / 60)}m ${Math.floor(etaSec % 60)}s` : `~${Math.floor(etaSec)}s`,
                  history: [...(prev.history ?? []), { epoch: epochIndex, accuracyPct, loss: lossRounded }],
                }
              : prev,
          )

          idx += 1
        }, frameMs)
      } else {
        // no epoch metrics - short animation + set final metrics (fractions)
        const finalAccFrac = toFraction(lr["accuracy_binary(>=0.5)"] ?? lr["accuracy_binary"] ?? lr["accuracy"] ?? 0)
        const finalAccPct = finalAccFrac * 100
        const finalLoss = Number(lr["mse"] ?? data.mse ?? 0)
        const animMs = 800
        const startTime = Date.now()
        const from = { progress: 0, accuracy: 0, loss: 1 }

        // Build tiny synthetic history so graph isn't empty
        const syntheticCount = Math.min(6, Math.max(3, Math.floor(totalEpochs / 5)))
        const synthetic = Array.from({ length: syntheticCount }, (_, i) => {
          const t = (i + 1) / syntheticCount
          // finalAccPct is percent already; scale by t and keep 2 decimals
          return { epoch: Math.round(t * totalEpochs), accuracyPct: Math.round(t * finalAccPct * 100) / 100, loss: Math.round((1 - t) * finalLoss * 1000) / 1000 }
        })

        setTrainingSession((prev) => (prev ? { ...prev, history: synthetic } : prev))

        const tick = () => {
          const t = Math.min(1, (Date.now() - startTime) / animMs)
          const interp = (a: number, b: number) => a + (b - a) * t
          const elapsedSec = (Date.now() - startTs) / 1000
          setTrainingSession((prev) =>
            prev
              ? {
                  ...prev,
                  progress: Math.round(interp(from.progress, 100) * 100) / 100,
                  accuracy: Math.round(interp(from.accuracy, finalAccPct) * 100) / 100,
                  loss: Math.round(interp(from.loss, finalLoss) * 1000) / 1000,
                  epoch: totalEpochs,
                  elapsedSeconds: elapsedSec,
                  etaSeconds: 0,
                  estimatedTime: "~0s",
                  // store metrics AS FRACTIONS 0..1
                  metrics: { precision: precisionFrac, recall: recallFrac, f1Score: f1Frac, auc: aucFrac },
                }
              : prev,
          )
          if (t < 1) requestAnimationFrame(tick)
          else {
            setIsTraining(false)
            if (elapsedTimer) {
              clearInterval(elapsedTimer)
              elapsedTimer = null
            }
          }
        }
        requestAnimationFrame(tick)
      }
    } catch (err: any) {
      console.error("Training failed:", err)
      setTrainingSession((prev) => (prev ? { ...prev, status: "failed" } : prev))
      setIsTraining(false)
      if (elapsedTimer) {
        clearInterval(elapsedTimer)
        elapsedTimer = null
      }
    }
  }

  const handlePauseTraining = () => {
    setTrainingSession((prev) => (prev ? { ...prev, status: "idle" } : null))
    setIsTraining(false)
  }

  const handleStopTraining = () => {
    setTrainingSession(null)
    setIsTraining(false)
  }

  return (
    <AppLayout>
      <div className="space-y-6 p-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Model Training</h1>
          <p className="text-muted-foreground mt-2">Train custom AI models with your data for personalized threat detection</p>
        </div>

        {trainingSession ? (
          <div className="space-y-6">
            <Card className="border-border">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>{trainingSession.name}</CardTitle>
                    <CardDescription>
                      {trainingSession.status === "training" && "Training in progress..."}
                      {trainingSession.status === "completed" && "Training completed successfully"}
                      {trainingSession.status === "failed" && "Training failed"}
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    {trainingSession.status === "training" && (
                      <>
                        <Button size="sm" variant="outline" onClick={handlePauseTraining}>
                          <Pause className="h-4 w-4" />
                        </Button>
                        <Button size="sm" variant="outline" onClick={handleStopTraining}>
                          <Square className="h-4 w-4" />
                        </Button>
                      </>
                    )}
                    {trainingSession.status === "completed" && <CheckCircle2 className="h-6 w-6 text-green-500" />}
                  </div>
                </div>
              </CardHeader>
            </Card>

            <TrainingProgress session={trainingSession} />
            <ModelPerformance metrics={trainingSession.metrics} accuracy={trainingSession.accuracy} />
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-1">
              <DatasetManager onDatasetSelected={handleDatasetSelected} />
            </div>

            <div className="lg:col-span-2">
              <ModelConfig onStartTraining={handleStartTraining} isDisabled={!datasetSelected} isLoading={isTraining} />
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  )
}
