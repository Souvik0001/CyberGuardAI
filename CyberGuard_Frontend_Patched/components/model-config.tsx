"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Play } from "lucide-react"

interface ModelConfigProps {
  onStartTraining: (config: any) => void
  isDisabled: boolean
  isLoading: boolean
}

export function ModelConfig({ onStartTraining, isDisabled, isLoading }: ModelConfigProps) {
  const [config, setConfig] = useState({
    modelName: "Custom Threat Detector",
    modelType: "neural-network",
    epochs: 50,
    batchSize: 32,
    learningRate: 0.001,
    validationSplit: 0.2,
    optimizer: "adam",
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onStartTraining(config)
  }

  return (
    <Card className="border-border">
      <CardHeader>
        <CardTitle>Model Configuration</CardTitle>
        <CardDescription>Configure training parameters</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Model Name */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Model Name</label>
            <input
              type="text"
              value={config.modelName}
              onChange={(e) => setConfig({ ...config, modelName: e.target.value })}
              className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          {/* Model Type */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Model Type</label>
            <select
              value={config.modelType}
              onChange={(e) => setConfig({ ...config, modelType: e.target.value })}
              className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="neural-network">Neural Network</option>
              <option value="random-forest">Random Forest</option>
              <option value="gradient-boost">Gradient Boosting</option>
              <option value="svm">Support Vector Machine</option>
            </select>
          </div>

          {/* Training Parameters Grid */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* Epochs */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Epochs</label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="10"
                  max="200"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: Number.parseInt(e.target.value) })}
                  className="flex-1"
                />
                <span className="text-sm font-semibold text-foreground w-12 text-right">{config.epochs}</span>
              </div>
            </div>

            {/* Batch Size */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Batch Size</label>
              <select
                value={config.batchSize}
                onChange={(e) => setConfig({ ...config, batchSize: Number.parseInt(e.target.value) })}
                className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="16">16</option>
                <option value="32">32</option>
                <option value="64">64</option>
                <option value="128">128</option>
              </select>
            </div>

            {/* Learning Rate */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Learning Rate</label>
              <select
                value={config.learningRate}
                onChange={(e) => setConfig({ ...config, learningRate: Number.parseFloat(e.target.value) })}
                className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="0.0001">0.0001</option>
                <option value="0.0005">0.0005</option>
                <option value="0.001">0.001</option>
                <option value="0.01">0.01</option>
              </select>
            </div>

            {/* Validation Split */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Validation Split</label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  value={config.validationSplit}
                  onChange={(e) => setConfig({ ...config, validationSplit: Number.parseFloat(e.target.value) })}
                  className="flex-1"
                />
                <span className="text-sm font-semibold text-foreground w-12 text-right">
                  {(config.validationSplit * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Optimizer */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Optimizer</label>
            <select
              value={config.optimizer}
              onChange={(e) => setConfig({ ...config, optimizer: e.target.value })}
              className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="rmsprop">RMSprop</option>
              <option value="adagrad">Adagrad</option>
            </select>
          </div>

          {/* Start Button */}
          <Button type="submit" disabled={isDisabled || isLoading} className="w-full gap-2">
            {isLoading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                Starting Training...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Start Training
              </>
            )}
          </Button>

          {isDisabled && <p className="text-xs text-muted-foreground text-center">Please upload a dataset first</p>}
        </form>
      </CardContent>
    </Card>
  )
}
