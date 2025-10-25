"use client"

import React, { useRef, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FileText, Trash2 } from "lucide-react"

interface DatasetItem {
  id: string
  name: string
  size: string
  samples: number
  uploadDate: string
  file?: File | null
}

interface DatasetManagerProps {
  onDatasetSelected?: (file?: File) => void
}

export function DatasetManager({ onDatasetSelected }: DatasetManagerProps) {
  const [datasets, setDatasets] = useState<DatasetItem[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const addFileAsDataset = (file: File) => {
    const newDataset: DatasetItem = {
      id: Date.now().toString(),
      name: file.name,
      size: (file.size / 1024 / 1024).toFixed(2) + " MB",
      samples: Math.max(1000, Math.floor(Math.random() * 5000) + 1000),
      uploadDate: new Date().toISOString().split("T")[0],
      file,
    }
    setDatasets((prev) => [newDataset, ...prev])
    setSelectedId(newDataset.id)
    if (onDatasetSelected) onDatasetSelected(file)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "copy"
    setIsDragging(true)
  }

  const handleDragLeave = () => setIsDragging(false)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files && files.length > 0) addFileAsDataset(files[0])
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) addFileAsDataset(file)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleDelete = (id: string) => {
    setDatasets((prev) => {
      const next = prev.filter((d) => d.id !== id)
      if (selectedId === id) {
        setSelectedId(null)
        if (onDatasetSelected) onDatasetSelected(undefined)
      }
      return next
    })
  }

  const handleSelect = (id: string) => {
    setSelectedId(id)
    const ds = datasets.find((d) => d.id === id)
    if (onDatasetSelected) onDatasetSelected(ds?.file ?? undefined)
  }

  const openFilePicker = () => fileInputRef.current?.click()

  return (
    <Card className="border-border sticky top-6">
      <CardHeader>
        <CardTitle>Datasets</CardTitle>
        <CardDescription>Manage training datasets</CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          role="button"
          onClick={openFilePicker}
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
            isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
          }`}
        >
          <Upload className="h-6 w-6 mx-auto text-muted-foreground mb-2" />
          <p className="text-xs font-medium text-foreground mb-1">Drag and drop dataset or click to upload</p>
          <p className="text-xs text-muted-foreground">CSV, JSON up to 500MB</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,application/json,text/csv"
            className="hidden"
            onChange={handleFileInputChange}
          />
        </div>

        <div className="space-y-2">
          <p className="text-xs font-semibold text-muted-foreground uppercase">Uploaded Datasets</p>

          {datasets.length === 0 && (
            <div className="p-3 rounded-lg bg-muted border border-border text-xs text-muted-foreground">No datasets uploaded yet.</div>
          )}

          {datasets.map((dataset) => {
            const isSelected = selectedId === dataset.id
            return (
              <div
                key={dataset.id}
                className={`p-3 rounded-lg border ${isSelected ? "border-primary bg-primary/5" : "border-border bg-muted"} flex flex-col gap-2`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-2 flex-1 min-w-0">
                    <FileText className="h-4 w-4 text-primary flex-shrink-0 mt-0.5" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">{dataset.name}</p>
                      <p className="text-xs text-muted-foreground">{dataset.size}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button size="sm" variant={isSelected ? "default" : "ghost"} onClick={() => handleSelect(dataset.id)} className="h-6">
                      {isSelected ? "Selected" : "Select"}
                    </Button>

                    <Button size="sm" variant="ghost" onClick={() => handleDelete(dataset.id)} className="h-6 w-6 p-0 flex-shrink-0" aria-label={`Delete ${dataset.name}`}>
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{dataset.samples.toLocaleString()} samples</span>
                  <span>{dataset.uploadDate}</span>
                </div>
              </div>
            )
          })}
        </div>

        <div className="p-3 rounded-lg bg-card border border-border space-y-2">
          <p className="text-xs font-semibold text-foreground">Dataset Requirements:</p>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Minimum 1,000 samples</li>
            <li>• Balanced classes</li>
            <li>• CSV or JSON format</li>
            <li>• Max 500MB file size</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}

export default DatasetManager
