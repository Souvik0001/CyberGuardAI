"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, FileImage } from "lucide-react"

interface ScreenshotUploadProps {
  onUpload: (file: File) => void
  isLoading: boolean
}

export function ScreenshotUpload({ onUpload, isLoading }: ScreenshotUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith("image/")) {
        setFileName(file.name)
        onUpload(file)
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      const file = files[0]
      setFileName(file.name)
      onUpload(file)
    }
  }

  return (
    <Card className="border-border sticky top-6">
      <CardHeader>
        <CardTitle>Upload Screenshot</CardTitle>
        <CardDescription>Verify authenticity and detect manipulations</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Upload Area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
            isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
          }`}
        >
          <input
            type="file"
            id="screenshot-upload"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isLoading}
          />
          <label htmlFor="screenshot-upload" className="cursor-pointer block">
            <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            <p className="text-sm font-medium text-foreground mb-1">Click to upload or drag and drop</p>
            <p className="text-xs text-muted-foreground">PNG, JPG, WebP up to 50MB</p>
          </label>
        </div>

        {/* File Info */}
        {fileName && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-muted">
            <FileImage className="h-4 w-4 text-primary flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">{fileName}</p>
            </div>
          </div>
        )}

        {/* Info Box */}
        <div className="p-3 rounded-lg bg-card border border-border space-y-2">
          <p className="text-xs font-semibold text-foreground">Analysis includes:</p>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Pixel-level analysis</li>
            <li>• Compression artifact detection</li>
            <li>• Font consistency check</li>
            <li>• Lighting analysis</li>
            <li>• Metadata verification</li>
          </ul>
        </div>

        {/* Status */}
        {isLoading && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-primary/10 border border-primary/20">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <p className="text-sm text-foreground">Analyzing screenshot...</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
