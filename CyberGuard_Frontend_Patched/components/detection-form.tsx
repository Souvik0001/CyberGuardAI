"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Send } from "lucide-react"

interface DetectionFormProps {
  onAnalyze: (data: any) => void
  isLoading: boolean
}

export function DetectionForm({ onAnalyze, isLoading }: DetectionFormProps) {
  const [formData, setFormData] = useState({
    accountName: "",
    platform: "instagram",
    dataType: "messages",
    description: "",
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onAnalyze(formData)
  }

  return (
    <Card className="border-border sticky top-6">
      <CardHeader>
        <CardTitle>Analyze Threat</CardTitle>
        <CardDescription>Provide details for analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Account Name */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Account/Username</label>
            <input
              type="text"
              placeholder="Enter account name"
              value={formData.accountName}
              onChange={(e) => setFormData({ ...formData, accountName: e.target.value })}
              className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          {/* Platform */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Platform</label>
            <select
              value={formData.platform}
              onChange={(e) => setFormData({ ...formData, platform: e.target.value })}
              className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="instagram">Instagram</option>
              <option value="twitter">Twitter/X</option>
              <option value="facebook">Facebook</option>
              <option value="tiktok">TikTok</option>
              <option value="discord">Discord</option>
              <option value="email">Email</option>
              <option value="other">Other</option>
            </select>
          </div>

          {/* Data Type */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Data Type</label>
            <select
              value={formData.dataType}
              onChange={(e) => setFormData({ ...formData, dataType: e.target.value })}
              className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="messages">Messages</option>
              <option value="comments">Comments</option>
              <option value="activity">Activity Log</option>
              <option value="profile">Profile Changes</option>
              <option value="interactions">Interactions</option>
            </select>
          </div>

          {/* Description */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Description</label>
            <textarea
              placeholder="Describe the suspicious behavior..."
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
              rows={4}
            />
          </div>

          {/* File Upload */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Upload Evidence (Optional)</label>
            <div className="border-2 border-dashed border-border rounded-lg p-4 text-center cursor-pointer hover:border-primary/50 transition-colors">
              <Upload className="h-5 w-5 mx-auto text-muted-foreground mb-2" />
              <p className="text-xs text-muted-foreground">Click to upload or drag and drop</p>
              <p className="text-xs text-muted-foreground">PNG, JPG, PDF up to 10MB</p>
              <input type="file" className="hidden" />
            </div>
          </div>

          {/* Submit Button */}
          <Button type="submit" disabled={isLoading || !formData.accountName} className="w-full gap-2">
            {isLoading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Analyze Now
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
