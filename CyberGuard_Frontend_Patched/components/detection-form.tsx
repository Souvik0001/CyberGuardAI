"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Send } from "lucide-react"
import EvidenceUploader from "./EvidenceUploader";

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

          {/* File Upload (replaced with EvidenceUploader) */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Upload Evidence (Optional)</label>

            {/* Use the shared EvidenceUploader component (client-only UI that posts to backend) */}
            <EvidenceUploader
              apiBase={process.env.NEXT_PUBLIC_API_URL}
              onComplete={(res) => {
                // res is the backend JSON returned by /analyze_screenshot
                // Save the result into your form state (adjust field name as needed).
                // If your component uses setFormData, update the form data here.
                // Example: attach the response to formData.evidence
                if (typeof setFormData === "function") {
                  setFormData((prev:any) => ({ ...(prev || {}), evidence: res }));
                } else {
                  // fallback: print to console and allow you to inspect
                  console.log("Evidence upload result:", res);
                }
              }}
            />
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
