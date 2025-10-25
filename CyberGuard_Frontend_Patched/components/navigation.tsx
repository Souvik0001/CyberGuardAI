"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Shield, BarChart3, Settings, Home } from "lucide-react"

export function Navigation() {
  const pathname = usePathname()

  const isActive = (path: string) => pathname === path

  return (
    <nav className="border-b border-border bg-card">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-2">
            <Shield className="w-6 h-6 text-accent" />
            <span className="text-xl font-bold text-foreground">CyberGuard</span>
          </Link>

          <div className="flex items-center gap-1">
            <Link
              href="/"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                isActive("/") ? "bg-accent text-accent-foreground" : "text-foreground hover:bg-muted"
              }`}
            >
              <Home className="w-4 h-4" />
              Home
            </Link>
            <Link
              href="/stalker-detection"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                isActive("/stalker-detection") ? "bg-accent text-accent-foreground" : "text-foreground hover:bg-muted"
              }`}
            >
              <Shield className="w-4 h-4" />
              Detection
            </Link>
            <Link
              href="/screenshot-analysis"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                isActive("/screenshot-analysis") ? "bg-accent text-accent-foreground" : "text-foreground hover:bg-muted"
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Analysis
            </Link>
            <Link
              href="/model-training"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                isActive("/model-training") ? "bg-accent text-accent-foreground" : "text-foreground hover:bg-muted"
              }`}
            >
              <Settings className="w-4 h-4" />
              Training
            </Link>
            <Link
              href="/reports"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                isActive("/reports") ? "bg-accent text-accent-foreground" : "text-foreground hover:bg-muted"
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Reports
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}
