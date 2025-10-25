"use client"

import { Button } from "@/components/ui/button"
import { Shield } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"

export function Header() {
  const pathname = usePathname()

  const isActive = (path: string) => pathname === path

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 font-bold text-lg">
            <Shield className="h-6 w-6 text-primary" />
            <span className="text-foreground">CyberGuard</span>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <Link
              href="/"
              className={`text-sm font-medium transition-colors ${
                isActive("/") ? "text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Home
            </Link>
            <Link
              href="/detection"
              className={`text-sm font-medium transition-colors ${
                isActive("/detection") ? "text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Detection
            </Link>
            <Link
              href="/analysis"
              className={`text-sm font-medium transition-colors ${
                isActive("/analysis") ? "text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Analysis
            </Link>
            <Link
              href="/reports"
              className={`text-sm font-medium transition-colors ${
                isActive("/reports") ? "text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Reports
            </Link>
          </nav>

          {/* CTA Button */}
          <div className="flex items-center gap-4">
            <Button variant="outline" size="sm">
              Sign In
            </Button>
            <Button size="sm">Get Started</Button>
          </div>
        </div>
      </div>
    </header>
  )
}
