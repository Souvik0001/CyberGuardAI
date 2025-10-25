import type React from "react"
import { Sidebar } from "@/components/sidebar"

export function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex-1 md:ml-64">{children}</main>
    </div>
  )
}
