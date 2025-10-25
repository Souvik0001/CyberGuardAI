import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Header } from "@/components/header"
import { Shield, Eye, Brain, BarChart3, ArrowRight, CheckCircle2 } from "lucide-react"
import Link from "next/link"

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <Header />

      {/* Hero Section */}
      <section className="relative overflow-hidden border-b border-border">
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <div className="grid gap-12 lg:grid-cols-2 lg:gap-8 items-center">
            <div className="space-y-6">
              <div className="space-y-2">
                <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-5xl">
                  Advanced Cyberstalking Detection & Screenshot Verification
                </h1>
                <p className="text-xl text-muted-foreground">
                  Protect yourself with AI-powered threat detection and authentic screenshot verification
                </p>
              </div>
              <div className="flex flex-col gap-3 sm:flex-row">
                <Link href="/detection">
                  <Button size="lg" className="gap-2">
                    Get Started <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Button size="lg" variant="outline">
                  Learn More
                </Button>
              </div>
            </div>

            <div className="relative h-96 rounded-lg border border-border bg-card p-8 flex items-center justify-center">
              <div className="text-center space-y-4">
                <Shield className="h-24 w-24 mx-auto text-primary" />
                <p className="text-muted-foreground">Advanced Security Platform</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="border-b border-border py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">Powerful Features</h2>
            <p className="text-lg text-muted-foreground">Everything you need to stay safe online</p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {/* Stalker Detection */}
            <Card className="border-border hover:border-primary/50 transition-colors">
              <CardHeader>
                <Eye className="h-8 w-8 text-primary mb-2" />
                <CardTitle className="text-lg">Stalker Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  AI-powered analysis to detect suspicious behavior patterns and potential threats
                </CardDescription>
              </CardContent>
            </Card>

            {/* Screenshot Analysis */}
            <Card className="border-border hover:border-primary/50 transition-colors">
              <CardHeader>
                <Brain className="h-8 w-8 text-primary mb-2" />
                <CardTitle className="text-lg">Screenshot Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Verify screenshot authenticity and detect manipulated or forged images
                </CardDescription>
              </CardContent>
            </Card>

            {/* Model Training */}
            <Card className="border-border hover:border-primary/50 transition-colors">
              <CardHeader>
                <Brain className="h-8 w-8 text-primary mb-2" />
                <CardTitle className="text-lg">Model Training</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Train custom AI models with your data for personalized threat detection
                </CardDescription>
              </CardContent>
            </Card>

            {/* Analytics & Reports */}
            <Card className="border-border hover:border-primary/50 transition-colors">
              <CardHeader>
                <BarChart3 className="h-8 w-8 text-primary mb-2" />
                <CardTitle className="text-lg">Analytics & Reports</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>Comprehensive reports and analytics to track threats and trends</CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid gap-12 lg:grid-cols-2 items-center">
            <div className="space-y-6">
              <h2 className="text-3xl font-bold text-foreground">Why Choose CyberGuard?</h2>
              <ul className="space-y-4">
                {[
                  "Real-time threat detection and alerts",
                  "Advanced AI-powered analysis",
                  "Screenshot authenticity verification",
                  "Comprehensive analytics dashboard",
                  "Custom model training capabilities",
                  "24/7 security monitoring",
                ].map((benefit, i) => (
                  <li key={i} className="flex gap-3 items-start">
                    <CheckCircle2 className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-foreground">{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="relative h-96 rounded-lg border border-border bg-card p-8 flex items-center justify-center">
              <div className="text-center space-y-4">
                <BarChart3 className="h-24 w-24 mx-auto text-primary" />
                <p className="text-muted-foreground">Analytics Dashboard</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="border-t border-border bg-card py-16">
        <div className="mx-auto max-w-4xl px-4 text-center sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-foreground mb-4">Ready to Protect Yourself?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Start using CyberGuard today and take control of your online safety
          </p>
          <Link href="/detection">
            <Button size="lg" className="gap-2">
              Start Free Trial <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </section>
    </div>
  )
}
