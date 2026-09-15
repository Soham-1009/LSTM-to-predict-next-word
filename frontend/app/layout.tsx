import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { TopNav } from "@/components/dashboard/TopNav";
import { ThemeScript } from "@/components/ui/ThemeScript";
import "./globals.css";

export const metadata: Metadata = {
  title: "LSTM Next Word Predictor",
  description:
    "Generative AI application for next-word prediction and text generation using a PyTorch LSTM served by FastAPI.",
  applicationName: "LSTM Next Word Predictor",
  authors: [{ name: "LSTM Next Word Predictor" }],
  keywords: [
    "LSTM",
    "next word prediction",
    "text generation",
    "PyTorch",
    "FastAPI",
    "Generative AI",
  ],
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8fafc" },
    { media: "(prefers-color-scheme: dark)", color: "#090b13" },
  ],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ThemeScript />
      </head>
      <body className="min-h-screen bg-surface-page font-sans text-ink antialiased">
        <TopNav />
        {children}
      </body>
    </html>
  );
}
