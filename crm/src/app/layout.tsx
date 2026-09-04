import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "3DIL CREATION CRM",
  description: "Leads, orders, payments and production for 3DIL CREATION",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
