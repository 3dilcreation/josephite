import Image from "next/image";
import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/auth";
import { LoginForm } from "./login-form";

export default async function LoginPage() {
  if (await getCurrentUser()) redirect("/dashboard");

  return (
    <main className="flex min-h-screen items-center justify-center bg-ink-950 px-4 py-10">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center">
          <div className="rounded-2xl bg-white p-4">
            <Image src="/logo.png" alt="3DIL CREATION" width={96} height={78} priority />
          </div>
          <h1 className="mt-4 text-lg font-semibold text-white">3DIL CREATION CRM</h1>
          <p className="text-sm text-ink-500">Sign in to continue</p>
        </div>
        <div className="card p-5">
          <LoginForm />
        </div>
        <p className="mt-4 text-center text-xs text-ink-500">
          Accounts are created by an administrator.
        </p>
      </div>
    </main>
  );
}
