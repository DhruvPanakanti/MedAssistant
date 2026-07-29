import {
  SignInButton,
  SignUpButton,
  Show,
  SignOutButton,
  UserButton,
} from "@clerk/nextjs";

export default function Home() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.18),_transparent_55%)] bg-zinc-50 font-sans text-zinc-900 dark:bg-black dark:text-zinc-100">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6 sm:px-8 lg:px-10">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.25em] text-sky-600">
            MedAssist
          </p>
          <h1 className="text-xl font-semibold">Your personal medical assistant</h1>
        </div>
        <div className="flex items-center gap-3">
          <Show when="signed-out">
            <SignInButton />
            <SignUpButton />
          </Show>
          <Show when="signed-in">
            <div className="flex items-center gap-3">
              <UserButton />
              <SignOutButton />
            </div>
          </Show>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col justify-center px-6 py-16 sm:px-8 lg:px-10 lg:py-24">
        <div className="grid gap-10 rounded-3xl border border-zinc-200 bg-white/80 p-8 shadow-xl shadow-zinc-200/60 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80 dark:shadow-black/30 lg:grid-cols-[1.2fr_0.8fr] lg:p-12">
          <div className="space-y-6">
            <span className="inline-flex rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-sm font-medium text-sky-700 dark:border-sky-900 dark:bg-sky-950/60 dark:text-sky-300">
              Secure sign-in for your health dashboard
            </span>
            <div className="space-y-4">
              <h2 className="text-4xl font-semibold leading-tight sm:text-5xl">
                Sign up or sign in to access your MedAssist experience.
              </h2>
              <p className="max-w-2xl text-lg leading-8 text-zinc-600 dark:text-zinc-400">
                Clerk authentication is now connected to your Next.js app, making it easy to protect your medical assistant workflows and keep the experience polished for every user.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Show when="signed-out">
                <SignUpButton />
                <SignInButton />
              </Show>
              <Show when="signed-in">
                <div className="rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/60 dark:text-emerald-300">
                  You are signed in.
                </div>
              </Show>
            </div>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-6 dark:border-zinc-800 dark:bg-zinc-900/70">
            <h3 className="text-lg font-semibold">What is ready now?</h3>
            <ul className="mt-4 space-y-3 text-sm leading-7 text-zinc-600 dark:text-zinc-400">
              <li>• Clerk provider is wired into the app shell</li>
              <li>• Sign-in and sign-up routes are available</li>
              <li>• Visible auth controls are present in the landing page</li>
              <li>• Clerk proxy matching is configured for the app router</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
