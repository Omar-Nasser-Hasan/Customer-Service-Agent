"use client";

import { GoogleLogin, GoogleOAuthProvider } from "@react-oauth/google";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { api, session } from "../../lib/api";

export default function Login() {
  const router = useRouter();
  const client = process.env.NEXT_PUBLIC_GOOGLE_OAUTH_CLIENT_ID ?? "";
  const [error, setError] = useState<string | null>(null);

  // A returning staff member should not need to see Google Sign-In again.
  useEffect(() => {
    session().then(() => router.replace("/cases")).catch(() => undefined);
  }, [router]);

  if (!client) {
    return <main className="thread"><h1>Souqly staff inbox</h1><p className="muted">Google sign-in is not configured.</p></main>;
  }

  return <main className="thread">
    <h1>Souqly staff inbox</h1>
    <GoogleOAuthProvider clientId={client}>
      <GoogleLogin
        onSuccess={async ({ credential }) => {
          if (!credential) return setError("Google did not provide an ID token.");
          try {
            await api("/admin/auth/google", { method: "POST", body: JSON.stringify({ credential }) });
            router.replace("/cases");
          } catch {
            setError("This Google account is not authorized for staff access.");
          }
        }}
        onError={() => setError("Google sign-in failed. Please try again.")}
      />
    </GoogleOAuthProvider>
    {error && <p role="alert">{error}</p>}
  </main>;
}
