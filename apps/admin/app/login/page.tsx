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

  const googleLogin = client ? <GoogleOAuthProvider clientId={client}><GoogleLogin
    onSuccess={async ({ credential }) => {
      if (!credential) return setError("Google did not provide an ID token.");
      try { await api("/admin/auth/google", { method: "POST", body: JSON.stringify({ credential }) }); router.replace("/cases"); }
      catch { setError("This Google account is not authorized for staff access."); }
    }}
    onError={() => setError("Google sign-in failed. Please try again.")}
  /></GoogleOAuthProvider> : <button className="btn" disabled>Google sign-in is not configured</button>;

  return <main className="login"><section className="login-panel"><p className="login-meta">● OPERATIONS GATEWAY · CAIRO HQ</p><div className="login-card"><div className="brand-mark">↟</div><p className="eyebrow">AGENT CONSOLE</p><h1>Welcome <span className="gold">back</span></h1><p>Sign in with your work Google account to open the secure support queue.</p><div className="login-provider">{googleLogin}</div>{error && <p className="alert" role="alert">{error}</p>}<hr /><p className="muted">Access is limited to authorized Souqly support staff.</p></div></section><section className="login-side"><p className="login-est">EST. 2026 · CAIRO, EGYPT</p><div className="login-wordmark">↟ Souqly</div><p className="login-footer">Souqly Unified Commerce Operations</p></section></main>;
}
