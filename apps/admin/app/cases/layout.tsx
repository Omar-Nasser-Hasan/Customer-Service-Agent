"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { cases, logout, session } from "../../lib/api";
import { subscribe } from "../../lib/websocket";
import { CaseListItem } from "../../components/cases/CaseListItem";

export default function Layout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [items, setItems] = useState<any[]>([]);
  const [user, setUser] = useState<{ email: string } | null>(null);
  const [filter, setFilter] = useState("all");
  const [online, setOnline] = useState(true);
  const [toast, setToast] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let close: (() => void) | undefined;
    const load = () => cases().then((value) => { setItems(value); setOnline(true); }).catch(() => setOnline(false));

    session().then(async (staff) => {
      setUser(staff); setReady(true); await load(); close = await subscribe((event) => { if (event.event === "message") setToast(true); load(); });
    }).catch(() => router.replace("/login"));
    return () => close?.();
  }, [router]);

  if (!ready) return <main className="empty-state"><div className="empty-state-inner"><div className="empty-glyph">⌁</div><h1>Opening the support hub</h1><p>Checking your secure staff session…</p></div></main>;

  const visible = items.filter((item) => {
    if (filter === "mine") return item.assigned_to_email === user?.email;
    if (filter === "open") return item.status === "open";
    if (filter === "resolved") return item.status === "resolved";
    return item.status !== "resolved";
  });
  const count = (name: string) => name === "all" ? items.filter((item) => item.status !== "resolved").length : name === "mine" ? items.filter((item) => item.assigned_to_email === user?.email).length : items.filter((item) => item.status === name).length;
  const signOut = async () => { setLoggingOut(true); try { await logout(); router.replace("/login"); } finally { setLoggingOut(false); } };

  return <div className="app-shell">
    <aside className="rail" aria-label="Primary navigation">
      <Link className="brand-mark" href="/cases" aria-label="Souqly home">↟</Link>
      <nav>
        <Link className={`rail-icon ${pathname.startsWith("/cases") ? "active" : ""}`} href="/cases" title="Inbox">▣</Link>
        <Link className={`rail-icon ${pathname === "/cases/templates" ? "active" : ""}`} href="/cases/templates" title="Approved templates">▤</Link>
        <span className="rail-icon" title="Reports">▥</span><span className="rail-icon" title="Settings">⚙</span>
      </nav>
      <div className="user-menu"><div className={`profile-menu ${profileOpen ? "open" : ""}`} role="menu"><div className="profile-menu-label">Profile</div><div className="profile-menu-email">{user?.email}</div><button className="logout-button" onClick={signOut} disabled={loggingOut} role="menuitem">{loggingOut ? "Logging out…" : "Log out"}</button></div><button className="agent-dot" onClick={() => setProfileOpen((value) => !value)} title="Open profile menu" aria-label="Open profile menu" aria-expanded={profileOpen}>{user?.email.slice(0, 2).toUpperCase()}</button><div className="presence" /></div>
    </aside>
    <aside className="case-pane">
      <header className="case-pane-header"><div className="pane-title"><h1>Inbox</h1><span className="counter">{count("mine")} mine</span></div>
        <div className="tabs" role="tablist" aria-label="Case filters">{[["all", "Unassigned"], ["mine", "Mine"], ["open", "Open"], ["resolved", "Resolved"]].map(([key, label]) => <button key={key} className={filter === key ? "active" : ""} onClick={() => setFilter(key)} role="tab" aria-selected={filter === key}>{label} ({count(key)})</button>)}</div>
      </header>
      <div className="case-scroll">{visible.map(item => <CaseListItem key={item.case_id} item={item} selected={pathname.endsWith(`/${item.case_id}`)} />)}</div>
      <footer className="case-pane-foot"><span className="live-dot">●</span><span>{online ? "WebSocket live" : "Reconnecting…"}</span><span>{visible.length} cases</span></footer>
    </aside>
    <section className="workspace"><header className="workspace-topbar"><div className="workspace-title">SOUQLY <small>/ Cairo Support Hub</small></div><div className="top-actions"><span className={`connection ${online ? "" : "offline"}`}>● {online ? "Nile Sync Active" : "Disconnected"}</span><input className="search" aria-label="Search customer, phone, or order" placeholder="Search customer, phone or #SQ-…" /></div></header>{!online && <div className="offline-banner">● Connection lost — reconnecting. Live updates are paused; outgoing messages will queue locally.</div>}{children}{toast && <button className="toast" onClick={() => setToast(false)}><b>● New message received</b><p>A customer update arrived. Select the case from the live inbox.</p></button>}</section>
  </div>;
}
