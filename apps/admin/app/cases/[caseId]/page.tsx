"use client";

import { useEffect, useMemo, useState } from "react";
import { act, detail, templates, type Template } from "../../../lib/api";
import { MessageThread } from "../../../components/cases/MessageThread";
import { ReplyBox } from "../../../components/cases/ReplyBox";

export default function Case({ params }: { params: Promise<{ caseId: string }> }) {
  const [data, setData] = useState<any>(); const [id, setId] = useState(""); const [catalog, setCatalog] = useState<Template[]>([]); const [error, setError] = useState("");
  useEffect(() => { params.then(({ caseId }) => { setId(caseId); detail(caseId).then(setData).catch(() => setError("This case could not be loaded.")); }); }, [params]);
  useEffect(() => { templates().then(setCatalog).catch(() => undefined); }, []);
  const refresh = () => detail(id).then(setData).catch(() => setError("Unable to refresh the case."));
  const lastCustomerMessage = useMemo(() => data?.messages?.filter((message: any) => message.direction === "customer").sort((a: any, b: any) => +new Date(b.created_at) - +new Date(a.created_at))[0], [data]);
  const freeTextAllowed = !!lastCustomerMessage && Date.now() - +new Date(lastCustomerMessage.created_at) < 24 * 60 * 60 * 1000;
  const run = async (action: string, body = {}) => { setError(""); try { await act(id, action, data.case.version, body); await refresh(); } catch (reason) { setError(reason instanceof Error ? reason.message : "Case action failed."); } };
  if (!data) return <main className="empty-state"><div className="empty-state-inner"><div className="empty-glyph">⌁</div><h1>{error || "Loading case…"}</h1></div></main>;
  const item = data.case; const name = item.handoff_summary?.customer_name ?? "Customer"; const initial = name.slice(0, 2).toUpperCase();
  return <main className="detail"><section className="conversation">
    <header className="case-head"><div className="customer"><span className="avatar">{initial}</span><div><h1>{name}</h1><p>Case #{item.case_id.slice(0, 8)} · {item.status === "claimed" ? "Claimed by staff" : "Awaiting assignment"}</p></div></div><div className="case-actions">{item.status === "open" ? <button className="btn primary" onClick={() => run("claim")}>Claim case</button> : <><button className="btn" onClick={() => run("release")}>Release</button><button className="btn primary" onClick={() => run("resolve")}>✓ Resolve</button></>}</div></header>
    <div className={`window-bar ${freeTextAllowed ? "" : "expired"}`}><span><b>{freeTextAllowed ? "● Session open" : "● Session expired"}</b> · {freeTextAllowed ? "WhatsApp free-text window is active" : "Approved templates are required"}</span><span>{lastCustomerMessage ? "Customer conversation recorded" : "No customer message recorded"}</span></div>
    {error && <div className="alert">{error}</div>}<MessageThread messages={data.messages} />
    {item.status === "claimed" ? <ReplyBox freeTextAllowed={freeTextAllowed} availableTemplates={catalog} onReply={(body) => run("reply", body)} /> : <section className="composer-wrap"><p className="composer-notice">Claim this case before responding or resolving it.</p></section>}
  </section><aside className="profile"><div className="profile-card"><div className="profile-person"><span className="avatar">{initial}</span><div><h2>{name}</h2><p>Privacy-preserving customer record</p></div></div><p className="section-label">CUSTOMER ATTRIBUTES</p><div className="chips"><span className="chip teal">Escalated support</span><span className="chip">WhatsApp</span><span className="chip">Case v{item.version}</span></div></div>
    <div className="profile-card order-card"><p className="section-label">CASE CONTEXT</p><div className="order-line"><span>Status</span><b>{item.status}</b></div><div className="order-line"><span>Handoff reason</span><b>{String(item.handoff_summary?.reason ?? "Specialist review")}</b></div><div className="order-line"><span>Thread reference</span><b className="mono">{item.thread_id.slice(0, 12)}…</b></div><p className="section-label">HANDOFF SUMMARY</p><p>{String(item.handoff_summary?.latest_customer_request ?? "No summary available.")}</p></div>
    <div className="profile-card"><p className="section-label">CASE ACTIVITY</p><div className="timeline"><div className="timeline-item active"><b>Case updated</b><small>{new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(item.updated_at))}</small></div><div className="timeline-item"><b>Created</b><small>{new Intl.DateTimeFormat("en", { dateStyle: "medium" }).format(new Date(item.created_at))}</small></div></div></div>
  </aside></main>;
}
