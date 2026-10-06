"use client";

import { useEffect, useState } from "react";
import type { Template } from "../../lib/api";

type Reply = { text?: string; template_id?: string; parameters?: string[] };

export function ReplyBox({ onReply, availableTemplates, freeTextAllowed }: { onReply: (reply: Reply) => Promise<void>; availableTemplates: Template[]; freeTextAllowed: boolean }) {
  const [mode, setMode] = useState<"text" | "template">(freeTextAllowed ? "text" : "template");
  const [text, setText] = useState("");
  const [selected, setSelected] = useState(availableTemplates[0]?.template_id ?? "");
  const [parameters, setParameters] = useState<string[]>([]);
  const [sending, setSending] = useState(false);
  useEffect(() => { if (!selected && availableTemplates[0]) setSelected(availableTemplates[0].template_id); }, [availableTemplates, selected]);
  const template = availableTemplates.find((item) => item.template_id === selected);
  const choose = (id: string) => { setSelected(id); setParameters([]); };
  const send = async () => {
    const payload = mode === "text" ? (text.trim() ? { text: text.trim() } : null) : (template && parameters.length === template.parameter_count && parameters.every(Boolean) ? { template_id: template.template_id, parameters } : null);
    if (!payload) return;
    setSending(true);
    try { await onReply(payload); setText(""); setParameters([]); } finally { setSending(false); }
  };
  return <section className="composer-wrap">
    {!freeTextAllowed && <p className="composer-notice">● The 24-hour WhatsApp service window has closed. Send an approved template to restart the conversation.</p>}
    <div className="composer-tabs"><button className={mode === "text" ? "active" : ""} disabled={!freeTextAllowed} onClick={() => setMode("text")}>Reply</button><button className={mode === "template" ? "active" : ""} onClick={() => setMode("template")}>Template (T)</button><span className="muted" style={{ marginLeft: "auto", fontSize: 12 }}>{freeTextAllowed ? "● WhatsApp free-text mode" : "Template dispatch only"}</span></div>
    {mode === "text" ? <div className="composer"><textarea value={text} onChange={(event) => setText(event.target.value)} placeholder="Type a message… (Ctrl+Enter to send)" onKeyDown={(event) => { if (event.ctrlKey && event.key === "Enter") send(); }} /><button className="btn primary" onClick={send} disabled={sending || !text.trim()}>{sending ? "Sending…" : "Send"}</button></div> : <div className="template-picker">
      {availableTemplates.length === 0 ? <p className="composer-notice">No approved WhatsApp templates are configured for this environment.</p> : <><select className="template-option" value={selected} onChange={(event) => choose(event.target.value)} aria-label="Approved template">{availableTemplates.map((item) => <option key={item.template_id} value={item.template_id}>{item.name} · {item.language.toUpperCase()}</option>)}</select>
      {Array.from({ length: template?.parameter_count ?? 0 }, (_, index) => <input className="template-option" key={index} value={parameters[index] ?? ""} onChange={(event) => setParameters((value) => { const next = [...value]; next[index] = event.target.value; return next; })} placeholder={`Parameter {{${index + 1}}}`} />)}
      <div className="composer"><span className="muted">Approved template · {template?.components.join(", ")}</span><button className="btn primary" onClick={send} disabled={sending || !template || parameters.length !== template.parameter_count || parameters.some((value) => !value)}>{sending ? "Sending…" : "Send template"}</button></div></>}
    </div>}
  </section>;
}
