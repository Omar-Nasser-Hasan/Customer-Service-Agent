"use client";

import { useEffect, useMemo, useState } from "react";
import { templates, type Template } from "../../../lib/api";

export default function TemplateManager() {
  const [items, setItems] = useState<Template[]>([]);
  const [selected, setSelected] = useState<Template | null>(null);
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("all");
  const [error, setError] = useState("");
  const load = () => templates().then((value) => { setItems(value); setSelected((current) => current ?? value[0] ?? null); setError(""); }).catch(() => setError("Approved templates could not be loaded."));
  useEffect(() => { load(); }, []);
  const visible = useMemo(() => items.filter((item) => (language === "all" || item.language.toLowerCase().startsWith(language)) && `${item.name} ${item.template_id}`.toLowerCase().includes(query.toLowerCase())), [items, language, query]);
  return <main className="templates-page"><header className="templates-heading"><div><p className="eyebrow">TEMPLATES</p><h1>Approved <span className="gold">templates</span></h1><p>Synced from WhatsApp Business. Read-only dispatch catalog.</p></div><button className="btn" onClick={load}>↻ Sync now</button></header>
    <div className="filter-bar"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search templates…" aria-label="Search templates" /><button className={`filter-button ${language === "all" ? "active" : ""}`} onClick={() => setLanguage("all")}>All</button><button className={`filter-button ${language === "en" ? "active" : ""}`} onClick={() => setLanguage("en")}>EN</button><button className={`filter-button ${language === "ar" ? "active" : ""}`} onClick={() => setLanguage("ar")}>AR</button></div>
    {error && <div className="alert">{error}</div>}<div className="template-grid"><section className="template-list">{visible.length ? visible.map((item) => <article className={`template-card ${selected?.template_id === item.template_id ? "selected" : ""}`} key={item.template_id}><button onClick={() => setSelected(item)}><div className="template-name"><span>{item.name}</span><span className="status-pill">Approved</span></div><p className="template-copy">Deployment-owned WhatsApp template with {item.parameter_count} {item.parameter_count === 1 ? "parameter" : "parameters"}. Available components: {item.components.join(", ")}.</p><div className="chips"><span className="chip">{item.template_id}</span><span className="chip">{item.language.toUpperCase()}</span></div></button></article>) : <div className="empty-templates">No approved templates match these filters.</div>}</section>
      <aside className="template-side"><div className="profile-card"><p className="section-label">SELECTED TEMPLATE</p><h2>{selected?.name ?? "No template selected"}</h2><p>{selected ? `Template ID: ${selected.template_id}` : "Select an approved deployment template to inspect its dispatch schema."}</p></div><div className="profile-card"><p className="section-label">PARAMETER SCHEMA</p>{selected ? selected.parameter_count ? <div className="timeline">{Array.from({ length: selected.parameter_count }, (_, index) => <div className="timeline-item active" key={index}><b>{`{{${index + 1}}}`}</b><small>Required text parameter</small></div>)}</div> : <p>No variables are required.</p> : <p className="muted">No schema selected.</p>}</div><div className="profile-card"><p className="section-label">WHATSAPP CONFIGURATION</p><div className="order-line"><span>Language</span><b>{selected?.language.toUpperCase() ?? "—"}</b></div><div className="order-line"><span>Components</span><b>{selected?.components.join(", ") ?? "—"}</b></div><p className="muted">Only templates returned by the secured admin API can be dispatched.</p></div></aside>
    </div></main>;
}
