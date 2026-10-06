import Link from "next/link";

export function CaseListItem({ item, selected = false }: { item: any; selected?: boolean }) {
  const request = item.handoff_summary?.latest_customer_request ?? "Escalated customer request";
  const status = item.status === "claimed" ? "Claimed" : item.status === "resolved" ? "Resolved" : "Unassigned";
  const time = new Intl.DateTimeFormat("en", { hour: "numeric", minute: "2-digit" }).format(new Date(item.updated_at));
  return <Link className={`case ${selected ? "selected" : ""}`} href={`/cases/${item.case_id}`}>
    <div className="case-row"><span className="case-name">{item.handoff_summary?.customer_name ?? "Customer case"}</span><span className="case-time">{time}</span></div>
    <div className="case-preview">{request}</div><div className="case-meta"><span className={`status-pill ${item.status}`}>{status}</span><span>v{item.version}</span></div>
  </Link>;
}
