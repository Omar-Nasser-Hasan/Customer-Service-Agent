"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { cases, session } from "../../lib/api";
import { subscribe } from "../../lib/websocket";
import { CaseListItem } from "../../components/cases/CaseListItem";
import { ScrollArea } from "../../components/ui/ScrollArea";

export default function Layout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [items, setItems] = useState<any[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let close: (() => void) | undefined;
    const load = () => cases().then(setItems).catch(() => undefined);

    session()
      .then(async () => {
        setReady(true);
        await load();
        close = await subscribe(() => load());
      })
      .catch(() => router.replace("/login"));

    return () => close?.();
  }, [router]);

  if (!ready) return <main className="thread">Checking staff session…</main>;

  return <div className="split"><aside className="pane"><h2 style={{ padding: "0 14px" }}>Cases</h2><ScrollArea>{items.map(item => <CaseListItem key={item.case_id} item={item} />)}</ScrollArea></aside><main>{children}</main></div>;
}
