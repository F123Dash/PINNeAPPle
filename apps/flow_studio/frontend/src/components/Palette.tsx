import { useMemo, useState } from "react";
import type { NodeType } from "../types";

export function Palette({ nodes, onAdd, onNewType }: { nodes: NodeType[]; onAdd: (t: string) => void; onNewType?: () => void }) {
  const [q, setQ] = useState("");
  const groups = useMemo(() => {
    const out: Record<string, NodeType[]> = {};
    for (const n of nodes) {
      const hay = `${n.title} ${n.type} ${n.category} ${n.description}`.toLowerCase();
      if (q && !hay.includes(q.toLowerCase())) continue;
      (out[n.category] ||= []).push(n);
    }
    // curated categories first; the nodes generated from the public API ("Library: ...") after them
    return Object.fromEntries(Object.entries(out).sort(([a], [b]) =>
      Number(a.startsWith("Library:")) - Number(b.startsWith("Library:")) || a.localeCompare(b)));
  }, [nodes, q]);
  return (
    <aside className="w-60 shrink-0 border-r border-slate-200 bg-white flex flex-col" data-testid="palette">
      <div className="p-2 border-b border-slate-100">
        <input
          className="w-full text-sm border border-slate-200 rounded px-2 py-1"
          placeholder="Search nodes…" value={q} onChange={(e) => setQ(e.target.value)}
        />
      </div>
      <div className="overflow-y-auto flex-1 p-2 space-y-3">
        {Object.entries(groups).map(([cat, list]) => (
          <div key={cat}>
            <div className="text-[10.5px] uppercase tracking-wider text-slate-400 font-semibold mb-1">{cat}</div>
            {list.map((n) => (
              <div
                key={n.type}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData("application/pinneapple-node", n.type);
                  e.dataTransfer.effectAllowed = "move";
                }}
                onDoubleClick={() => onAdd(n.type)}
                title={`${n.description}\n\nwraps: ${n.calls}${n.available ? "" : `\n\nunavailable: ${n.unavailable_reason}`}`}
                className={`text-[12.5px] px-2 py-1.5 mb-1 rounded border cursor-grab active:cursor-grabbing select-none ${n.available ? "border-slate-200 bg-slate-50 hover:bg-brand-50 hover:border-brand-500" : "border-dashed border-slate-300 text-slate-400"}`}
                data-testid={`palette-${n.type}`}
              >
                {n.title}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="p-2 text-[10.5px] text-slate-400 border-t">
        Drag onto the canvas (or double-click).
        {onNewType && <button className="block mt-1 text-sky-700" onClick={onNewType} data-testid="new-node-type">+ new node type from a library call</button>}
      </div>
    </aside>
  );
}
