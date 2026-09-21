import type { LucideIcon } from "lucide-react";

interface KpiCardProps {
  label: string;
  value: string;
  detail: string;
  icon: LucideIcon;
  tone?: "coral" | "blue" | "green" | "amber";
}

export function KpiCard({ label, value, detail, icon: Icon, tone = "coral" }: KpiCardProps) {
  return (
    <article className="kpi-card">
      <div className={`kpi-icon kpi-icon--${tone}`}><Icon size={20} /></div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}
