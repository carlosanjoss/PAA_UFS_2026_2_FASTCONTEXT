import type { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  description: string;
  badge?: string;
  children: ReactNode;
  className?: string;
}

export function ChartCard({ title, description, badge, children, className = "" }: ChartCardProps) {
  return (
    <article className={`panel chart-card ${className}`}>
      <div className="panel-heading">
        <div><h2>{title}</h2><p>{description}</p></div>
        {badge && <span className="panel-badge">{badge}</span>}
      </div>
      <div className="chart-body">{children}</div>
    </article>
  );
}
