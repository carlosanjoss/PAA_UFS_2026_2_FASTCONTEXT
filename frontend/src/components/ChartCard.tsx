import type { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  description: string;
  badge?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function ChartCard({ title, description, badge, action, children, className = "" }: ChartCardProps) {
  return (
    <article className={`panel chart-card ${className}`}>
      <div className="panel-heading">
        <div><h2>{title}</h2><p>{description}</p></div>
        {(badge || action) && <div className="chart-heading-action">{badge && <span className="panel-badge">{badge}</span>}{action}</div>}
      </div>
      <div className="chart-body">{children}</div>
    </article>
  );
}
