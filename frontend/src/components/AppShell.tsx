import {
  Activity,
  BarChart3,
  BookOpenText,
  Braces,
  LayoutDashboard,
  Menu,
  X,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import type { AppView, HealthResponse } from "../types";

const navigation: Array<{ id: AppView; label: string; description: string; icon: typeof Activity }> = [
  { id: "overview", label: "Visão geral", description: "Resumo do estudo", icon: LayoutDashboard },
  { id: "ask", label: "Perguntar", description: "Resposta e evidências", icon: BookOpenText },
  { id: "experiments", label: "Comparações", description: "Métricas e gráficos", icon: BarChart3 },
];

interface AppShellProps {
  activeView: AppView;
  health: HealthResponse | null;
  onNavigate: (view: AppView) => void;
  children: ReactNode;
}

export function AppShell({ activeView, health, onNavigate, children }: AppShellProps) {
  const [open, setOpen] = useState(false);

  const navigate = (view: AppView) => {
    onNavigate(view);
    setOpen(false);
  };

  return (
    <div className="app-shell">
      <button className="mobile-menu" onClick={() => setOpen(true)} aria-label="Abrir menu">
        <Menu size={21} />
      </button>
      {open && <button className="sidebar-scrim" onClick={() => setOpen(false)} aria-label="Fechar menu" />}

      <aside className={`sidebar ${open ? "sidebar--open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><Braces size={21} /></div>
          <div>
            <strong>FastContext</strong>
            <span>Algorithm Lab</span>
          </div>
          <button className="sidebar-close" onClick={() => setOpen(false)} aria-label="Fechar menu">
            <X size={19} />
          </button>
        </div>

        <nav className="navigation" aria-label="Navegação principal">
          <span className="nav-caption">Laboratório</span>
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-item ${activeView === item.id ? "nav-item--active" : ""}`}
                onClick={() => navigate(item.id)}
              >
                <Icon size={19} />
                <span><strong>{item.label}</strong><small>{item.description}</small></span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-card">
          <div className="status-line">
            <span className={`status-dot status-dot--${health?.status ?? "loading"}`} />
            <strong>{health?.status === "healthy" ? "Sistema pronto" : "Verificando sistema"}</strong>
          </div>
          <p>{health?.corpus_size?.toLocaleString("pt-BR") ?? "—"} trechos preparados</p>
          <div className="mini-code"><span>rank</span> score ↓ · id ↑</div>
        </div>

        <div className="sidebar-footer">
          <Activity size={17} />
          <span>Projeto e Análise de Algoritmos</span>
        </div>
      </aside>

      <main className="main-content">{children}</main>
    </div>
  );
}
