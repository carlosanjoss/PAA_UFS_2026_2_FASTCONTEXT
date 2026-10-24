import { Component, type ErrorInfo, type ReactNode } from "react";

interface AppErrorBoundaryProps {
  children: ReactNode;
}

interface AppErrorBoundaryState {
  error: Error | null;
}

export class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("FastContext interface error", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <main className="app-crash-state" role="alert">
        <div>
          <span>FastContext</span>
          <h1>A interface encontrou um erro inesperado</h1>
          <p>Se o backend acabou de ser atualizado, reinicie o servidor e recarregue a aplicação.</p>
          <button type="button" onClick={() => window.location.reload()}>Recarregar aplicação</button>
        </div>
      </main>
    );
  }
}
