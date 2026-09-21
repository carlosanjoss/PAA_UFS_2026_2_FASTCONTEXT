import { AlertCircle, LoaderCircle } from "lucide-react";

export function LoadingState({ label = "Carregando dados reais…" }: { label?: string }) {
  return <div className="empty-state"><LoaderCircle className="spin" size={24} /><span>{label}</span></div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="empty-state empty-state--error"><AlertCircle size={24} /><span>{message}</span></div>;
}
