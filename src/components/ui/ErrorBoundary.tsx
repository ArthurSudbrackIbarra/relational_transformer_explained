/**
 * Error boundary: a failing figure or chapter shows a calm message instead of
 * blanking the page. `fallback` may be a node or a render function.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react';

export interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  /** Change this value to reset the boundary (e.g. the route path). */
  resetKey?: unknown;
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface State {
  error: Error | null;
  key: unknown;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, State> {
  state: State = { error: null, key: this.props.resetKey };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  static getDerivedStateFromProps(props: ErrorBoundaryProps, state: State): Partial<State> | null {
    if (props.resetKey !== state.key) return { error: null, key: props.resetKey };
    return null;
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.props.onError?.(error, info);
  }

  reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const { fallback } = this.props;
    if (typeof fallback === 'function') return fallback(error, this.reset);
    if (fallback !== undefined) return fallback;
    return (
      <div role="alert" className="card my-6 p-5 text-[17px]">
        <p className="font-medium text-ink">This part of the page could not be rendered.</p>
        <p className="mt-1 text-ink-2">{error.message}</p>
        <button type="button" onClick={this.reset} className="mt-3 text-[16px] text-accent-text underline underline-offset-2">
          Try again
        </button>
      </div>
    );
  }
}
