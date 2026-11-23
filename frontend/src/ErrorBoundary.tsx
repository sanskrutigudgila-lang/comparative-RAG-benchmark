import { Component, type ReactNode } from "react";

/** Keeps one failing tab (e.g. WebGL unavailable) from blanking the whole app. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { err: Error | null }> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) { return { err }; }
  render() {
    return this.state.err
      ? <div className="card err" role="alert">This tab failed to render: {this.state.err.message}</div>
      : this.props.children;
  }
}
