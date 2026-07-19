// frontend/lib/sim-control.ts
// Module-level ref that stores the sendControl function from useFleetWebSocket.
// This allows TopBar (and any component) to call pause/resume without
// calling the hook a second time and creating duplicate intervals.

type ControlFn = (action: "pause" | "resume" | "status") => void;

let _controlFn: ControlFn | null = null;

export function registerSimControl(fn: ControlFn) {
  _controlFn = fn;
}

export function getSimControl(): ControlFn | null {
  return _controlFn;
}
