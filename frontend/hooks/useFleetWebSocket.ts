"use client";

import { useCallback, useEffect, useRef } from "react";
import toast from "react-hot-toast";
import { WS_URL } from "@/lib/api-client";
import { registerSimControl } from "@/lib/sim-control";
import { useFleetStore } from "@/stores/fleet-store";
import type { AgentAlert, CyclePayload, FleetSnapshotResponse, WSMessage } from "@/types";

const RECONNECT_MS = 2000;

export function useFleetWebSocket() {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(false);
  const { setWsConnected, updateLiveEngine, addAlert, setGlobalCycle } = useFleetStore();

  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      if (!mountedRef.current) return;
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
      setWsConnected(true);
      toast.success("Backend telemetry connected", { id: "ws-connected", duration: 2500 });
    };

      socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as WSMessage;
      if (message.type === "cycle_update") {
        for (const payload of message.payload as CyclePayload[]) {
          if (!payload.warming_up) updateLiveEngine(payload);
        }
        if (message.global_cycle != null) setGlobalCycle(message.global_cycle);
      } else if (message.type === "initial_snapshot") {
        const snapshot = message.payload as FleetSnapshotResponse;
        for (const payload of snapshot.engines ?? []) updateLiveEngine(payload);
      } else if (message.type === "agent_alert") {
        addAlert(message.payload as AgentAlert);
      }
    };

      socket.onclose = () => {
      setWsConnected(false);
      if (mountedRef.current) reconnectRef.current = setTimeout(connect, RECONNECT_MS);
    };
      socket.onerror = () => socket.close();
    };

    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      socketRef.current?.close();
      setWsConnected(false);
    };
  }, [addAlert, setGlobalCycle, setWsConnected, updateLiveEngine]);

  const sendControl = useCallback((action: "pause" | "resume" | "status") => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) {
      toast.error("Backend telemetry is not connected");
      return;
    }
    socketRef.current.send(JSON.stringify({ action }));
  }, []);

  useEffect(() => registerSimControl(sendControl), [sendControl]);
  return { sendControl };
}
