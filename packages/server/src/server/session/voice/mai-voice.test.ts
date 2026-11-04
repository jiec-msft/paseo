import { once } from "node:events";
import { afterEach, describe, expect, test, vi } from "vitest";
import { WebSocketServer, type WebSocket } from "ws";
import { z } from "zod";
import type { SessionOutboundMessage } from "../../messages.js";
import { MaiVoiceSession } from "./mai-voice.js";

const PCM_FORMAT = "audio/pcm;rate=16000;bits=16";
const frames = z.record(z.string(), z.unknown());
const cleanup: Array<() => Promise<void>> = [];

afterEach(async () => {
  for (const close of cleanup.splice(0).toReversed()) await close();
});

async function createHarness(
  options: { initialize?: boolean; pendingPermissionIds?: string[] } = {},
) {
  const server = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await once(server, "listening");
  cleanup.push(() => new Promise((resolve) => server.close(() => resolve())));
  const address = server.address();
  if (typeof address === "string" || !address) throw new Error("Missing test socket address");
  const received: Array<z.infer<typeof frames>> = [];
  const emitted: SessionOutboundMessage[] = [];
  const errors: string[] = [];
  const tasks: string[] = [];
  let peer: WebSocket | null = null;
  server.on("connection", (socket) => {
    peer = socket;
    socket.on("message", (data) => {
      const message = frames.parse(JSON.parse(data.toString()));
      received.push(message);
      if (message.type === "start_session" && options.initialize !== false) {
        socket.send(JSON.stringify({ type: "session_init", session_id: "test-voice" }));
      }
    });
  });
  const session = new MaiVoiceSession({
    url: `ws://127.0.0.1:${address.port}`,
    token: "test-credential",
    agentId: "selected-agent",
    label: "Test agent",
    language: "zh-CN",
    pendingPermissionIds: options.pendingPermissionIds ?? [],
    emit: (message) => emitted.push(message),
    sendTask: async (text) => {
      tasks.push(text);
    },
    onError: (error) => errors.push(error.message),
  });
  cleanup.push(async () => session.close());
  function send(message: Record<string, unknown>) {
    if (!peer) throw new Error("Test socket is not connected");
    peer.send(JSON.stringify(message));
  }
  async function waitMessage(type: string, count = 1) {
    await vi.waitFor(() =>
      expect(received.filter((frame) => frame.type === type)).toHaveLength(count),
    );
    const result = received.findLast((frame) => frame.type === type);
    if (!result) throw new Error("Test message is missing");
    return result;
  }
  return { session, received, emitted, tasks, errors, send, message: waitMessage };
}

describe("MAI voice transport", () => {
  test("answers session inspection without running another agent task", async () => {
    const h = await createHarness();
    await h.session.connect();
    h.send({ type: "tool_call", call_id: "info", name: "get_session_info", args: {} });
    const result = await h.message("tool_result");
    expect(JSON.parse(String(result.result))).toMatchObject({
      total_sessions: 1,
      sessions: [{ id: "selected-agent", is_active: true }],
    });
    expect(h.tasks).toEqual([]);
  });

  test("grounds exactly one existing agent and streams PCM without a pipeline model", async () => {
    const h = await createHarness();
    await h.session.connect();
    expect(await h.message("start_session")).toMatchObject({
      auto_narrate: false,
      session_context: {
        sessions: [{ id: "selected-agent", is_active: true, agent_state: "idle" }],
        display_locale: "zh-CN",
      },
    });
    h.session.appendAudio("AAAA", PCM_FORMAT);
    expect(await h.message("ptt_start")).toMatchObject({
      has_active_session: false,
      passive: true,
    });
    expect(await h.message("ptt_audio_chunk")).toEqual({ type: "ptt_audio_chunk", audio: "AAAA" });
    expect(() => h.session.appendAudio("AAAA", "audio/wav")).toThrow("mono PCM16");
  });

  test("deduplicates task delivery and never claims the accepted task is complete", async () => {
    const h = await createHarness();
    await h.session.connect();
    const task = {
      type: "tool_call",
      call_id: "task-1",
      name: "send_to_chat",
      args: { text: "Write a demo note" },
    };
    h.send(task);
    h.send(task);
    const result = await h.message("tool_result");
    expect(h.tasks).toEqual(["Write a demo note"]);
    expect(JSON.parse(String(result.result))).toEqual({
      ok: true,
      request_status: "accepted",
      coding_task_status: "pending",
    });
  });

  test("refuses other sessions, approval tools and voice commands while approval is pending", async () => {
    const h = await createHarness({ pendingPermissionIds: ["permission-1"] });
    await h.session.connect();
    h.send({
      type: "tool_call",
      call_id: "wrong-target",
      name: "send_to_chat",
      args: { text: "Run", coding_session_id: "other-agent" },
    });
    expect(String((await h.message("tool_result")).result)).toContain("Only the selected");
    h.send({ type: "tool_call", call_id: "approve", name: "auto_approve_session", args: {} });
    expect(String((await h.message("tool_result", 2)).result)).toContain("sending tasks only");
    h.send({
      type: "tool_call",
      call_id: "blocked-task",
      name: "send_to_chat",
      args: { text: "Yes, allow it" },
    });
    expect(String((await h.message("tool_result", 3)).result)).toContain("approval buttons");
    expect(h.tasks).toEqual([]);
  });

  test("forwards final transcripts but does not execute them without a tool call", async () => {
    const h = await createHarness();
    await h.session.connect();
    h.send({ type: "transcription", status: "partial", text: "Run" });
    h.send({ type: "transcription", status: "final", text: "Run the demo", turn_id: "turn-1" });
    await vi.waitFor(() =>
      expect(h.emitted).toEqual([
        {
          type: "transcription_result",
          payload: { text: "Run the demo", requestId: "turn-1" },
        },
      ]),
    );
    expect(h.tasks).toEqual([]);
  });

  test("streams response audio, avoids speaker feedback and acknowledges actual playback", async () => {
    const h = await createHarness();
    await h.session.connect();
    h.send({
      type: "audio_response",
      audio: "AAAA",
      is_first_chunk: true,
      is_final: false,
      narration_id: "n1",
      playback_id: "p1",
    });
    h.send({
      type: "audio_response",
      audio: "BBBB",
      is_first_chunk: false,
      is_final: true,
      narration_id: "n1",
      playback_id: "p1",
    });
    await vi.waitFor(() => expect(h.emitted).toHaveLength(2));
    const outputs = h.emitted.filter((message) => message.type === "audio_output");
    expect(outputs.map((message) => message.payload)).toEqual([
      expect.objectContaining({ audio: "AAAA", format: "wav", chunkIndex: 0, isLastChunk: false }),
      expect.objectContaining({ audio: "BBBB", format: "wav", chunkIndex: 1, isLastChunk: true }),
    ]);
    expect(outputs[0].payload.groupId).toBe(outputs[1].payload.groupId);
    h.session.appendAudio("feedback", PCM_FORMAT);
    expect(h.received.filter((frame) => frame.type === "ptt_audio_chunk")).toEqual([]);
    h.session.audioPlayed(outputs[0].payload.id);
    expect(h.received.filter((frame) => frame.type === "narration_playback_complete")).toEqual([]);
    h.session.audioPlayed(outputs[1].payload.id);
    expect(await h.message("narration_playback_complete")).toEqual({
      type: "narration_playback_complete",
      coding_session_id: "selected-agent",
      narration_id: "n1",
      playback_id: "p1",
    });
  });

  test("narrates actual completion and queues permission alerts without granting them", async () => {
    const h = await createHarness();
    await h.session.connect();
    h.session.handleAgentEvent({ type: "turn_started", provider: "copilot" });
    h.session.handleAgentEvent({
      type: "timeline",
      provider: "copilot",
      item: { type: "assistant_message", text: "The demo " },
    });
    h.session.handleAgentEvent({
      type: "timeline",
      provider: "copilot",
      item: { type: "assistant_message", text: "note is ready." },
    });

    h.session.handleAgentEvent({ type: "turn_completed", provider: "copilot" });
    expect(await h.message("request_narration")).toMatchObject({
      coding_session_id: "selected-agent",
      kind: "response",
      text: "The demo note is ready.",
    });
    h.session.handleAgentEvent({
      type: "permission_requested",
      provider: "copilot",
      request: { id: "p1", provider: "copilot", kind: "tool", name: "write_file" },
    });
    h.send({
      type: "tool_call",
      call_id: "pending",
      name: "send_to_chat",
      args: { text: "Go ahead" },
    });
    expect(String((await h.message("tool_result")).result)).toContain("approval buttons");
    expect(h.tasks).toEqual([]);
  });

  test("grounds an approval narration in the exact pending request", async () => {
    const h = await createHarness();
    await h.session.connect();
    h.session.handleAgentEvent({
      type: "permission_requested",
      provider: "copilot",
      turnId: "turn-1",
      request: { id: "permission-1", provider: "copilot", kind: "tool", name: "write_file" },
    });
    expect(await h.message("session_context")).toMatchObject({
      mode: "delta",
      upserts: [
        {
          agent_state: "waiting_for_confirmation",
          pending: { type: "approval", pending_id: "permission-1", request_id: "turn-1" },
        },
      ],
    });
    expect(await h.message("request_narration")).toMatchObject({
      kind: "confirmation",
      pending_id: "permission-1",
      confirmation_type: "tool",
    });
    expect(h.tasks).toEqual([]);
  });

  test("surfaces sanitized service errors and stops accepting microphone input", async () => {
    const h = await createHarness();
    await h.session.connect();
    h.send({ type: "error", detail: "Do not expose test-credential" });
    await vi.waitFor(() =>
      expect(h.errors).toEqual([
        "MAI voice reported a session error. Check the endpoint and restart Voice mode.",
      ]),
    );
    expect(() => h.session.appendAudio("AAAA", PCM_FORMAT)).toThrow("not connected");
  });

  test("cancels a connection that has not finished initialization", async () => {
    const h = await createHarness({ initialize: false });
    const connecting = h.session.connect();
    const rejected = expect(connecting).rejects.toThrow("stopped");
    await h.message("start_session");
    h.session.close();
    await rejected;
    expect(h.errors).toEqual([]);
  });
});
