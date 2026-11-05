import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { promisify } from "node:util";
import { WebSocket } from "ws";
import { z } from "zod";
import type { AgentStreamEvent } from "../../agent/agent-sdk-types.js";
import type { SessionOutboundMessage } from "../../messages.js";

const execFileAsync = promisify(execFile);
const MAX_REPLY_LENGTH = 6000;
export const MAI_AUDIO_ID_PREFIX = "mai:";
const EventSchema = z.looseObject({
  type: z.string(),
  call_id: z.string().nullish(),
  name: z.string().nullish(),
  args: z.record(z.string(), z.unknown()).nullish(),
  text: z.string().nullish(),
  status: z.string().nullish(),
  turn_id: z.string().nullish(),
  audio: z.string().nullish(),
  is_first_chunk: z.boolean().nullish(),
  is_final: z.boolean().nullish(),
  narration_id: z.string().nullish(),
  playback_id: z.string().nullish(),
  disposition: z.string().nullish(),
  reason: z.string().nullish(),
});
type VoiceEvent = z.infer<typeof EventSchema>;

interface Playback {
  groupId: string;
  chunkIndex: number;
  finalChunkId: string | null;
  narrationId: string | undefined;
  playbackId: string | undefined;
}

interface Narration {
  id: string;
  kind: "response" | "confirmation";
  text: string;
  pendingId?: string;
}

interface PendingApproval {
  type: "approval";
  pending_id: string;
  request_id: string;
  message: string;
}

export interface MaiVoiceOptions {
  url: string;
  token: string;
  agentId: string;
  label: string;
  language: string;
  pendingPermissionIds: string[];
  emit(message: SessionOutboundMessage): void;
  sendTask(text: string): Promise<void>;
  onError(error: Error): void;
}

export async function getMaiVoiceToken(): Promise<string> {
  const configured = process.env.PASEO_MAI_VOICE_TOKEN?.trim();
  if (configured) return configured;
  try {
    const { stdout } = await execFileAsync("gh", ["auth", "token", "--hostname", "github.com"], {
      timeout: 10_000,
      windowsHide: true,
    });
    const token = stdout.trim();
    if (token) return token;
  } catch {
    // The subprocess error can contain credential output; never propagate it.
    throw new Error("MAI voice authentication failed. Run gh auth login on the daemon host.");
  }
  throw new Error("MAI voice needs a GitHub token on the daemon host.");
}

/** Hosted speech controls conversation only; the selected Paseo agent owns execution. */
export class MaiVoiceSession {
  private socket: WebSocket | null = null;
  private closed = false;
  private ready = false;
  private inputTurnId: string | null = null;
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private resumeTimer: ReturnType<typeof setTimeout> | null = null;
  private outputTimer: ReturnType<typeof setTimeout> | null = null;
  private cancelConnect: (() => void) | null = null;
  private awaitingPong = false;
  private playback: Playback | null = null;
  private readonly callIds = new Set<string>();
  private readonly narrations: Narration[] = [];
  private narrationInFlight: Narration | null = null;
  private readonly pendingApprovals = new Map<string, PendingApproval>();
  private reply = "";
  private agentState = "idle";

  constructor(private readonly options: MaiVoiceOptions) {
    for (const id of options.pendingPermissionIds) {
      this.pendingApprovals.set(id, {
        type: "approval",
        pending_id: id,
        request_id: id,
        message: "Approval is required in Paseo.",
      });
    }
    if (this.pendingApprovals.size > 0) this.agentState = "waiting_for_confirmation";
  }

  private get pending(): PendingApproval | null {
    return this.pendingApprovals.values().next().value ?? null;
  }

  async connect(): Promise<void> {
    let url: URL;
    try {
      url = new URL(this.options.url);
    } catch {
      throw new Error("PASEO_MAI_VOICE_URL must be a valid WebSocket URL.");
    }
    const localTestSocket = url.protocol === "ws:" && url.hostname === "127.0.0.1";
    if (
      (!localTestSocket && url.protocol !== "wss:") ||
      url.username ||
      url.password ||
      url.search ||
      url.hash
    ) {
      throw new Error(
        "PASEO_MAI_VOICE_URL must be a wss URL without credentials or query parameters.",
      );
    }
    if (this.closed) throw new Error("MAI voice was stopped.");
    url.searchParams.set("token", this.options.token);
    const socket = new WebSocket(url, { handshakeTimeout: 10_000, maxPayload: 4 * 1024 * 1024 });
    this.socket = socket;
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => fail("MAI voice session initialization timed out."), 15_000);
      const finish = () => {
        clearTimeout(timeout);
        this.cancelConnect = null;
      };
      const fail = (message: string) => {
        finish();
        reject(new Error(message));
        this.fail(new Error(message));
      };
      this.cancelConnect = () => {
        finish();
        reject(new Error("MAI voice was stopped."));
      };
      socket.on("open", () => {
        this.send({
          type: "start_session",
          machine_id: randomUUID(),
          session_context: this.context(),
          auto_narrate: false,
          turn_config: {
            auto_end_mode: "vad",
            silence_ms: 1500,
            stop_phrases: [],
            vad_gate_asr: true,
          },
          voice_instructions:
            "You are a voice interface for one Paseo coding agent. Send tasks only to the active " +
            "session. Never approve permissions, change models, create sessions or claim work " +
            "is complete before receiving results. Permission requests must be answered in the UI.",
        });
        this.send({ type: "set_language", language: this.options.language });
      });
      socket.on("message", (data) => {
        try {
          const event = EventSchema.parse(JSON.parse(data.toString()));
          if (event.type === "session_init") {
            if (this.ready) return;
            this.ready = true;
            finish();
            this.startHeartbeat();
            resolve();
          }
          this.receive(event);
        } catch (error) {
          const fields =
            error instanceof z.ZodError
              ? error.issues.map((issue) => issue.path.join(".")).join(", ")
              : "JSON";
          fail(`MAI voice returned an invalid message (${fields}).`);
        }
      });
      socket.on("error", () =>
        fail("MAI voice connection failed. Check the endpoint and GitHub login."),
      );
      socket.on("close", (code) => {
        if (!this.closed)
          fail(`MAI voice disconnected (code ${code}). Start Voice mode again to retry.`);
      });
    });
  }

  appendAudio(audio: string, format: string): void {
    if (!this.ready || this.closed) throw new Error("MAI voice is not connected.");
    if (format !== "audio/pcm;rate=16000;bits=16") {
      throw new Error("MAI voice requires mono PCM16 audio at 16000 Hz.");
    }
    // Half-duplex for this preview: do not feed speaker output back into recognition.
    if (this.playback || this.resumeTimer) return;
    if (!this.inputTurnId) {
      this.inputTurnId = randomUUID();
      this.send({
        type: "ptt_start",
        turn_id: this.inputTurnId,
        has_active_session: this.agentState === "thinking",
        passive: true,
      });
    }
    this.send({ type: "ptt_audio_chunk", audio });
  }

  handleAgentEvent(event: AgentStreamEvent): void {
    if (!this.ready || this.closed) return;
    switch (event.type) {
      case "turn_started":
        this.reply = "";
        this.agentState = "thinking";
        this.syncContext();
        break;
      case "timeline":
        if (event.item.type === "assistant_message") {
          this.reply = (this.reply + event.item.text).slice(-MAX_REPLY_LENGTH);
        }
        break;
      case "permission_requested":
        if (this.pendingApprovals.has(event.request.id)) break;
        this.agentState = "waiting_for_confirmation";
        const pending: PendingApproval = {
          type: "approval",
          pending_id: event.request.id,
          request_id: event.turnId ?? event.request.id,
          message: `Approval needed: ${event.request.title ?? event.request.name}. Please use the approval buttons in Paseo.`,
        };
        this.pendingApprovals.set(event.request.id, pending);
        this.syncContext();
        this.narrate("confirmation", pending.message, event.request.id);
        break;
      case "permission_resolved":
        this.pendingApprovals.delete(event.requestId);
        this.agentState = this.pendingApprovals.size > 0 ? "waiting_for_confirmation" : "thinking";
        this.syncContext();
        this.flushNarration();
        break;
      case "turn_completed":
        this.agentState = "idle";
        this.syncContext();
        this.narrate(
          "response",
          this.reply || "The agent finished its turn. Check Paseo for the result.",
        );
        break;
      case "turn_failed":
        this.agentState = "idle";
        this.syncContext();
        this.narrate("response", `The task failed: ${event.error}`);
        break;
      case "turn_canceled":
        this.agentState = "idle";
        this.syncContext();
        this.narrate("response", "The task was canceled.");
        break;
    }
  }

  audioPlayed(id: string): void {
    const playback = this.playback;
    if (!playback || playback.finalChunkId !== id) return;
    if (playback.narrationId && playback.playbackId) {
      this.send({
        type: "narration_playback_complete",
        coding_session_id: this.options.agentId,
        narration_id: playback.narrationId,
        playback_id: playback.playbackId,
      });
    }
    if (playback.narrationId === this.narrationInFlight?.id) this.narrationInFlight = null;
    this.playback = null;
    if (this.outputTimer) clearTimeout(this.outputTimer);
    this.outputTimer = null;
    this.resumeTimer = setTimeout(() => {
      this.resumeTimer = null;
      this.flushNarration();
    }, 300);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.ready = false;
    this.cancelConnect?.();
    if (this.heartbeat) clearInterval(this.heartbeat);
    if (this.resumeTimer) clearTimeout(this.resumeTimer);
    if (this.outputTimer) clearTimeout(this.outputTimer);
    this.heartbeat = null;
    this.resumeTimer = null;
    this.outputTimer = null;
    this.narrations.length = 0;
    this.playback = null;
    this.socket?.terminate();
    this.socket = null;
  }

  private context() {
    return {
      display_locale: this.options.language,
      sessions: [
        {
          id: this.options.agentId,
          label: this.options.label,
          session_type: "agent",
          is_active: true,
          agent_state: this.agentState,
          last_response_summary: this.reply,
          pending: this.pending,
          ...(this.pending
            ? { agent_state_detail: this.pending.message, confirmation_type: "tool" }
            : {}),
        },
      ],
    };
  }

  private syncContext(): void {
    this.send({
      type: "session_context",
      mode: "delta",
      upserts: this.context().sessions,
      removes: [],
    });
  }

  private send(message: Record<string, unknown>): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    if (this.socket.bufferedAmount > 1024 * 1024) {
      this.fail(new Error("MAI voice connection is too slow. Restart Voice mode."));
      return;
    }
    this.socket.send(JSON.stringify(message));
  }

  private fail(error: Error): void {
    if (this.closed) return;
    this.close();
    this.options.onError(error);
  }

  private startHeartbeat(): void {
    this.heartbeat = setInterval(() => {
      if (this.awaitingPong) {
        this.fail(new Error("MAI voice heartbeat timed out. Start Voice mode again to retry."));
        return;
      }
      this.awaitingPong = true;
      this.send({ type: "ping" });
    }, 25_000);
  }

  private receive(event: VoiceEvent): void {
    switch (event.type) {
      case "pong":
        this.awaitingPong = false;
        break;
      case "speech_started":
        this.options.emit({ type: "voice_input_state", payload: { isSpeaking: true } });
        break;
      case "turn_auto_ended":
        this.inputTurnId = null;
        this.options.emit({ type: "voice_input_state", payload: { isSpeaking: false } });
        break;
      case "transcription":
        if (event.status === "partial" || !event.text?.trim()) break;
        this.options.emit({
          type: "transcription_result",
          payload: { text: event.text, requestId: event.turn_id ?? randomUUID() },
        });
        break;
      case "audio_response":
        this.receiveAudio(event);
        break;
      case "tool_call":
        void this.dispatchTool(event);
        break;
      case "narration_ack":
        this.handleNarrationAck(event);
        break;
      case "narration_unblocked":
        if (this.narrationInFlight && event.narration_id === this.narrationInFlight.id) {
          const narration = this.narrationInFlight;
          this.narrationInFlight = null;
          this.narrations.unshift(narration);
          this.flushNarration();
        }
        break;
      case "error":
        this.fail(
          new Error(
            "MAI voice reported a session error. Check the endpoint and restart Voice mode.",
          ),
        );
        break;
    }
  }

  private handleNarrationAck(event: VoiceEvent): void {
    if (event.narration_id !== this.narrationInFlight?.id) return;
    if (event.disposition !== "invalid" && event.disposition !== "suppressed") return;
    const pendingId = this.narrationInFlight?.pendingId;
    if (pendingId && !this.pendingApprovals.has(pendingId)) {
      this.narrationInFlight = null;
      this.flushNarration();
      return;
    }
    const reason = event.reason && /^[a-z_]{1,80}$/.test(event.reason) ? ` (${event.reason})` : "";
    this.fail(
      new Error(`MAI voice could not narrate the agent update${reason}. Restart Voice mode.`),
    );
  }

  private receiveAudio(event: VoiceEvent): void {
    this.watchOutput();
    if (event.is_first_chunk !== false || !this.playback) {
      this.inputTurnId = null;
      this.send({ type: "ptt_end" });
      this.playback = {
        groupId: randomUUID(),
        chunkIndex: 0,
        finalChunkId: null,
        narrationId: event.narration_id ?? undefined,
        playbackId: event.playback_id ?? undefined,
      };
    }
    const playback = this.playback;
    const id = `${MAI_AUDIO_ID_PREFIX}${randomUUID()}`;
    if (event.is_final) playback.finalChunkId = id;
    this.options.emit({
      type: "audio_output",
      payload: {
        audio: event.audio ?? "",
        format: "wav",
        id,
        isVoiceMode: true,
        groupId: playback.groupId,
        chunkIndex: playback.chunkIndex++,
        isLastChunk: event.is_final ?? false,
      },
    });
  }

  private async dispatchTool(event: VoiceEvent): Promise<void> {
    const callId = event.call_id;
    if (!callId || this.callIds.has(callId)) return;
    if (this.callIds.size >= 256) {
      this.fail(new Error("MAI voice preview session limit reached. Start Voice mode again."));
      return;
    }
    this.callIds.add(callId);
    const args = event.args ?? {};
    const target = args.coding_session_id;
    let result: string;
    try {
      if (target !== undefined && target !== this.options.agentId) {
        throw new Error("Only the selected Paseo agent is available.");
      }
      if (event.name === "get_session_info") {
        result = JSON.stringify({ total_sessions: 1, sessions: this.context().sessions });
      } else {
        if (event.name !== "send_to_chat") {
          throw new Error(
            "This preview supports sending tasks only, plus reading session information. Use Paseo controls for approvals and other actions.",
          );
        }
        const text = z.string().trim().min(1).max(16_000).parse(args.text);
        if (this.pendingApprovals.size > 0) {
          throw new Error("Use Paseo approval buttons before sending another task.");
        }
        await this.options.sendTask(text);
        result = JSON.stringify({
          ok: true,
          request_status: "accepted",
          coding_task_status: "pending",
        });
      }
    } catch (error) {
      result = JSON.stringify({
        ok: false,
        error: error instanceof Error ? error.message : "Task was not sent.",
      });
    }
    if (!this.closed) this.send({ type: "tool_result", call_id: callId, result });
  }

  private narrate(kind: Narration["kind"], text: string, pendingId?: string): void {
    if (this.narrations.length >= 16) {
      this.fail(
        new Error("MAI voice updates exceeded the preview queue limit. Restart Voice mode."),
      );
      return;
    }
    this.narrations.push({
      id: randomUUID(),
      kind,
      text: text.slice(0, MAX_REPLY_LENGTH),
      pendingId,
    });
    this.flushNarration();
  }

  private flushNarration(): void {
    if (this.closed || this.playback || this.narrationInFlight) return;
    let narration = this.narrations[0];
    while (narration?.pendingId && !this.pendingApprovals.has(narration.pendingId)) {
      this.narrations.shift();
      narration = this.narrations[0];
    }
    if (!narration) {
      if (this.outputTimer) clearTimeout(this.outputTimer);
      this.outputTimer = null;
      return;
    }
    if (narration.pendingId && narration.pendingId !== this.pending?.pending_id) return;
    this.narrations.shift();
    this.narrationInFlight = narration;
    this.watchOutput();
    this.send({
      type: "request_narration",
      coding_session_id: this.options.agentId,
      kind: narration.kind,
      text: narration.text,
      narration_id: narration.id,
      ...(narration.kind === "confirmation" ? { confirmation_type: "tool" } : {}),
      ...(narration.pendingId ? { pending_id: narration.pendingId } : {}),
    });
  }

  private watchOutput(): void {
    if (this.outputTimer) clearTimeout(this.outputTimer);
    this.outputTimer = setTimeout(
      () => this.fail(new Error("MAI voice output timed out. Restart Voice mode.")),
      120_000,
    );
  }
}
