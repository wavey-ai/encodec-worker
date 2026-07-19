import { availableParallelism } from "node:os";
import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Worker } from "node:worker_threads";

const serverRoot = path.dirname(fileURLToPath(import.meta.url));
const assetRoot = path.resolve(process.env.ENCODEC_ASSET_ROOT || path.join(serverRoot, "..", "runtime"));
const workerScript = path.join(serverRoot, "encode-worker.mjs");
const profiles = Object.freeze([
  "encodec_48khz_6kbps_1333ms",
  "encodec_48khz_6kbps_1800ms",
  "encodec_48khz_12kbps_1333ms",
  "encodec_48khz_12kbps_1800ms",
]);
const defaultProfile = process.env.ENCODEC_DEFAULT_PROFILE || "encodec_48khz_12kbps_1333ms";
const workerCount = Math.max(1, Math.min(16, Number(process.env.ENCODEC_WORKERS) || Math.min(4, availableParallelism())));
const port = Math.max(1, Number(process.env.PORT) || 8787);
const host = process.env.HOST || "127.0.0.1";
const originToken = String(process.env.ENCODEC_ORIGIN_TOKEN || "");
const maxQueue = Math.max(1, Number(process.env.ENCODEC_MAX_QUEUE) || 20);
const maxSegments = Math.max(1, Number(process.env.ENCODEC_MAX_SEGMENTS) || 2048);
const activeLeaseMs = Math.max(30_000, Number(process.env.ENCODEC_ACTIVE_LEASE_MS) || 90_000);
const queuedLeaseMs = Math.max(5000, Number(process.env.ENCODEC_QUEUED_LEASE_MS) || 15_000);
const heartbeatIntervalMs = Math.max(500, Number(process.env.ENCODEC_HEARTBEAT_INTERVAL_MS) || 1000);
const finishedRetentionMs = Math.max(60_000, Number(process.env.ENCODEC_FINISHED_RETENTION_MS) || 10 * 60_000);
const maxBatchBytes = 12 * 1024 * 1024;
const batchStreamContentType = "text/event-stream; charset=utf-8";

function secureEqual(left, right) {
  const a = Buffer.from(String(left || ""));
  const b = Buffer.from(String(right || ""));
  return a.length === b.length && a.length > 0 && timingSafeEqual(a, b);
}

function authorized(request) {
  if (!originToken) return true;
  const header = String(request.headers.authorization || "");
  if (!header) return true;
  return header.startsWith("Bearer ") && secureEqual(header.slice(7), originToken);
}

function applyCors(request, response) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
  response.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type,X-Encodec-Session-Token,Authorization",
  );
  response.setHeader("Access-Control-Max-Age", "86400");
  response.setHeader("Vary", "Origin");
}

function id(bytes = 18) {
  return randomBytes(bytes).toString("base64url");
}

function json(response, status, body, headers = {}) {
  const payload = Buffer.from(JSON.stringify(body));
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": String(payload.length),
    "Cache-Control": "no-store",
    ...headers,
  });
  response.end(payload);
}

function errorJson(response, status, code, message, extra = {}) {
  json(response, status, { ok: false, error: code, message, ...extra });
}

function wantsBatchStream(request) {
  return String(request.headers.accept || "").toLowerCase().includes("text/event-stream");
}

function startBatchStream(response, session) {
  response.writeHead(200, {
    "Content-Type": batchStreamContentType,
    "Cache-Control": "no-store, no-transform",
    "X-Accel-Buffering": "no",
  });
  response.flushHeaders?.();
  writeBatchStreamEvent(response, "start", { session: publicSession(session) });
}

function writeBatchStreamEvent(response, event, payload) {
  if (response.destroyed || response.writableEnded) return false;
  return response.write(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
}

function writeBatchStreamHeartbeat(response) {
  if (response.destroyed || response.writableEnded) return false;
  return response.write(`: heartbeat ${Date.now()}\n\n`);
}

async function readBody(request, maxBytes) {
  const contentLength = Number(request.headers["content-length"] || 0);
  if (contentLength > maxBytes) throw Object.assign(new Error("Request body is too large"), { status: 413 });
  const chunks = [];
  let byteLength = 0;
  for await (const chunk of request) {
    byteLength += chunk.byteLength;
    if (byteLength > maxBytes) throw Object.assign(new Error("Request body is too large"), { status: 413 });
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, byteLength);
}

async function readJson(request, maxBytes = 64 * 1024) {
  const body = await readBody(request, maxBytes);
  try {
    return JSON.parse(body.toString("utf8") || "{}");
  } catch (_error) {
    throw Object.assign(new Error("Request body is not valid JSON"), { status: 400 });
  }
}

class EncoderPool {
  constructor(count) {
    this.count = count;
    this.taskId = 0;
    this.queue = [];
    this.workers = [];
    for (let index = 0; index < count; index += 1) this.spawn(index);
  }

  spawn(index) {
    const worker = new Worker(workerScript, {
      workerData: { assetRoot, profiles },
    });
    const entry = { index, worker, busy: false, task: null };
    this.workers[index] = entry;
    worker.on("message", (message) => this.complete(entry, message));
    worker.on("error", (error) => this.fail(entry, error));
    worker.on("exit", (code) => {
      if (this.workers[index] !== entry) return;
      if (code !== 0) this.fail(entry, new Error(`Encoder worker ${index} exited with code ${code}`));
    });
  }

  run(task, transfer = []) {
    return new Promise((resolve, reject) => {
      this.queue.push({ ...task, id: ++this.taskId, transfer, resolve, reject });
      this.drain();
    });
  }

  drain() {
    for (const entry of this.workers) {
      if (!entry || entry.busy || !this.queue.length) continue;
      const task = this.queue.shift();
      entry.busy = true;
      entry.task = task;
      const { transfer, resolve: _resolve, reject: _reject, ...message } = task;
      entry.worker.postMessage(message, transfer);
    }
  }

  complete(entry, message) {
    if (!entry.task || message?.id !== entry.task.id) return;
    const task = entry.task;
    entry.task = null;
    entry.busy = false;
    if (message.ok) task.resolve(message.result);
    else {
      const error = new Error(message?.error?.message || "Encoder worker failed");
      error.name = message?.error?.name || "Error";
      if (message?.error?.stack) error.stack = message.error.stack;
      task.reject(error);
    }
    this.drain();
  }

  fail(entry, error) {
    if (this.workers[entry.index] !== entry) return;
    const task = entry.task;
    this.workers[entry.index] = null;
    try { entry.worker.terminate(); } catch (_error) {}
    if (task) task.reject(error);
    this.spawn(entry.index);
    this.drain();
  }

  async warm(profile) {
    return Promise.all(Array.from({ length: this.count }, () => this.run({ type: "warm", profile })));
  }

  encode(task) {
    return this.run({ type: "encode", ...task }, [task.audioBuffer]);
  }

  async close() {
    await Promise.all(this.workers.filter(Boolean).map((entry) => entry.worker.terminate()));
  }
}

const pool = new EncoderPool(workerCount);
const descriptorCache = new Map();
let encodecModulePromise = null;

async function ensureDescriptorModule() {
  if (!encodecModulePromise) {
    encodecModulePromise = Promise.all([
      import(pathToFileURL(path.join(assetRoot, "encodec", "pkg", "encodec_rs.js")).href),
      readFile(path.join(assetRoot, "encodec", "pkg", "encodec_rs_bg.wasm")),
    ]).then(([module, wasmBytes]) => {
      module.initSync({ module: wasmBytes });
      module.initPanicHook?.();
      return module;
    });
  }
  return encodecModulePromise;
}

function payloadDescriptor(header, meta) {
  const bytes = header instanceof Uint8Array ? header : new Uint8Array(header);
  if (bytes.length < 9 || Buffer.from(bytes.subarray(0, 4)).toString("ascii") !== "ECDC") {
    throw new Error("Generated ECDC header is invalid");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const metadataLength = view.getUint32(5, false);
  const metadataEnd = 9 + metadataLength;
  if (metadataLength <= 0 || metadataEnd > bytes.length) throw new Error("Generated ECDC metadata is truncated");
  const codecMetadata = Array.from(bytes.subarray(9, metadataEnd));
  JSON.parse(new TextDecoder().decode(bytes.subarray(9, metadataEnd)));
  const outputSamples = Number(meta.segment_stride);
  const blockSamples = Number(meta.segment_samples);
  return {
    container: "ECDC",
    codec: "ECDC",
    sampleRate: Number(meta.sample_rate),
    channels: Number(meta.channels),
    blockSamples,
    outputOffsetSamples: Math.max(0, Math.floor((blockSamples - outputSamples) / 2)),
    outputSamples,
    codecMetadata,
  };
}

async function descriptorForProfile(profile) {
  if (descriptorCache.has(profile)) return descriptorCache.get(profile);
  const promise = Promise.all([
    ensureDescriptorModule(),
    readFile(path.join(assetRoot, "bundles", profile, "bundle.json"), "utf8"),
    readFile(path.join(assetRoot, "bundles", profile, "lm_weights_q8.bin")),
  ]).then(([module, bundleJson, lmBytes]) => {
    const meta = JSON.parse(bundleJson);
    const lmWeights = new Uint8Array(lmBytes);
    const header = module.lmEcdcFixedHeaderForWeights(bundleJson, Number(meta.segment_stride), 2, lmWeights);
    const lmHash = createHash("sha256").update(lmBytes).digest("hex");
    const cacheKeyPrefix = [
      "ecdc-lm-fixed-frame-v2",
      profile,
      meta.encode_model || "",
      meta.lm_quant_weight_model || "",
      String(meta.sample_rate || ""),
      String(meta.channels || ""),
      String(meta.segment_samples || ""),
      String(meta.segment_stride || ""),
      String(meta.frame_length || ""),
      String(meta.num_codebooks || ""),
      lmHash,
    ].join("|");
    return { descriptor: payloadDescriptor(header, meta), cacheKeyPrefix, meta };
  }).catch((error) => {
    descriptorCache.delete(profile);
    throw error;
  });
  descriptorCache.set(profile, promise);
  return promise;
}

const sessions = new Map();
const queue = [];
let activeSession = null;

function positionOf(session) {
  if (session.state === "active") return 0;
  if (session.state !== "queued") return null;
  const index = queue.indexOf(session);
  return index < 0 ? null : index + 1;
}

function publicSession(session, { includeToken = false } = {}) {
  const position = positionOf(session);
  const now = Date.now();
  const leaseRemainingMs = Math.max(0, Math.floor(Number(session.expiresAt) || 0) - now);
  return {
    id: session.id,
    state: session.state,
    profile: session.profile,
    segmentCount: session.segmentCount,
    completedSegments: session.results.size,
    queuePosition: position,
    workerCount,
    maxBatchSegments: workerCount,
    descriptor: session.descriptor,
    cacheKeyPrefix: session.cacheKeyPrefix,
    createdAt: new Date(session.createdAt).toISOString(),
    activatedAt: session.activatedAt ? new Date(session.activatedAt).toISOString() : null,
    expiresAt: new Date(session.expiresAt).toISOString(),
    serverTime: new Date(now).toISOString(),
    heartbeat: {
      transport: "poll",
      intervalMs: heartbeatIntervalMs,
      timeoutMs: session.state === "queued" ? queuedLeaseMs : activeLeaseMs,
      leaseRemainingMs,
      queueDepth: queue.length,
    },
    ...(includeToken ? { sessionToken: session.token } : {}),
  };
}

function activateNext() {
  if (activeSession || !queue.length) return;
  const session = queue.shift();
  if (!session || session.state !== "queued") return activateNext();
  session.state = "active";
  session.activatedAt = Date.now();
  session.expiresAt = Date.now() + activeLeaseMs;
  activeSession = session;
}

function finishSession(session, state) {
  session.state = state;
  session.finishedAt = Date.now();
  session.expiresAt = Date.now() + finishedRetentionMs;
  if (activeSession === session) activeSession = null;
  const queuedIndex = queue.indexOf(session);
  if (queuedIndex >= 0) queue.splice(queuedIndex, 1);
  activateNext();
}

function owner(request) {
  return String(request.headers["x-encodec-owner"] || "anonymous").trim().slice(0, 256) || "anonymous";
}

function verifySessionAccess(request, response, session) {
  if (!session) {
    errorJson(response, 404, "session_not_found", "Encode session was not found");
    return false;
  }
  if (!secureEqual(request.headers["x-encodec-session-token"], session.token)) {
    errorJson(response, 403, "invalid_session_token", "Encode session token is invalid");
    return false;
  }
  return true;
}

function parseBatch(body, session) {
  if (body.length < 8 || body.subarray(0, 4).toString("ascii") !== "YECB") {
    throw Object.assign(new Error("Batch magic is invalid"), { status: 400 });
  }
  const version = body.readUInt16BE(4);
  const count = body.readUInt16BE(6);
  if (version !== 1) throw Object.assign(new Error(`Unsupported batch version ${version}`), { status: 422 });
  if (count <= 0 || count > workerCount) {
    throw Object.assign(new Error(`Batch contains ${count} segments; expected 1-${workerCount}`), { status: 422 });
  }
  const tableEnd = 8 + count * 16;
  if (tableEnd > body.length) throw Object.assign(new Error("Batch table is truncated"), { status: 400 });
  const items = [];
  const seen = new Set();
  let payloadOffset = tableEnd;
  for (let itemIndex = 0; itemIndex < count; itemIndex += 1) {
    const offset = 8 + itemIndex * 16;
    const segmentIndex = body.readUInt32BE(offset);
    const segmentSamples = body.readUInt32BE(offset + 4);
    const frameLength = body.readUInt32BE(offset + 8);
    const byteLength = body.readUInt32BE(offset + 12);
    if (segmentIndex >= session.segmentCount) throw Object.assign(new Error(`Segment index ${segmentIndex} is out of range`), { status: 422 });
    if (seen.has(segmentIndex)) throw Object.assign(new Error(`Segment index ${segmentIndex} is duplicated`), { status: 422 });
    seen.add(segmentIndex);
    const expectedBytes = Number(session.meta.channels) * Number(session.meta.segment_samples) * 4;
    if (byteLength !== expectedBytes) throw Object.assign(new Error(`Segment ${segmentIndex} has ${byteLength} bytes; expected ${expectedBytes}`), { status: 422 });
    const end = payloadOffset + byteLength;
    if (end > body.length) throw Object.assign(new Error(`Segment ${segmentIndex} payload is truncated`), { status: 400 });
    const source = body.subarray(payloadOffset, end);
    const audioBuffer = source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
    items.push({ segmentIndex, segmentSamples, frameLength, audioBuffer });
    payloadOffset = end;
  }
  if (payloadOffset !== body.length) throw Object.assign(new Error("Batch has trailing bytes"), { status: 400 });
  return items;
}

async function createSession(request, response) {
  const input = await readJson(request);
  const profile = String(input.profile || "");
  const admission = input.admission === "queue" ? "queue" : "try";
  const segmentCount = Number(input.segmentCount);
  if (!profiles.includes(profile)) return errorJson(response, 422, "unsupported_profile", "Requested EnCodec profile is not available", { profiles });
  if (!Number.isInteger(segmentCount) || segmentCount <= 0 || segmentCount > maxSegments) {
    return errorJson(response, 422, "invalid_segment_count", `segmentCount must be between 1 and ${maxSegments}`);
  }
  const sessionOwner = owner(request);
  if (activeSession && admission === "try") {
    return errorJson(response, 409, "remote_busy", "Remote encoder is busy; use local processing", {
      activeRemainingSegments: Math.max(0, activeSession.segmentCount - activeSession.results.size),
      queueDepth: queue.length,
      retryAfterMs: 1000,
    });
  }
  if (activeSession && queue.length >= maxQueue) return errorJson(response, 429, "queue_full", "Remote encode queue is full");

  const profileDescriptor = await descriptorForProfile(profile);
  const now = Date.now();
  const session = {
    id: id(),
    token: id(24),
    owner: sessionOwner,
    profile,
    segmentCount,
    state: activeSession ? "queued" : "active",
    createdAt: now,
    activatedAt: activeSession ? 0 : now,
    expiresAt: now + (activeSession ? queuedLeaseMs : activeLeaseMs),
    descriptor: profileDescriptor.descriptor,
    cacheKeyPrefix: profileDescriptor.cacheKeyPrefix,
    meta: profileDescriptor.meta,
    results: new Map(),
    inFlight: false,
  };
  sessions.set(session.id, session);
  if (session.state === "active") activeSession = session;
  else queue.push(session);
  json(response, session.state === "active" ? 201 : 202, { ok: true, session: publicSession(session, { includeToken: true }) });
}

async function encodeBatch(request, response, session) {
  if (!verifySessionAccess(request, response, session)) return;
  if (session.state !== "active" || activeSession !== session) {
    return errorJson(response, 409, "session_not_active", "Encode session is not active", { session: publicSession(session) });
  }
  if (session.inFlight) return errorJson(response, 409, "batch_in_flight", "Another batch is already running for this session");
  const body = await readBody(request, maxBatchBytes);
  const items = parseBatch(body, session);
  const streamResults = wantsBatchStream(request);
  session.inFlight = true;
  session.expiresAt = Date.now() + activeLeaseMs;
  const results = [];
  const errors = [];
  let streamHeartbeat = null;
  const publicResult = (result) => ({
    segmentIndex: result.segmentIndex,
    ecdcChunkBase64: Buffer.from(result.bytes).toString("base64"),
    sha256: createHash("sha256").update(result.bytes).digest("hex"),
    cached: result.cached,
  });
  try {
    if (streamResults) {
      startBatchStream(response, session);
      streamHeartbeat = setInterval(() => writeBatchStreamHeartbeat(response), heartbeatIntervalMs);
      streamHeartbeat.unref?.();
    }
    await Promise.all(items.map(async (item) => {
      const cached = session.results.get(item.segmentIndex);
      if (cached) {
        const result = { segmentIndex: item.segmentIndex, bytes: cached, cached: true };
        results.push(result);
        if (streamResults) writeBatchStreamEvent(response, "result", publicResult(result));
        return;
      }
      try {
        const encoded = new Uint8Array(await pool.encode({
          profile: session.profile,
          segmentIndex: item.segmentIndex,
          segmentSamples: item.segmentSamples,
          frameLength: item.frameLength,
          audioBuffer: item.audioBuffer,
        }));
        if (!encoded.byteLength) throw new Error("Encoder returned an empty ECDC chunk");
        session.results.set(item.segmentIndex, encoded);
        const result = { segmentIndex: item.segmentIndex, bytes: encoded, cached: false };
        results.push(result);
        if (streamResults) writeBatchStreamEvent(response, "result", publicResult(result));
      } catch (error) {
        const resultError = { segmentIndex: item.segmentIndex, error: "encode_failed", message: error?.message || String(error) };
        errors.push(resultError);
        if (streamResults) writeBatchStreamEvent(response, "error", resultError);
      }
    }));
  } finally {
    if (streamHeartbeat) clearInterval(streamHeartbeat);
    session.inFlight = false;
    session.expiresAt = Date.now() + activeLeaseMs;
  }
  const responseResults = results.map(publicResult);
  if (session.results.size >= session.segmentCount) finishSession(session, "complete");
  const responsePayload = {
    ok: errors.length === 0,
    results: responseResults,
    errors,
    session: publicSession(session),
  };
  if (streamResults) {
    writeBatchStreamEvent(response, "complete", {
      ok: responsePayload.ok,
      errors,
      session: responsePayload.session,
    });
    response.end();
    return;
  }
  json(response, errors.length ? 207 : 200, responsePayload);
}

async function route(request, response) {
  applyCors(request, response);
  const url = new URL(request.url, `http://${request.headers.host || "localhost"}`);
  if (request.method === "OPTIONS") {
    response.writeHead(204, { "Cache-Control": "no-store" });
    response.end();
    return;
  }
  if (request.method === "GET" && url.pathname === "/healthz") {
    return json(response, 200, {
      ok: true,
      service: "yl-encodec",
      active: Boolean(activeSession),
      queueDepth: queue.length,
      heartbeat: {
        transport: "poll",
        intervalMs: heartbeatIntervalMs,
        queuedLeaseMs,
        activeLeaseMs,
      },
    });
  }
  if (request.method === "GET" && url.pathname === "/readyz") {
    const ready = warmState === "ready";
    return json(response, ready ? 200 : 503, {
      ok: ready,
      state: warmState,
      workers: workerCount,
      defaultProfile,
      heartbeat: {
        transport: "poll",
        intervalMs: heartbeatIntervalMs,
        queuedLeaseMs,
        activeLeaseMs,
      },
    });
  }
  if (!authorized(request)) return errorJson(response, 401, "unauthorized", "Origin authorization is required");
  if (request.method === "POST" && url.pathname === "/v1/sessions") return createSession(request, response);

  const match = url.pathname.match(/^\/v1\/sessions\/([A-Za-z0-9_-]+)(?:\/(batches))?$/);
  if (!match) return errorJson(response, 404, "not_found", "Route was not found");
  const session = sessions.get(match[1]);
  if (match[2] === "batches" && request.method === "POST") return encodeBatch(request, response, session);
  if (!verifySessionAccess(request, response, session)) return;
  if (request.method === "GET") {
    if (session.state === "active") session.expiresAt = Date.now() + activeLeaseMs;
    if (session.state === "queued") session.expiresAt = Date.now() + queuedLeaseMs;
    return json(response, 200, { ok: true, session: publicSession(session) });
  }
  if (request.method === "DELETE") {
    if (!["complete", "cancelled", "expired"].includes(session.state)) finishSession(session, "cancelled");
    return json(response, 200, { ok: true, session: publicSession(session) });
  }
  return errorJson(response, 405, "method_not_allowed", "Method is not allowed");
}

let warmState = "warming";
const warmPromise = pool.warm(defaultProfile).then(() => {
  warmState = "ready";
}).catch((error) => {
  warmState = "error";
  console.error("Encoder pool warm-up failed", error);
});

const server = createServer((request, response) => {
  route(request, response).catch((error) => {
    console.error("Request failed", { method: request.method, url: request.url, error });
    if (!response.headersSent) errorJson(response, error?.status || 500, "request_failed", error?.message || "Request failed");
    else response.destroy(error);
  });
});

const sweep = setInterval(() => {
  const now = Date.now();
  for (const session of sessions.values()) {
    if (["active", "queued"].includes(session.state) && session.expiresAt <= now) finishSession(session, "expired");
    if (["complete", "cancelled", "expired"].includes(session.state) && session.expiresAt <= now) sessions.delete(session.id);
  }
}, 5000);
sweep.unref();

server.listen(port, host, () => {
  console.log(JSON.stringify({ event: "listening", host, port, workers: workerCount, defaultProfile, assetRoot }));
});

async function shutdown(signal) {
  console.log(JSON.stringify({ event: "shutdown", signal }));
  clearInterval(sweep);
  await new Promise((resolve) => server.close(resolve));
  await pool.close();
  process.exit(0);
}

process.once("SIGTERM", () => void shutdown("SIGTERM"));
process.once("SIGINT", () => void shutdown("SIGINT"));
await warmPromise;
