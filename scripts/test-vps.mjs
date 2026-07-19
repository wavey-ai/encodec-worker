import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

const baseUrl = process.env.ENCODEC_TEST_URL || "http://127.0.0.1:8787";
const originToken = process.env.ENCODEC_ORIGIN_TOKEN ?? "test-origin-token";
const authorizationHeaders = originToken ? { Authorization: `Bearer ${originToken}` } : {};
const profile = process.env.ENCODEC_TEST_PROFILE || "encodec_48khz_12kbps_1333ms";
const segmentCount = Math.max(1, Math.min(16, Number(process.env.ENCODEC_TEST_SEGMENTS) || 2));
const owner = `vps-test-${Date.now()}`;
const root = path.resolve(import.meta.dirname, "..");
const runtimeRoot = process.env.ENCODEC_ASSET_ROOT || path.join(root, "dist-vps", "runtime");
const meta = JSON.parse(await readFile(path.join(runtimeRoot, "bundles", profile, "bundle.json"), "utf8"));

async function request(pathname, init = {}) {
  const response = await fetch(`${baseUrl}${pathname}`, {
    ...init,
    headers: {
      ...authorizationHeaders,
      "X-Encodec-Owner": owner,
      ...(init.headers || {}),
    },
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`${response.status} ${body.error || "request_failed"}: ${body.message || ""}`);
  return body;
}

async function requestStream(pathname, init = {}) {
  const response = await fetch(`${baseUrl}${pathname}`, {
    ...init,
    headers: {
      ...authorizationHeaders,
      "X-Encodec-Owner": owner,
      Accept: "text/event-stream",
      ...(init.headers || {}),
    },
  });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(`${response.status} ${body.error || "request_failed"}: ${body.message || ""}`);
  }
  if (!String(response.headers.get("Content-Type") || "").includes("text/event-stream")) {
    throw new Error("Encode endpoint did not return an event stream");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const events = [];
  let buffer = "";
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      if (!block.trim() || block.trimStart().startsWith(":")) continue;
      const event = block.match(/^event:\s*(.+)$/m)?.[1]?.trim() || "message";
      const data = block.match(/^data:\s*(.+)$/m)?.[1] || "null";
      events.push({ event, payload: JSON.parse(data) });
    }
  }
  return events;
}

function tone(segmentIndex) {
  const frames = Number(meta.segment_samples);
  const channels = Number(meta.channels);
  const audio = new Float32Array(channels * frames);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let frame = 0; frame < frames; frame += 1) {
      audio[channel * frames + frame] = Math.sin(2 * Math.PI * (220 + segmentIndex * 55) * frame / Number(meta.sample_rate)) * 0.05;
    }
  }
  return audio;
}

function frameBatch(items) {
  const tableBytes = 8 + items.length * 16;
  const payloadBytes = items.reduce((total, item) => total + item.audio.byteLength, 0);
  const output = new Uint8Array(tableBytes + payloadBytes);
  output.set(new TextEncoder().encode("YECB"), 0);
  const view = new DataView(output.buffer);
  view.setUint16(4, 1, false);
  view.setUint16(6, items.length, false);
  let payloadOffset = tableBytes;
  items.forEach((item, itemIndex) => {
    const offset = 8 + itemIndex * 16;
    view.setUint32(offset, item.segmentIndex, false);
    view.setUint32(offset + 4, item.segmentSamples, false);
    view.setUint32(offset + 8, Number(meta.frame_length), false);
    view.setUint32(offset + 12, item.audio.byteLength, false);
    output.set(new Uint8Array(item.audio.buffer, item.audio.byteOffset, item.audio.byteLength), payloadOffset);
    payloadOffset += item.audio.byteLength;
  });
  return output;
}

function decodeBase64(value) {
  return Buffer.from(value, "base64");
}

const created = await request("/v1/sessions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ profile, segmentCount, admission: "try" }),
});
const session = created.session;
if (session.state !== "active") throw new Error(`Expected active session, got ${session.state}`);

const submittedOrder = Array.from({ length: segmentCount }, (_, index) => segmentCount - index - 1);
const expectedIndexes = new Set(submittedOrder);
const body = frameBatch(submittedOrder.map((segmentIndex) => ({
  segmentIndex,
  segmentSamples: Number(meta.segment_stride),
  audio: tone(segmentIndex),
})));
const events = await requestStream(`/v1/sessions/${session.id}/batches`, {
  method: "POST",
  headers: {
    "Content-Type": "application/vnd.yl.encodec-batch",
    "X-Encodec-Session-Token": session.sessionToken,
  },
  body,
});
const encoded = {
  results: events.filter((event) => event.event === "result").map((event) => event.payload),
  ...(events.findLast((event) => event.event === "complete")?.payload || {}),
};
if (encoded.errors?.length) throw new Error(`Encode errors: ${JSON.stringify(encoded.errors)}`);

const reconciled = new Map();
for (const result of encoded.results || []) {
  if (!expectedIndexes.has(result.segmentIndex)) throw new Error(`Unexpected result index ${result.segmentIndex}`);
  if (reconciled.has(result.segmentIndex)) throw new Error(`Duplicate result index ${result.segmentIndex}`);
  const bytes = decodeBase64(result.ecdcChunkBase64);
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== result.sha256) throw new Error(`Checksum mismatch for segment ${result.segmentIndex}`);
  if (!bytes.byteLength) throw new Error(`Empty result for segment ${result.segmentIndex}`);
  reconciled.set(result.segmentIndex, bytes);
}
if (reconciled.size !== segmentCount) throw new Error(`Expected ${segmentCount} reconciled results, got ${reconciled.size}`);
const reconciledOrder = Array.from({ length: segmentCount }, (_value, segmentIndex) => segmentIndex);
const ordered = reconciledOrder.map((segmentIndex) => reconciled.get(segmentIndex).byteLength);
if (encoded.session.state !== "complete") throw new Error(`Expected complete session, got ${encoded.session.state}`);

console.log(JSON.stringify({
  ok: true,
  profile,
  submittedOrder,
  responseOrder: encoded.results.map((result) => result.segmentIndex),
  reconciledOrder,
  encodedBytes: ordered,
}, null, 2));
