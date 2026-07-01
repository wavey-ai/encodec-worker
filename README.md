** ON HOLD TOO MUCH RAM **

# encodec-worker

Rust-first Cloudflare Worker proof of concept for one fixed segment:

`SoundKit length-prefixed Opus packets → pure-Rust libopus-rs decode → ORT Web CPU inference → encodec-rs ECDC assembly`

All request handling, packet parsing, Opus decoding, validation, hashing, runtime coordination, and ECDC construction are in `src/lib.rs`. There is no hand-written JavaScript service file. `worker-build` generates the normal Cloudflare JavaScript/WASM shim, and the Rust module imports `onnxruntime-web/wasm` for CPU ONNX inference.

## Place it in the repository

Adjust the two workspace paths in `Cargo.toml if necessary:

- `encodec-rs = { path = "../../encodec-rs" }`
- `libopus-rs = { path = "../../soundkit/libopus-rs" }`

Replace the three placeholder files under:

`assets/encodec_48khz_12kbps_1800ms/`

with the real fixed bundle:

- `bundle.json`
- `encode_frame.onnx`
- `lm_weights_q8.bin`

## Request

```bash
curl -X POST http://localhost:8787/encode/ecdc \
  -H 'Content-Type: application/vnd.soundkit.opus-packets' \
  -H 'X-Encodec-Profile: encodec_48khz_12kbps_1800ms' \
  -H 'X-Encodec-Expected-Samples: 86400' \
  -H 'X-Encodec-Sample-Rate: 48000' \
  -H 'X-Encodec-Channels: 1' \
  --data-binary '@segment.opus-packets' \
  --output segment.ecdc
```

The body is repeated `u32be packet_length` followed by raw Opus packet bytes.

## Commands

```bash
npm install
npm run dev
npm run build
npm run deploy
```

## Deliberate limits

- one 1800 ms mono segment
- 48000 Hz
- 86400 decoded samples
- one fixed model bundle
- no storage, queues, authentication, fallback, or track assembly
- natural Cloudflare isolate reuse only

## Important compile check

The current `encodec-rs` bundle exposes `lm_ecdc_fixed_header_for_weights` and `lm_ecdc_chunk_from_frame` from `encodec_rs::format`. If the real crate re-exports them at its root instead, update that single import line.

The ORT result selection currently discovers integer codes and the scalar float scale by tensor type. Replace this with the exact output names used by the browser wrapper once confirmed.
# encodec-worker
