import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const workspace = path.resolve(root, "..");
const source = path.join(workspace, "encodec-rs", "dist", "wasm-fixed-bundles");
const ortSource = path.join(workspace, "encodec-rs", "browser-smoke", "node_modules", "onnxruntime-web", "dist");
const output = path.join(root, "dist-vps");
const runtime = path.join(output, "runtime");
const profiles = [
  "encodec_48khz_6kbps_1333ms",
  "encodec_48khz_6kbps_1800ms",
  "encodec_48khz_12kbps_1333ms",
  "encodec_48khz_12kbps_1800ms",
];

await rm(output, { recursive: true, force: true });
await mkdir(path.join(output, "server"), { recursive: true });
await mkdir(path.join(runtime, "encodec", "pkg"), { recursive: true });
await mkdir(path.join(runtime, "onnxruntime"), { recursive: true });
await mkdir(path.join(runtime, "bundles"), { recursive: true });

await cp(path.join(root, "server", "main.mjs"), path.join(output, "server", "main.mjs"));
await cp(path.join(root, "server", "encode-worker.mjs"), path.join(output, "server", "encode-worker.mjs"));
for (const file of ["encodec_rs.js", "encodec_rs_bg.wasm"]) {
  await cp(path.join(source, "pkg", file), path.join(runtime, "encodec", "pkg", file));
}
await writeFile(path.join(runtime, "encodec", "pkg", "package.json"), "{\"type\":\"module\"}\n");
for (const file of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await cp(path.join(ortSource, file), path.join(runtime, "onnxruntime", file));
}
for (const profile of profiles) {
  const target = path.join(runtime, "bundles", profile);
  await mkdir(target, { recursive: true });
  for (const file of ["bundle.json", "encode_frame.onnx", "lm_weights_q8.bin"]) {
    await cp(path.join(source, "bundles", profile, file), path.join(target, file));
  }
}
await writeFile(path.join(output, "package.json"), `${JSON.stringify({
  name: "yl-encodec-vps",
  private: true,
  type: "module",
  engines: { node: ">=22" },
  scripts: { start: "node server/main.mjs" },
}, null, 2)}\n`);

const packageJson = JSON.parse(await readFile(path.join(output, "package.json"), "utf8"));
console.log(JSON.stringify({ output, profiles, package: packageJson.name }, null, 2));

