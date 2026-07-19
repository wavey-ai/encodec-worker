import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parentPort, workerData } from "node:worker_threads";

const assetRoot = path.resolve(workerData.assetRoot);
const ortModuleUrl = pathToFileURL(path.join(assetRoot, "onnxruntime", "ort.wasm.min.mjs")).href;
const ortWasmBaseUrl = pathToFileURL(path.join(assetRoot, "onnxruntime") + path.sep).href;
const encodecModuleUrl = pathToFileURL(path.join(assetRoot, "encodec", "pkg", "encodec_rs.js")).href;
const encodecWasmPath = path.join(assetRoot, "encodec", "pkg", "encodec_rs_bg.wasm");

let ortPromise = null;
let encodecPromise = null;
let loadedProfile = null;

function allowedProfile(name) {
  return workerData.profiles.includes(String(name || ""));
}

async function ensureOrt() {
  if (!ortPromise) {
    ortPromise = import(ortModuleUrl).then((ort) => {
      ort.env.wasm.wasmPaths = ortWasmBaseUrl;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      return ort;
    });
  }
  return ortPromise;
}

async function ensureEncodec() {
  if (!encodecPromise) {
    encodecPromise = Promise.all([
      import(encodecModuleUrl),
      readFile(encodecWasmPath),
    ]).then(([module, wasmBytes]) => {
      module.initSync({ module: wasmBytes });
      module.initPanicHook?.();
      return module;
    });
  }
  return encodecPromise;
}

async function disposeLoadedProfile() {
  if (!loadedProfile?.session) return;
  try {
    await loadedProfile.session.release?.();
  } catch (_error) {
  }
  try {
    await loadedProfile.session.dispose?.();
  } catch (_error) {
  }
  loadedProfile = null;
}

async function ensureProfile(profileName) {
  if (!allowedProfile(profileName)) {
    throw new Error(`Unsupported EnCodec profile: ${profileName}`);
  }
  if (loadedProfile?.name === profileName) return loadedProfile;

  await disposeLoadedProfile();
  const profileRoot = path.join(assetRoot, "bundles", profileName);
  const [ort, encodec, bundleJson, modelBytes, lmWeights] = await Promise.all([
    ensureOrt(),
    ensureEncodec(),
    readFile(path.join(profileRoot, "bundle.json"), "utf8"),
    readFile(path.join(profileRoot, "encode_frame.onnx")),
    readFile(path.join(profileRoot, "lm_weights_q8.bin")),
  ]);
  const meta = JSON.parse(bundleJson);
  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  loadedProfile = {
    name: profileName,
    ort,
    encodec,
    bundleJson,
    meta,
    lmWeights: new Uint8Array(lmWeights),
    session,
  };
  return loadedProfile;
}

function findEncodeOutputs(outputs) {
  const tensors = Object.values(outputs || {});
  const codesTensor = tensors.find((tensor) => tensor.type === "int64");
  const scaleTensor = tensors.find(
    (tensor) => tensor.type === "float32" && tensor.dims.length === 2,
  );
  if (!codesTensor || !scaleTensor) {
    const summary = Object.fromEntries(Object.entries(outputs || {}).map(([name, tensor]) => [name, {
      type: tensor.type,
      dims: tensor.dims,
      length: tensor.data?.length || 0,
    }]));
    throw new Error(`Unexpected encoder outputs: ${JSON.stringify(summary)}`);
  }
  return { codesTensor, scaleTensor };
}

function toU16Codes(codes, segmentIndex) {
  const output = new Uint16Array(codes.length);
  for (let index = 0; index < codes.length; index += 1) {
    const value = Number(codes[index]);
    if (!Number.isInteger(value) || value < 0 || value > 65535) {
      throw new Error(`Invalid code at segment ${segmentIndex}, code ${index}`);
    }
    output[index] = value;
  }
  return output;
}

function disposeTensorMap(tensors) {
  for (const tensor of Object.values(tensors || {})) {
    try {
      tensor?.dispose?.();
    } catch (_error) {
    }
  }
}

async function encodeSegment({ profile, segmentIndex, segmentSamples, frameLength, audioBuffer }) {
  const runtime = await ensureProfile(profile);
  const meta = runtime.meta;
  const audio = new Float32Array(audioBuffer);
  const expectedSamples = Number(meta.channels) * Number(meta.segment_samples);
  if (audio.length !== expectedSamples) {
    throw new Error(`Segment ${segmentIndex} has ${audio.length} float samples; expected ${expectedSamples}`);
  }
  if (!Number.isInteger(segmentSamples) || segmentSamples <= 0 || segmentSamples > Number(meta.segment_stride)) {
    throw new Error(`Segment ${segmentIndex} has invalid owned sample count ${segmentSamples}`);
  }
  if (frameLength !== Number(meta.frame_length)) {
    throw new Error(`Segment ${segmentIndex} frame length ${frameLength} does not match ${meta.frame_length}`);
  }

  let feeds = null;
  let outputs = null;
  try {
    feeds = {
      [runtime.session.inputNames[0]]: new runtime.ort.Tensor("float32", audio, [
        1,
        Number(meta.channels),
        Number(meta.segment_samples),
      ]),
    };
    outputs = await runtime.session.run(feeds);
    const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
    const expectedCodes = Number(meta.num_codebooks) * Number(meta.frame_length);
    if (codesTensor.data.length !== expectedCodes) {
      throw new Error(`Segment ${segmentIndex} produced ${codesTensor.data.length} codes; expected ${expectedCodes}`);
    }
    const rawScale = Number(scaleTensor.data?.[0] ?? 1);
    const scale = Number.isFinite(rawScale) ? rawScale : 1;
    const chunk = runtime.encodec.lmEcdcChunkFromFrame(
      runtime.bundleJson,
      runtime.lmWeights,
      scale,
      toU16Codes(codesTensor.data, segmentIndex),
      Number(meta.frame_length),
    );
    const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  } finally {
    disposeTensorMap(feeds);
    disposeTensorMap(outputs);
  }
}

async function handleTask(task) {
  if (task.type === "warm") {
    const runtime = await ensureProfile(task.profile);
    return {
      profile: runtime.name,
      segmentSamples: Number(runtime.meta.segment_samples),
      frameLength: Number(runtime.meta.frame_length),
    };
  }
  if (task.type === "encode") {
    return encodeSegment(task);
  }
  throw new Error(`Unknown encoder worker task: ${task.type}`);
}

parentPort.on("message", async (task) => {
  try {
    const result = await handleTask(task);
    if (result instanceof ArrayBuffer) {
      parentPort.postMessage({ id: task.id, ok: true, result }, [result]);
    } else {
      parentPort.postMessage({ id: task.id, ok: true, result });
    }
  } catch (error) {
    parentPort.postMessage({
      id: task.id,
      ok: false,
      error: {
        name: error?.name || "Error",
        message: error?.message || String(error),
        stack: error?.stack || "",
      },
    });
  }
});

