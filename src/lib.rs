use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use encodec_rs::wasm::{lm_ecdc_chunk_from_frame, lm_ecdc_fixed_header_for_weights};
use js_sys::{
    Array, BigInt64Array, Float32Array, Int32Array, Object, Promise, Reflect, Uint8Array,
};
use libopus_rs::Decoder as OpusDecoder;
use serde::Serialize;
use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use worker::{
    console_error, event, Context, Env, Headers, Method, Request, Response, Result, Router,
};

const PROFILE: &str = "encodec_48khz_12kbps_1800ms";
const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: usize = 1;
const EXPECTED_FRAMES: usize = 86_400;
const MAX_BODY_BYTES: usize = 512 * 1024;
const BITSTREAM_VERSION: u8 = 2;
const BUNDLE_JSON: &str = include_str!("../assets/encodec_48khz_12kbps_1800ms/bundle.json");
const MODEL_BYTES: &[u8] =
    include_bytes!("../assets/encodec_48khz_12kbps_1800ms/encode_frame.onnx");
const LM_WEIGHTS: &[u8] = include_bytes!("../assets/encodec_48khz_12kbps_1800ms/lm_weights_q8.bin");

thread_local! {
    static RUNTIME: RefCell<Option<Rc<Runtime>>> = const { RefCell::new(None) };
    static RUNTIME_PROMISE: RefCell<Option<Promise>> = const { RefCell::new(None) };
}

#[wasm_bindgen(module = "onnxruntime-web/wasm")]
extern "C" {
    #[wasm_bindgen(js_name = InferenceSession)]
    type InferenceSession;

    #[wasm_bindgen(static_method_of = InferenceSession, js_name = create)]
    fn create_session(model: Uint8Array, options: &JsValue) -> Promise;

    #[wasm_bindgen(method, getter, js_name = inputNames)]
    fn input_names(this: &InferenceSession) -> Array;

    #[wasm_bindgen(method)]
    fn run(this: &InferenceSession, feeds: &JsValue) -> Promise;

    #[wasm_bindgen(js_name = Tensor)]
    type Tensor;

    #[wasm_bindgen(constructor)]
    fn new_tensor(data_type: &str, data: Float32Array, dims: Array) -> Tensor;
}

struct Runtime {
    session: InferenceSession,
    instance_id: String,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
    message: &'a str,
    request_id: &'a str,
}

#[derive(Serialize)]
struct HealthBody<'a> {
    ok: bool,
    service: &'a str,
}

#[derive(Serialize)]
struct ReadyBody<'a> {
    ok: bool,
    profile: &'a str,
    runtime_instance: &'a str,
    runtime_reused: bool,
}

#[event(fetch)]
pub async fn fetch(request: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    Router::new()
        .get("/health", |_, _| {
            Response::from_json(&HealthBody {
                ok: true,
                service: "encodec-worker",
            })
        })
        .get_async("/ready", |request, ctx| async move {
            ready(request, ctx.env).await
        })
        .options("/encode/ecdc", |request, ctx| preflight(&request, &ctx.env))
        .post_async("/encode/ecdc", |request, ctx| async move {
            encode(request, ctx.env).await
        })
        .run(request, env)
        .await
}

async fn ready(request: Request, env: Env) -> Result<Response> {
    let request_id = random_id();
    let cors = match cors_headers(&request, &env)? {
        Some(headers) => headers,
        None => {
            return error_response(
                "origin_not_allowed",
                "The request origin is not allowed.",
                403,
                &request_id,
                None,
            )
        }
    };
    let reused = runtime_exists();
    match get_runtime().await {
        Ok(runtime) => {
            let mut response = Response::from_json(&ReadyBody {
                ok: true,
                profile: PROFILE,
                runtime_instance: &runtime.instance_id,
                runtime_reused: reused,
            })?;
            apply_headers(response.headers_mut(), &cors)?;
            Ok(response)
        }
        Err(error) => {
            console_error!("runtime initialisation failed: {:?}", error);
            error_response(
                "runtime_initialisation_failed",
                "The encoder runtime could not be initialised.",
                503,
                &request_id,
                Some(cors),
            )
        }
    }
}

fn preflight(request: &Request, env: &Env) -> Result<Response> {
    let request_id = random_id();
    let cors = match cors_headers(request, env)? {
        Some(headers) => headers,
        None => {
            return error_response(
                "origin_not_allowed",
                "The request origin is not allowed.",
                403,
                &request_id,
                None,
            )
        }
    };
    let mut response = Response::empty()?.with_status(204);
    apply_headers(response.headers_mut(), &cors)?;
    Ok(response)
}

async fn encode(mut request: Request, env: Env) -> Result<Response> {
    let request_id = random_id();
    let cors = match cors_headers(&request, &env)? {
        Some(headers) => headers,
        None => {
            return error_response(
                "origin_not_allowed",
                "The request origin is not allowed.",
                403,
                &request_id,
                None,
            )
        }
    };

    if request.method() != Method::Post {
        return error_response(
            "method_not_allowed",
            "Use POST for this endpoint.",
            405,
            &request_id,
            Some(cors),
        );
    }

    let content_type = request.headers().get("Content-Type")?.unwrap_or_default();
    let content_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if content_type != "application/octet-stream"
        && content_type != "application/vnd.soundkit.opus-packets"
    {
        return error_response(
            "unsupported_content_type",
            "Expected a SoundKit Opus packet bundle.",
            415,
            &request_id,
            Some(cors),
        );
    }

    let profile = request
        .headers()
        .get("X-Encodec-Profile")?
        .unwrap_or_else(|| PROFILE.to_string());
    if profile != PROFILE {
        return error_response(
            "unsupported_profile",
            "Only encodec_48khz_12kbps_1800ms is supported.",
            422,
            &request_id,
            Some(cors),
        );
    }

    for (name, expected) in [
        ("X-Encodec-Expected-Samples", EXPECTED_FRAMES),
        ("X-Encodec-Sample-Rate", SAMPLE_RATE as usize),
        ("X-Encodec-Channels", CHANNELS),
    ] {
        if let Some(raw) = request.headers().get(name)? {
            if raw.parse::<usize>().ok() != Some(expected) {
                return error_response(
                    "invalid_profile_parameters",
                    "The profile parameters do not match the fixed encoder bundle.",
                    422,
                    &request_id,
                    Some(cors),
                );
            }
        }
    }

    if let Some(raw) = request.headers().get("Content-Length")? {
        if raw.parse::<usize>().unwrap_or(usize::MAX) > MAX_BODY_BYTES {
            return error_response(
                "body_too_large",
                "The request body is too large.",
                413,
                &request_id,
                Some(cors),
            );
        }
    }

    let body = request.bytes().await?;
    if body.is_empty() {
        return error_response(
            "body_missing",
            "The request body is empty.",
            400,
            &request_id,
            Some(cors),
        );
    }
    if body.len() > MAX_BODY_BYTES {
        return error_response(
            "body_too_large",
            "The request body is too large.",
            413,
            &request_id,
            Some(cors),
        );
    }

    let reused = runtime_exists();
    let runtime = match get_runtime().await {
        Ok(runtime) => runtime,
        Err(error) => {
            console_error!("request {} runtime error: {:?}", request_id, error);
            return error_response(
                "runtime_initialisation_failed",
                "The encoder runtime could not be initialised.",
                500,
                &request_id,
                Some(cors),
            );
        }
    };

    let started = js_sys::Date::now();
    let pcm = match decode_packet_bundle(&body) {
        Ok(pcm) => pcm,
        Err(error) => {
            console_error!("request {} opus error: {}", request_id, error);
            return error_response(
                "invalid_opus",
                "The submitted SoundKit Opus packet bundle could not be decoded.",
                422,
                &request_id,
                Some(cors),
            );
        }
    };
    let decoded_at = js_sys::Date::now();

    let (codes, scale, frame_length) = match run_inference(&runtime.session, &pcm).await {
        Ok(output) => output,
        Err(error) => {
            console_error!("request {} inference error: {:?}", request_id, error);
            return error_response(
                "ort_inference_failed",
                "The EnCodec model failed to encode the decoded PCM.",
                500,
                &request_id,
                Some(cors),
            );
        }
    };
    let inferred_at = js_sys::Date::now();

    let mut ecdc = match lm_ecdc_fixed_header_for_weights(
        BUNDLE_JSON,
        EXPECTED_FRAMES,
        BITSTREAM_VERSION,
        LM_WEIGHTS,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            console_error!("request {} header error: {:?}", request_id, error);
            return error_response(
                "ecdc_encoding_failed",
                "The ECDC header could not be produced.",
                500,
                &request_id,
                Some(cors),
            );
        }
    };
    let chunk = match lm_ecdc_chunk_from_frame(BUNDLE_JSON, LM_WEIGHTS, scale, &codes, frame_length)
    {
        Ok(bytes) => bytes,
        Err(error) => {
            console_error!("request {} chunk error: {:?}", request_id, error);
            return error_response(
                "ecdc_encoding_failed",
                "The ECDC payload could not be produced.",
                500,
                &request_id,
                Some(cors),
            );
        }
    };
    ecdc.extend_from_slice(&chunk);
    let completed_at = js_sys::Date::now();
    let digest = hex::encode(Sha256::digest(&ecdc));

    let mut headers = Headers::new();
    headers.set("Content-Type", "application/vnd.encodec.ecdc")?;
    headers.set("Cache-Control", "no-store")?;
    headers.set("X-Encodec-Request-Id", &request_id)?;
    headers.set("X-Encodec-Profile", PROFILE)?;
    headers.set("X-Encodec-Decoded-Samples", &EXPECTED_FRAMES.to_string())?;
    headers.set(
        "X-Encodec-Decode-Elapsed-Ms",
        &format!("{:.3}", decoded_at - started),
    )?;
    headers.set(
        "X-Encodec-Ort-Elapsed-Ms",
        &format!("{:.3}", inferred_at - decoded_at),
    )?;
    headers.set(
        "X-Encodec-Ecdc-Elapsed-Ms",
        &format!("{:.3}", completed_at - inferred_at),
    )?;
    headers.set(
        "X-Encodec-Total-Elapsed-Ms",
        &format!("{:.3}", completed_at - started),
    )?;
    headers.set("X-Encodec-Runtime-Reused", if reused { "1" } else { "0" })?;
    headers.set("X-Encodec-Runtime-Instance", &runtime.instance_id)?;
    headers.set("X-Encodec-Output-Bytes", &ecdc.len().to_string())?;
    headers.set("X-Encodec-Output-Sha256", &digest)?;
    apply_headers(&mut headers, &cors)?;

    Ok(Response::from_bytes(ecdc)?.with_headers(headers))
}

async fn get_runtime() -> std::result::Result<Rc<Runtime>, JsValue> {
    if let Some(runtime) = RUNTIME.with(|cell| cell.borrow().clone()) {
        return Ok(runtime);
    }

    let promise = RUNTIME_PROMISE.with(|cell| {
        if let Some(promise) = cell.borrow().clone() {
            promise
        } else {
            let future = async {
                let model = Uint8Array::from(MODEL_BYTES);
                let options = Object::new();
                let providers = Array::new();
                providers.push(&JsValue::from_str("wasm"));
                Reflect::set(
                    &options,
                    &JsValue::from_str("executionProviders"),
                    &providers,
                )?;
                Reflect::set(
                    &options,
                    &JsValue::from_str("graphOptimizationLevel"),
                    &JsValue::from_str("all"),
                )?;
                let session =
                    JsFuture::from(InferenceSession::create_session(model, &options.into()))
                        .await?;
                let session: InferenceSession = session.unchecked_into();
                let runtime = Rc::new(Runtime {
                    session,
                    instance_id: random_id(),
                });
                RUNTIME.with(|cell| *cell.borrow_mut() = Some(runtime));
                Ok(JsValue::UNDEFINED)
            };
            let promise = wasm_bindgen_futures::future_to_promise(future);
            *cell.borrow_mut() = Some(promise.clone());
            promise
        }
    });

    if let Err(error) = JsFuture::from(promise).await {
        RUNTIME_PROMISE.with(|cell| *cell.borrow_mut() = None);
        return Err(error);
    }

    RUNTIME
        .with(|cell| cell.borrow().clone())
        .ok_or_else(|| JsValue::from_str("runtime initialisation completed without a runtime"))
}

fn runtime_exists() -> bool {
    RUNTIME.with(|cell| cell.borrow().is_some())
        || RUNTIME_PROMISE.with(|cell| cell.borrow().is_some())
}

fn decode_packet_bundle(input: &[u8]) -> std::result::Result<Vec<f32>, String> {
    let mut decoder =
        OpusDecoder::new(SAMPLE_RATE as i32, CHANNELS).map_err(|error| error.to_string())?;
    let mut cursor = 0usize;
    let mut output = Vec::with_capacity(EXPECTED_FRAMES * CHANNELS);
    let mut packet_count = 0usize;

    while cursor < input.len() {
        if input.len() - cursor < 4 {
            return Err(format!(
                "truncated packet length at packet {}",
                packet_count
            ));
        }
        let packet_len = u32::from_be_bytes(
            input[cursor..cursor + 4]
                .try_into()
                .map_err(|_| "invalid packet length")?,
        ) as usize;
        cursor += 4;
        if packet_len == 0 || input.len() - cursor < packet_len {
            return Err(format!(
                "invalid packet {} length {}",
                packet_count, packet_len
            ));
        }
        let decoded = decoder
            .decode_i16(&input[cursor..cursor + packet_len], false)
            .map_err(|error| error.to_string())?;
        cursor += packet_len;
        for sample in decoded {
            output.push(sample as f32 / 32768.0);
            if output.len() > EXPECTED_FRAMES * CHANNELS {
                return Err("decoded PCM exceeds the fixed segment length".to_string());
            }
        }
        packet_count += 1;
    }

    if packet_count == 0 {
        return Err("packet bundle contains no packets".to_string());
    }
    if output.len() != EXPECTED_FRAMES * CHANNELS {
        return Err(format!(
            "decoded {} samples, expected {}",
            output.len(),
            EXPECTED_FRAMES * CHANNELS
        ));
    }
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err("decoded PCM contains non-finite samples".to_string());
    }
    Ok(output)
}

async fn run_inference(
    session: &InferenceSession,
    pcm: &[f32],
) -> std::result::Result<(Vec<u16>, f32, usize), JsValue> {
    let input_names = session.input_names();
    let input_name = input_names
        .get(0)
        .as_string()
        .ok_or_else(|| JsValue::from_str("model has no input name"))?;
    let data = Float32Array::from(pcm);
    let dims = Array::new();
    dims.push(&JsValue::from_f64(1.0));
    dims.push(&JsValue::from_f64(CHANNELS as f64));
    dims.push(&JsValue::from_f64(EXPECTED_FRAMES as f64));
    let tensor = Tensor::new_tensor("float32", data, dims);
    let feeds = Object::new();
    Reflect::set(&feeds, &JsValue::from_str(&input_name), tensor.as_ref())?;
    let results = JsFuture::from(session.run(&feeds.into())).await?;
    let results: Object = results
        .dyn_into()
        .map_err(|_| JsValue::from_str("model returned a non-object result"))?;
    let entries = Object::entries(&results);

    let mut codes: Option<Vec<u16>> = None;
    let mut scale: Option<f32> = None;
    let mut frame_length = 0usize;

    for entry in entries.iter() {
        let pair = Array::from(&entry);
        let tensor = pair.get(1);
        let tensor_type = Reflect::get(&tensor, &JsValue::from_str("type"))?
            .as_string()
            .unwrap_or_default();
        let tensor_data = Reflect::get(&tensor, &JsValue::from_str("data"))?;
        let dims_value = Reflect::get(&tensor, &JsValue::from_str("dims"))?;
        let output_dims = Array::from(&dims_value);

        if tensor_type == "float32" {
            let values = Float32Array::new(&tensor_data);
            if values.length() == 1 {
                scale = Some(values.get_index(0));
            }
        } else if tensor_type == "int32" {
            let values = Int32Array::new(&tensor_data);
            let mut converted = Vec::with_capacity(values.length() as usize);
            for index in 0..values.length() {
                let value = values.get_index(index);
                if value < 0 || value > u16::MAX as i32 {
                    return Err(JsValue::from_str("model returned an out-of-range code"));
                }
                converted.push(value as u16);
            }
            frame_length = output_dims
                .get(output_dims.length().saturating_sub(1))
                .as_f64()
                .unwrap_or_default() as usize;
            codes = Some(converted);
        } else if tensor_type == "int64" {
            let values = BigInt64Array::new(&tensor_data);
            let mut converted = Vec::with_capacity(values.length() as usize);
            for index in 0..values.length() {
                let value = values.get_index(index);
                if value < 0 || value > u16::MAX as i64 {
                    return Err(JsValue::from_str("model returned an out-of-range code"));
                }
                converted.push(value as u16);
            }
            frame_length = output_dims
                .get(output_dims.length().saturating_sub(1))
                .as_f64()
                .unwrap_or_default() as usize;
            codes = Some(converted);
        }
    }

    let codes = codes.ok_or_else(|| JsValue::from_str("model did not return integer codes"))?;
    let scale = scale.ok_or_else(|| JsValue::from_str("model did not return a scalar scale"))?;
    if frame_length == 0 {
        return Err(JsValue::from_str("model returned an invalid frame length"));
    }
    Ok((codes, scale, frame_length))
}

fn cors_headers(request: &Request, env: &Env) -> Result<Option<Headers>> {
    let headers = Headers::new();
    let Some(origin) = request.headers().get("Origin")? else {
        return Ok(Some(headers));
    };
    let allowed = env
        .var("ALLOWED_ORIGINS")
        .map(|value| value.to_string())
        .unwrap_or_default();
    let allowed: HashSet<&str> = allowed
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    if !allowed.contains(origin.as_str()) {
        return Ok(None);
    }
    headers.set("Access-Control-Allow-Origin", &origin)?;
    headers.set("Access-Control-Allow-Methods", "POST,OPTIONS")?;
    headers.set("Access-Control-Allow-Headers", "Content-Type,X-Encodec-Profile,X-Encodec-Expected-Samples,X-Encodec-Sample-Rate,X-Encodec-Channels")?;
    headers.set("Access-Control-Expose-Headers", "X-Encodec-Profile,X-Encodec-Decoded-Samples,X-Encodec-Decode-Elapsed-Ms,X-Encodec-Ort-Elapsed-Ms,X-Encodec-Ecdc-Elapsed-Ms,X-Encodec-Total-Elapsed-Ms,X-Encodec-Runtime-Reused,X-Encodec-Runtime-Instance,X-Encodec-Output-Bytes,X-Encodec-Output-Sha256,X-Encodec-Request-Id")?;
    headers.set("Vary", "Origin")?;
    Ok(Some(headers))
}

fn error_response(
    code: &str,
    message: &str,
    status: u16,
    request_id: &str,
    headers: Option<Headers>,
) -> Result<Response> {
    let mut response = Response::from_json(&ErrorBody {
        error: code,
        message,
        request_id,
    })?
    .with_status(status);
    response.headers_mut().set("Cache-Control", "no-store")?;
    response
        .headers_mut()
        .set("X-Encodec-Request-Id", request_id)?;
    if let Some(headers) = headers {
        apply_headers(response.headers_mut(), &headers)?;
    }
    Ok(response)
}

fn apply_headers(target: &mut Headers, source: &Headers) -> Result<()> {
    for (name, value) in source.entries() {
        target.set(&name, &value)?;
    }
    Ok(())
}

fn random_id() -> String {
    let mut bytes = [0u8; 8];
    getrandom::getrandom(&mut bytes).unwrap_or_else(|_| {
        let now = js_sys::Date::now().to_bits().to_be_bytes();
        bytes.copy_from_slice(&now);
    });
    hex::encode(bytes)
}
