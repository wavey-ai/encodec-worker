ENCODEC_DIST ?= ../encodec-rs/dist/wasm-fixed-bundles/bundles
wasm_DIR ?= wasm

PROFILES := \
	encodec_48khz_12kbps_1333ms \
	encodec_48khz_12kbps_1800ms

.PHONY: wasm wasm-clean wasm-verify build dev test deploy

wasm: wasm-clean
	@set -eu; \
	for profile in $(PROFILES); do \
		src="$(ENCODEC_DIST)/$$profile"; \
		dst="$(wasm_DIR)/$$profile"; \
		test -d "$$src" || { echo "Missing bundle directory: $$src"; exit 1; }; \
		mkdir -p "$$dst"; \
		for file in bundle.json encode_frame.onnx lm_weights_q8.bin manifest.json; do \
			test -s "$$src/$$file" || { echo "Missing or empty asset: $$src/$$file"; exit 1; }; \
			cp "$$src/$$file" "$$dst/$$file"; \
		done; \
		echo "Copied $$profile"; \
	done
	@$(MAKE) wasm-verify

wasm-clean:
	rm -rf \
		"$(wasm_DIR)/encodec_48khz_12kbps_1333ms" \
		"$(wasm_DIR)/encodec_48khz_12kbps_1800ms"

wasm-verify:
	@set -eu; \
	for profile in $(PROFILES); do \
		for file in bundle.json encode_frame.onnx lm_weights_q8.bin manifest.json; do \
			path="$(wasm_DIR)/$$profile/$$file"; \
			test -s "$$path" || { echo "Missing or empty asset: $$path"; exit 1; }; \
		done; \
	done
	@find "$(wasm_DIR)" -maxdepth 2 -type f -exec ls -lh {} \;

build: wasm
	worker-build --release

dev: wasm
	wrangler dev

test:
	cargo test

deploy: wasm
	wrangler deploy
