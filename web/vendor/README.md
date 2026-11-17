Vendored, unmodified onnxruntime-web **1.31.1**, from the npm package
`onnxruntime-web@1.31.1`. Only the WASM entry point and its SIMD runtime are
included. Runtime licensing is MIT; see `onnxruntime-web-LICENSE` and
`onnxruntime-web-ThirdPartyNotices.txt` (upstream v1.31.1).

The demo uses one WASM thread so GitHub Pages and the standard Python HTTP server
do not need cross-origin isolation headers. No CDN requests are made.
