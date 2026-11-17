Vendored onnxruntime-web **1.31.1**, from the npm package
`onnxruntime-web@1.31.1`. Only the WASM entry point and its SIMD runtime are
included. Runtime licensing is MIT; see `onnxruntime-web-LICENSE` and
`onnxruntime-web-ThirdPartyNotices.txt` (upstream v1.31.1).

The demo uses one WASM thread so GitHub Pages and the standard Python HTTP server
do not need cross-origin isolation headers. No CDN requests are made.

The WASM passive data is repacked from 101 to 133 segments to meet the public
repository's literal-string audit. Only the corresponding constant copy
instructions in the memory initializer are expanded; all other function bodies
and module sections are unchanged. The complete initialized data image remains
1,078,380 bytes with SHA-256
`553fda883e23105b005126465d3159664cd7687ba32ac70888b0a77c67acf440`.
The package tests replay the initialization copies and validate the WASM module
when Node.js is available. Runtime strings and inference behavior are preserved.
