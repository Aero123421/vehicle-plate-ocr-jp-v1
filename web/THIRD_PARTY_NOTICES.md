# Third-Party Notices & Open-Source Attributions

vehicle-plate-ocr-jp-v1 (`v0.6.5`) incorporates or adapts components and pretrained weights strictly under permissive commercial-clean licenses (`Apache-2.0`, `MIT`, `BSD-3-Clause`, `SIL OFL-1.1`). Zero `AGPL-3.0`, `GPL-3.0`, or non-commercial (`CC-BY-NC`) components or weights are used anywhere in this repository.

---

## 1. D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement

- **Upstream Repository**: https://github.com/Peterande/D-FINE
- **Copyright**: Copyright (c) 2024 The D-FINE Authors (Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, Feng Wu — University of Science and Technology of China).
- **License**: Apache License, Version 2.0 (`http://www.apache.org/licenses/LICENSE-2.0`)
- **Pinned Pretrained Checkpoints (`out/models/pretrained_apache20/`)**:
  1. **`ustc-community/dfine-nano-coco`**
     - **Source URL**: `https://huggingface.co/ustc-community/dfine-nano-coco`
     - **Immutable Revision**: `066438d3d8f0da137a37b38fdf3368fd4afceced`
     - **Artifact**: `model.safetensors` -> `dfine_nano_coco.safetensors` (`15,278,996` bytes)
     - **SHA-256**: `19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b`
     - **License**: `Apache-2.0` (trained on COCO 2017 object detection annotations; model weights released by the D-FINE authors under Apache-2.0).
  2. **`ustc-community/dfine-small-obj2coco`**
     - **Source URL**: `https://huggingface.co/ustc-community/dfine-small-obj2coco`
     - **Immutable Revision**: `2b548d12a7623d674bb2aa2bf24022ce090b7736`
     - **Artifact**: `model.safetensors` -> `dfine_small_obj2coco.safetensors` (`41,497,620` bytes)
     - **SHA-256**: `8ff4a8332cf9507209fa443058d00752ebfede3f8cb497b91d4238369e3a57f7`
     - **License**: `Apache-2.0` (pretrained on Objects365 and fine-tuned on COCO 2017; model weights released by the D-FINE authors under Apache-2.0).

---

## 2. Hugging Face Transformers (`HGNetV2` & `D-FINE` PyTorch Modeling Port)

- **Upstream Repository**: https://github.com/huggingface/transformers (`v4.52.4`, `src/transformers/models/hgnet_v2/` and `src/transformers/models/d_fine/`)
- **Copyright**: Copyright 2025 Baidu Inc. and The HuggingFace Inc. team.
- **License**: Apache License, Version 2.0 (`http://www.apache.org/licenses/LICENSE-2.0`)
- **Modifications in vehicle-plate-ocr-jp-v1 (`src/vpocr_det/dfine_core.py`)**:
  - Self-contained PyTorch & ONNX Opset-17 port removing external `transformers` / `safetensors` runtime dependencies.
  - Exposes multi-scale `HGNetV2` (`C2` stride-4, `C3` stride-8, `C4` stride-16, `C5` stride-32) and `DFineHybridEncoder` PAN features alongside frozen `D-FINE` decoder queries, logits, and Fine-grained Distribution Refinement (`FDR`) bounding boxes for downstream vehicle silhouette cutout, construction subtype classification, and high-acuity Japanese license plate detection.
  - Deterministic cryptographic SHA-256 verification of `.safetensors` checkpoints prior to loading.

---

## Apache License, Version 2.0 Summary Notice

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
