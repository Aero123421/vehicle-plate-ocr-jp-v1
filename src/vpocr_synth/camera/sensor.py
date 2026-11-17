"""CMOS sensor Poisson-Gaussian noise, gain amplification, 3DNR ghosting, and ISP sharpening."""

import cv2
import numpy as np

from vpocr_synth.camera.models import CameraProfile


class SensorSimulator:
    """Simulates CMOS sensor Poisson-Gaussian noise, 3DNR temporal ghosting, and ISP sharpening."""

    def __init__(self, profile: CameraProfile) -> None:
        self.profile = profile
        self.gain_linear = 10.0 ** (profile.gain_db / 20.0)
        self.prev_frame_float: np.ndarray | None = None

    def apply_sensor(
        self,
        bgr_image: np.ndarray,
        frame_seed: int,
    ) -> tuple[np.ndarray, float]:
        """Apply sensor noise, optional 3DNR ghosting, and ISP sharpening."""
        rng = np.random.default_rng(frame_seed)
        img_float = bgr_image.astype(np.float32)

        # 1. Poisson-Gaussian Noise Model
        shot_scale = max(1.0, 45.0 / self.gain_linear)
        shot_noise = rng.normal(0.0, 1.0, img_float.shape).astype(np.float32) * np.sqrt(
            np.maximum(1.0, img_float) / shot_scale
        )

        base_read_sigma = 1.2
        effective_read_sigma = base_read_sigma * self.gain_linear
        read_noise = rng.normal(0.0, effective_read_sigma, img_float.shape).astype(np.float32)

        noisy = img_float + shot_noise + read_noise
        noise_sigma = float(effective_read_sigma + np.sqrt(128.0 / shot_scale))

        # 2. 3DNR Temporal Noise Reduction Ghosting (common in low-light CCTV ISPs)
        tnr = float(np.clip(self.profile.temporal_nr_strength, 0.0, 0.90))
        if tnr > 0.01:
            if self.prev_frame_float is not None and self.prev_frame_float.shape == noisy.shape:
                diff = np.mean(np.abs(noisy - self.prev_frame_float), axis=2, keepdims=True)
                # Motion-adaptive blend: imperfectly suppresses fast-moving plate edges -> ghost trails
                motion_gate = np.exp(-diff / (32.0 + 75.0 * tnr))
                alpha_ghost = tnr * (0.35 + 0.65 * motion_gate)
                noisy = (1.0 - alpha_ghost) * noisy + alpha_ghost * self.prev_frame_float
            self.prev_frame_float = noisy.copy()

        # 3. ISP Edge Sharpening & Ringing Halos (Unsharp Masking)
        if self.profile.enable_sharpening and self.profile.sharpening_amount > 0.05:
            blurred = cv2.GaussianBlur(noisy, (5, 5), 1.2)
            high_pass = noisy - blurred
            sharpened = noisy + self.profile.sharpening_amount * high_pass
            noisy = np.clip(sharpened, 0, 255)
        else:
            noisy = np.clip(noisy, 0, 255)

        return noisy.astype(np.uint8), noise_sigma
