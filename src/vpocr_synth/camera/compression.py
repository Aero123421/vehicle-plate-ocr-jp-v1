"""H.264 / GOP-aware video compression and low-resolution RTSP substream simulation."""

import cv2
import numpy as np

from vpocr_synth.camera.models import CameraProfile


class CompressionSimulator:
    """Simulates video compression artifacts, GOP quality fluctuations, and RTSP substream scaling."""

    def __init__(self, profile: CameraProfile) -> None:
        self.profile = profile
        self.gop_size = max(1, profile.gop_size)
        self.q_i = profile.jpeg_quality_i
        self.q_p = profile.jpeg_quality_p
        self.substream_scale = float(np.clip(profile.substream_scale, 0.2, 1.0))

    def compress_frame(
        self,
        bgr_image: np.ndarray,
        frame_index: int,
    ) -> tuple[np.ndarray, int, str]:
        """Compress frame returning (compressed_bgr, quality_factor, frame_type 'I' or 'P')."""
        orig_h, orig_w = bgr_image.shape[:2]
        work_img = bgr_image

        # Optional RTSP low-resolution substream downscale before codec compression
        if self.substream_scale < 0.99:
            sub_w = max(64, int(round(orig_w * self.substream_scale)))
            sub_h = max(36, int(round(orig_h * self.substream_scale)))
            work_img = cv2.resize(bgr_image, (sub_w, sub_h), interpolation=cv2.INTER_AREA)

        gop_pos = frame_index % self.gop_size
        if gop_pos == 0:
            frame_type = "I"
            quality = self.q_i
        else:
            frame_type = "P"
            drift = int(round(5.0 * (gop_pos / float(self.gop_size))))
            quality = max(18, self.q_p - drift)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, encoded = cv2.imencode(".jpg", work_img, encode_param)
        if not success:
            decoded = work_img
        else:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if self.substream_scale < 0.99:
            decoded = cv2.resize(decoded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return decoded, quality, frame_type
