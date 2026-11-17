"""Rendering engine for motion sequences: homography warping and 3D scene composition."""

import math
from typing import Any
import cv2
import numpy as np
from PIL import Image

from vpocr_synth.motion.camera import PinholeCamera
from vpocr_synth.motion.models import (
    CameraParams,
    CharacterQuad,
    FrameAnnotation,
    Pose3D,
    Quad2D,
)
from vpocr_synth.plate.models import BoundingBox, PlateSize, RenderResult


class MotionRenderer:
    """Renders perspective-warped license plates into realistic 3D multi-camera scenes."""

    def __init__(self, camera: PinholeCamera) -> None:
        self.camera = camera
        self.params = camera.params

    def render_frame(
        self,
        canonical_result: RenderResult,
        pose: Pose3D,
        timestamp: float,
        frame_index: int,
        vehicle_color: tuple[int, int, int] = (180, 182, 186),
        is_rear: bool = False,
        plate_side: int | None = None,
        vehicle_length_m: float = 4.6,
        wind_seed: int = 0,
    ) -> tuple[Image.Image, FrameAnnotation]:
        """Render a single perspective frame with ground-truth quadrilaterals and incidence telemetry."""
        active_cam = self.camera.with_wind_shake(timestamp, seed=wind_seed + self.params.camera_id * 97)
        cam_w = active_cam.params.width
        cam_h = active_cam.params.height

        if plate_side is None:
            plate_side = 1 if is_rear else 0

        # 1. Physical Plate Dimensions
        is_large = canonical_result.config.plate_size == PlateSize.LARGE
        plate_w_m = 0.440 if is_large else 0.330
        plate_h_m = 0.220 if is_large else 0.165

        # 2. Incidence Angle & 3D Plate Projection
        incidence_deg, _, _ = active_cam.compute_incidence_angle(pose)
        is_back_facing = incidence_deg >= 88.0

        corners_3d = active_cam.compute_plate_corners_3d(pose, plate_w_m, plate_h_m, is_rear)
        pts_2d: list[tuple[float, float]] = []
        depths: list[float] = []
        for c3 in corners_3d:
            u, v, z = active_cam.project_point(c3)
            pts_2d.append((u, v))
            depths.append(z)

        plate_quad = Quad2D(points=pts_2d)
        mean_depth = float(np.mean(depths))
        true_dist = float(np.linalg.norm(pose.position - active_cam.params.position))

        # Check visibility (back-facing >= 88 deg is occluded by vehicle body)
        visibility = 0.0 if is_back_facing else active_cam.compute_visibility(plate_quad)
        plate_w_px = plate_quad.width if not is_back_facing else 0.0
        plate_h_px = plate_quad.height if not is_back_facing else 0.0
        is_visible = (
            not is_back_facing
            and mean_depth > 0.2
            and visibility >= 0.05
            and plate_w_px >= 8.0
            and all(z > 0.1 for z in depths)
        )

        # Readable view requires front-hemisphere non-extreme angle (<= 76 deg) and sufficient pixel height
        is_readable_view = (
            is_visible
            and incidence_deg <= 76.0
            and plate_side != 2
            and plate_h_px >= 8.0
            and visibility >= 0.25
        )

        angles_deg = active_cam.compute_relative_angles(pose)

        # 3. Create Scene Canvas (Road background + 3D Vehicle Body Box)
        canvas_bgr = self._render_background_and_vehicle_3d(
            cam=active_cam,
            pose=pose,
            vehicle_color=vehicle_color,
            is_rear=is_rear,
            vehicle_length_m=vehicle_length_m,
        )

        # 4. Homography Warping of Canonical Plate
        char_quads: list[CharacterQuad] = []
        field_quads: dict[str, Quad2D] = {}

        if is_visible:
            src_img = canonical_result.image
            src_w = float(src_img.width)
            src_h = float(src_img.height)

            src_corners = np.array([
                [0.0, 0.0],
                [src_w, 0.0],
                [src_w, src_h],
                [0.0, src_h],
            ], dtype=np.float32)

            dst_corners = np.array(pts_2d, dtype=np.float32)
            H = cv2.getPerspectiveTransform(src_corners, dst_corners)

            src_bgr = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)

            # Anti-aliasing pre-filtering when downscaling
            scale_ratio = plate_w_px / max(1.0, src_w)
            if scale_ratio < 0.6:
                down_w = max(24, int(round(src_w * max(scale_ratio * 1.5, 0.18))))
                down_h = max(12, int(round(src_h * max(scale_ratio * 1.5, 0.18))))
                src_bgr_filtered = cv2.resize(src_bgr, (down_w, down_h), interpolation=cv2.INTER_AREA)
                src_corners_down = np.array([
                    [0.0, 0.0],
                    [float(down_w), 0.0],
                    [float(down_w), float(down_h)],
                    [0.0, float(down_h)],
                ], dtype=np.float32)
                H_down = cv2.getPerspectiveTransform(src_corners_down, dst_corners)
                warped_plate = cv2.warpPerspective(
                    src_bgr_filtered,
                    H_down,
                    (cam_w, cam_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
            else:
                warped_plate = cv2.warpPerspective(
                    src_bgr,
                    H,
                    (cam_w, cam_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

            mask = np.zeros((cam_h, cam_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_corners), 255, lineType=cv2.LINE_AA)
            mask_float = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0.5)[:, :, np.newaxis]
            canvas_bgr = (canvas_bgr * (1.0 - mask_float) + warped_plate * mask_float).astype(np.uint8)

            # 5. Project Character Quadrilaterals
            for cb in canonical_result.character_boxes:
                b = cb.bbox
                c_pts = np.array([
                    [b.x_min, b.y_min],
                    [b.x_max, b.y_min],
                    [b.x_max, b.y_max],
                    [b.x_min, b.y_max],
                ], dtype=np.float32).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(c_pts, H).reshape(-1, 2)
                quad = Quad2D(points=[(float(p[0]), float(p[1])) for p in proj])
                char_quads.append(
                    CharacterQuad(char=cb.char, field=cb.field, index=cb.index, quad=quad)
                )

            # 6. Project Field Quadrilaterals
            for fname, fb in canonical_result.field_boxes.items():
                f_pts = np.array([
                    [fb.x_min, fb.y_min],
                    [fb.x_max, fb.y_min],
                    [fb.x_max, fb.y_max],
                    [fb.x_min, fb.y_max],
                ], dtype=np.float32).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(f_pts, H).reshape(-1, 2)
                field_quads[fname] = Quad2D(points=[(float(p[0]), float(p[1])) for p in proj])

        rgb_img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))

        annotation = FrameAnnotation(
            frame_index=frame_index,
            timestamp=timestamp,
            corners=plate_quad,
            plate_width_px=plate_w_px,
            plate_height_px=plate_h_px,
            distance_m=mean_depth if mean_depth > 0.1 else true_dist,
            angles_deg=angles_deg,
            visibility=visibility,
            is_visible=is_visible,
            pose=pose,
            character_quads=char_quads,
            field_quads=field_quads,
            incidence_deg=incidence_deg,
            is_readable_view=is_readable_view,
            camera_id=active_cam.params.camera_id,
            plate_side=plate_side,
        )

        return rgb_img, annotation

    def _render_background_and_vehicle_3d(
        self,
        cam: PinholeCamera,
        pose: Pose3D,
        vehicle_color: tuple[int, int, int],
        is_rear: bool,
        vehicle_length_m: float = 4.6,
    ) -> np.ndarray:
        """Render vectorized sky/asphalt ground and painter's-sorted 3D vehicle body box."""
        w = cam.params.width
        h = cam.params.height
        pitch_rad = math.radians(cam.params.pitch_deg)
        horizon_y = int(round(cam.params.cy - cam.params.fy * math.tan(pitch_rad)))
        horizon_clamped = max(0, min(h, horizon_y))

        canvas = np.empty((h, w, 3), dtype=np.uint8)
        if horizon_clamped > 0:
            # Subtle sky gradient from top to horizon
            sky_rows = np.linspace(0.0, 1.0, horizon_clamped, dtype=np.float32)[:, np.newaxis]
            sky_b = (224.0 - 12.0 * sky_rows).astype(np.uint8)
            sky_g = (212.0 - 4.0 * sky_rows).astype(np.uint8)
            sky_r = (198.0 + 8.0 * sky_rows).astype(np.uint8)
            sky_col = np.stack([sky_b, sky_g, sky_r], axis=-1)
            canvas[:horizon_clamped, :, :] = sky_col

        if horizon_clamped < h:
            # Vectorized asphalt ground gradient from horizon to bottom
            n_ground = h - horizon_clamped
            prog = np.linspace(0.0, 1.0, n_ground, dtype=np.float32)[:, np.newaxis]
            val = (76.0 - 28.0 * prog).astype(np.uint8)
            g_col = np.stack([val, val + np.uint8(2), val + np.uint8(4)], axis=-1)
            canvas[horizon_clamped:, :, :] = g_col

        # Lane markings on asphalt (subtle gate stop line & lane edges)
        for lx in (-1.65, 1.65):
            lp0 = cam.project_point(np.array([lx, -2.0, 0.01]))
            lp1 = cam.project_point(np.array([lx, 24.0, 0.01]))
            if lp0[2] > 0.2 and lp1[2] > 0.2:
                cv2.line(
                    canvas,
                    (int(round(lp0[0])), int(round(lp0[1]))),
                    (int(round(lp1[0])), int(round(lp1[1]))),
                    (112, 114, 116),
                    max(1, int(round(w / 640))),
                    lineType=cv2.LINE_AA,
                )

        # 3D Vehicle Body Box oriented along plate local axes
        u_p, v_p, n_plate = cam.compute_plate_axes(pose)
        # -n_plate always points from the active plate face INTO the vehicle interior
        into_car = -n_plate

        half_bw = 0.925  # 1.85m vehicle width
        veh_len = max(3.2, vehicle_length_m)
        z_bot_off = -0.34
        z_bumper_top_off = 0.42
        z_roof_off = 1.08

        c0 = pose.position
        # 8 corners of lower/main body prism [0..veh_len along into_car]
        f_bl = c0 - half_bw * u_p + z_bot_off * v_p
        f_br = c0 + half_bw * u_p + z_bot_off * v_p
        f_tr = c0 + half_bw * u_p + z_bumper_top_off * v_p
        f_tl = c0 - half_bw * u_p + z_bumper_top_off * v_p

        b_bl = f_bl + veh_len * into_car
        b_br = f_br + veh_len * into_car
        b_tr = f_tr + veh_len * into_car
        b_tl = f_tl + veh_len * into_car

        # Cabin upper prism [0.95m .. veh_len - 0.55m]
        cab_d0 = min(veh_len * 0.24, 1.05)
        cab_d1 = max(cab_d0 + 1.2, veh_len - 0.55)
        half_cw = half_bw * 0.86
        c_f_tl = c0 - half_cw * u_p + z_roof_off * v_p + cab_d0 * into_car
        c_f_tr = c0 + half_cw * u_p + z_roof_off * v_p + cab_d0 * into_car
        c_b_tr = c0 + half_cw * u_p + z_roof_off * v_p + cab_d1 * into_car
        c_b_tl = c0 - half_cw * u_p + z_roof_off * v_p + cab_d1 * into_car

        c_base_ftl = f_tl + cab_d0 * into_car
        c_base_ftr = f_tr + cab_d0 * into_car
        c_base_btr = f_tr + cab_d1 * into_car
        c_base_btl = f_tl + cab_d1 * into_car

        # Ground shadow beneath vehicle
        sh_pts_3d = [
            np.array([f_bl[0], f_bl[1], 0.02]),
            np.array([f_br[0], f_br[1], 0.02]),
            np.array([b_br[0], b_br[1], 0.02]),
            np.array([b_bl[0], b_bl[1], 0.02]),
        ]
        sh_proj = [cam.project_point(p) for p in sh_pts_3d]
        if all(p[2] > 0.15 for p in sh_proj):
            sh_poly = np.array([(p[0], p[1]) for p in sh_proj], dtype=np.int32)
            cv2.fillConvexPoly(canvas, sh_poly, (26, 27, 29), lineType=cv2.LINE_AA)

        def shade_bgr(rgb: tuple[int, int, int], mult: float) -> tuple[int, int, int]:
            return (
                int(np.clip(rgb[2] * mult, 0, 255)),
                int(np.clip(rgb[1] * mult, 0, 255)),
                int(np.clip(rgb[0] * mult, 0, 255)),
            )

        # Assemble 3D faces with directional shading multipliers for painter's depth sort
        faces: list[tuple[list[np.ndarray], tuple[int, int, int], str]] = [
            ([f_tl, f_tr, f_br, f_bl], shade_bgr(vehicle_color, 1.00), "plate_end"),
            ([b_tr, b_tl, b_bl, b_br], shade_bgr(vehicle_color, 0.82), "opposite_end"),
            ([b_tl, f_tl, f_bl, b_bl], shade_bgr(vehicle_color, 0.88), "side_left"),
            ([f_tr, b_tr, b_br, f_br], shade_bgr(vehicle_color, 0.92), "side_right"),
            ([b_tl, b_tr, f_tr, f_tl], shade_bgr(vehicle_color, 1.06), "deck_" "top"),
            # Cabin / windshield / side windows
            ([c_f_tl, c_f_tr, c_base_ftr, c_base_ftl], (58, 66, 74), "windshield_near"),
            ([c_b_tr, c_b_tl, c_base_btl, c_base_btr], (52, 58, 66), "windshield_far"),
            ([c_b_tl, c_f_tl, c_base_ftl, c_base_btl], (62, 70, 78), "window_left"),
            ([c_f_tr, c_b_tr, c_base_btr, c_base_ftr], (64, 72, 80), "window_right"),
            ([c_b_tl, c_b_tr, c_f_tr, c_f_tl], shade_bgr(vehicle_color, 1.08), "roof"),
        ]

        scored_faces: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], str]] = []
        for pts3, col_bgr, tag in faces:
            projs = [cam.project_point(p) for p in pts3]
            if all(pr[2] > 0.15 for pr in projs):
                centroid_3d = sum(pts3) / float(len(pts3))
                dist_cam = float(np.linalg.norm(centroid_3d - cam.params.position))
                pts2 = [(pr[0], pr[1]) for pr in projs]
                scored_faces.append((dist_cam, pts2, col_bgr, tag))

        # Painter's algorithm: farthest face first, closest face last
        scored_faces.sort(key=lambda item: item[0], reverse=True)

        for _, pts2, col_bgr, tag in scored_faces:
            poly = np.array(pts2, dtype=np.int32)
            cv2.fillConvexPoly(canvas, poly, col_bgr, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [poly], True, (38, 40, 44), 1, lineType=cv2.LINE_AA)

            if tag == "plate_end":
                # Draw dark grill/recess & headlights (or taillights if rear) on the plate-bearing end
                grill_w = 0.90
                grill_h = 0.38
                g_pts3 = [
                    c0 - (grill_w / 2.0) * u_p + (grill_h / 2.0) * v_p,
                    c0 + (grill_w / 2.0) * u_p + (grill_h / 2.0) * v_p,
                    c0 + (grill_w / 2.0) * u_p - (grill_h / 2.0) * v_p,
                    c0 - (grill_w / 2.0) * u_p - (grill_h / 2.0) * v_p,
                ]
                g_pr = [cam.project_point(gp) for gp in g_pts3]
                if all(gp[2] > 0.15 for gp in g_pr):
                    g_poly = np.array([(gp[0], gp[1]) for gp in g_pr], dtype=np.int32)
                    cv2.fillConvexPoly(canvas, g_poly, (26, 27, 30), lineType=cv2.LINE_AA)

                lamp_col = (195, 242, 255) if not is_rear else (35, 35, 225)
                for lamp_side in (-0.76, 0.76):
                    lp = c0 + (lamp_side * u_p) + (0.16 * v_p)
                    lu, lv, lz = cam.project_point(lp)
                    if lz > 0.15 and -50 <= lu < w + 50 and -50 <= lv < h + 50:
                        lr = max(3, int(round(32.0 / lz)))
                        cv2.circle(canvas, (int(round(lu)), int(round(lv))), lr, lamp_col, -1, lineType=cv2.LINE_AA)

        # Draw wheels along left/right sides if visible
        for side_sign in (-1.0, 1.0):
            for axle_d in (0.75, veh_len - 0.85):
                wp = c0 + (side_sign * half_bw * u_p) + (-0.18 * v_p) + (axle_d * into_car)
                wu, wv, wz = cam.project_point(wp)
                if wz > 0.2 and -40 <= wu < w + 40 and -40 <= wv < h + 40:
                    wr = max(3, int(round(48.0 / wz)))
                    cv2.circle(canvas, (int(round(wu)), int(round(wv))), wr, (22, 23, 25), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (int(round(wu)), int(round(wv))), max(1, wr // 2), (115, 118, 122), -1, lineType=cv2.LINE_AA)

        return canvas
