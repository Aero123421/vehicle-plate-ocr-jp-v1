"""Pinhole camera geometry, 3D projection, and visibility calculation."""

import math
from typing import Any
import numpy as np

from vpocr_synth.motion.models import CameraParams, Pose3D, Quad2D


class PinholeCamera:
    """Projective 3D-to-2D pinhole camera with extrinsics and radial distortion."""

    def __init__(self, params: CameraParams | None = None) -> None:
        self.params = params or CameraParams()
        self._R_cw = self._compute_extrinsic_rotation()

    def _compute_extrinsic_rotation(self) -> np.ndarray:
        """Compute 3x3 rotation matrix R_cw mapping world vectors to camera frame.

        OpenCV Camera Coordinate System:
        - X_c: Right
        - Y_c: Down
        - Z_c: Forward (Optical axis)

        World Coordinate System:
        - X_w: Right
        - Y_w: Forward along road/lane
        - Z_w: Upward (Z=0 road surface)
        """
        pitch_rad = math.radians(self.params.pitch_deg)  # Downward tilt (俯角)
        yaw_rad = math.radians(self.params.yaw_deg)      # Horizontal pan toward lane
        roll_rad = math.radians(self.params.roll_deg)

        # Optical axis direction in world coordinates:
        # Panning by yaw_rad, tilted down by pitch_rad
        zc = np.array([
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad),
            -math.sin(pitch_rad),
        ], dtype=np.float64)
        zc = zc / np.linalg.norm(zc)

        # World up reference
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Camera horizontal right vector xc = cross(zc, world_up)
        xc = np.cross(zc, world_up)
        norm_xc = np.linalg.norm(xc)
        if norm_xc < 1e-6:
            xc = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            xc = xc / norm_xc

        # Camera downward vector yc = cross(zc, xc)
        yc = np.cross(zc, xc)
        yc = yc / np.linalg.norm(yc)

        # Apply roll rotation around optical axis zc if present
        if abs(roll_rad) > 1e-5:
            cos_r = math.cos(roll_rad)
            sin_r = math.sin(roll_rad)
            xc_rolled = cos_r * xc + sin_r * yc
            yc_rolled = -sin_r * xc + cos_r * yc
            xc = xc_rolled
            yc = yc_rolled

        # R_cw maps world vector diff = (P_w - C_w) to camera frame: P_c = R_cw @ diff
        return np.vstack([xc, yc, zc])

    @property
    def R_cw(self) -> np.ndarray:
        return self._R_cw

    def world_to_camera(self, p_w: np.ndarray) -> np.ndarray:
        """Transform 3D point from world coordinates to camera coordinates."""
        diff = p_w - self.params.position
        return self._R_cw @ diff

    def camera_to_image(self, p_c: np.ndarray) -> tuple[float, float, float]:
        """Project camera coordinate point [Xc, Yc, Zc] to image pixel coordinates (u, v, depth)."""
        xc, yc, zc = float(p_c[0]), float(p_c[1]), float(p_c[2])
        if zc <= 0.05:
            # Behind or at camera plane
            return -1000.0, -1000.0, zc

        xn = xc / zc
        yn = yc / zc

        # Radial lens distortion
        k1 = self.params.k1
        k2 = self.params.k2
        if abs(k1) > 1e-7 or abs(k2) > 1e-7:
            r2 = xn * xn + yn * yn
            radial = 1.0 + k1 * r2 + k2 * (r2 * r2)
            xn *= radial
            yn *= radial

        u = self.params.fx * xn + self.params.cx
        v = self.params.fy * yn + self.params.cy
        return u, v, zc

    def project_point(self, p_w: np.ndarray) -> tuple[float, float, float]:
        """Transform world 3D point to image pixel coordinates and depth."""
        pc = self.world_to_camera(p_w)
        return self.camera_to_image(pc)

    @staticmethod
    def compute_plate_axes(pose: Pose3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute orthonormal plate basis (u_p, v_p, n_plate) in world coordinates.

        - u_p: horizontal left-to-right reading axis on plate face
        - v_p: vertical bottom-to-top axis on plate face
        - n_plate: outward surface normal pointing away from the vehicle body
        """
        yaw_rad = math.radians(pose.yaw_deg)
        pitch_rad = math.radians(pose.pitch_deg)
        roll_rad = math.radians(pose.roll_deg)

        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # Vehicle interior-pointing axis (-n_plate), right axis (u_p), and up axis (v_p)
        forward = np.array([-sin_y, cos_y, 0.0], dtype=np.float64)
        right = np.array([cos_y, sin_y, 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        if abs(pitch_rad) > 1e-5:
            cos_p = math.cos(pitch_rad)
            sin_p = math.sin(pitch_rad)
            f_new = cos_p * forward + sin_p * up
            u_new = -sin_p * forward + cos_p * up
            forward, up = f_new, u_new

        if abs(roll_rad) > 1e-5:
            cos_r = math.cos(roll_rad)
            sin_r = math.sin(roll_rad)
            r_new = cos_r * right + sin_r * up
            u_new = -sin_r * right + cos_r * up
            right, up = r_new, u_new

        u_p = right / np.linalg.norm(right)
        v_p = up / np.linalg.norm(up)
        n_plate = np.cross(u_p, v_p)
        n_norm = np.linalg.norm(n_plate)
        if n_norm > 1e-8:
            n_plate = n_plate / n_norm
        else:
            n_plate = np.array([sin_y, -cos_y, 0.0], dtype=np.float64)
        return u_p, v_p, n_plate

    def compute_plate_corners_3d(
        self,
        pose: Pose3D,
        width_m: float = 0.330,
        height_m: float = 0.165,
        is_rear: bool = False,
    ) -> list[np.ndarray]:
        """Calculate 4 corners in 3D world coordinates for the plate.

        Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        """
        u_p, v_p, _ = self.compute_plate_axes(pose)

        half_w = width_m / 2.0
        half_h = height_m / 2.0
        center = pose.position

        # Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        c_tl = center - half_w * u_p + half_h * v_p
        c_tr = center + half_w * u_p + half_h * v_p
        c_br = center + half_w * u_p - half_h * v_p
        c_bl = center - half_w * u_p - half_h * v_p

        return [c_tl, c_tr, c_br, c_bl]

    def project_plate_corners(
        self,
        pose: Pose3D,
        width_m: float = 0.330,
        height_m: float = 0.165,
        is_rear: bool = False,
    ) -> Quad2D:
        """Compute projected 2D quadrilateral in camera pixels for plate corners."""
        corners_3d = self.compute_plate_corners_3d(pose, width_m, height_m, is_rear)
        pts_2d: list[tuple[float, float]] = []
        for c3 in corners_3d:
            u, v, _ = self.project_point(c3)
            pts_2d.append((u, v))
        return Quad2D(points=pts_2d)

    def compute_incidence_angle(self, pose: Pose3D) -> tuple[float, float, float]:
        """Compute (incidence_deg, plate_azimuth_deg, plate_elevation_deg) between camera ray and plate normal.

        - incidence_deg: 0 deg = dead-on frontal; 90 deg = edge-on; > 90 deg = back-facing (behind plate).
        """
        ray_vec = self.params.position - pose.position
        dist = float(np.linalg.norm(ray_vec))
        if dist < 1e-6:
            return 0.0, 0.0, 0.0

        ray_unit = ray_vec / dist
        u_p, v_p, n_plate = self.compute_plate_axes(pose)

        cos_inc = float(np.clip(np.dot(n_plate, ray_unit), -1.0, 1.0))
        incidence_deg = math.degrees(math.acos(cos_inc))

        rx = float(np.dot(ray_unit, u_p))
        ry = float(np.dot(ray_unit, v_p))
        rz = float(np.dot(ray_unit, n_plate))
        plate_azimuth_deg = math.degrees(math.atan2(rx, rz))
        plate_elevation_deg = math.degrees(math.atan2(ry, max(1e-6, math.hypot(rx, rz))))
        return incidence_deg, plate_azimuth_deg, plate_elevation_deg

    def compute_relative_angles(self, pose: Pose3D) -> dict[str, float]:
        """Compute relative viewing angles (yaw, pitch, roll, incidence) between camera and plate."""
        diff = pose.position - self.params.position
        dist = float(np.linalg.norm(diff))
        if dist < 1e-5:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "incidence": 0.0,
                "plate_azimuth": 0.0,
                "plate_elevation": 0.0,
            }

        # Angle of line-of-sight
        los_cam = self._R_cw @ (diff / dist)

        rel_yaw = math.degrees(math.atan2(los_cam[0], los_cam[2]))
        rel_pitch = math.degrees(math.atan2(los_cam[1], los_cam[2]))
        rel_roll = float(pose.roll_deg - self.params.roll_deg)
        inc_deg, p_az, p_el = self.compute_incidence_angle(pose)

        return {
            "yaw": rel_yaw,
            "pitch": rel_pitch,
            "roll": rel_roll,
            "incidence": inc_deg,
            "plate_azimuth": p_az,
            "plate_elevation": p_el,
        }

    def with_wind_shake(self, timestamp: float, seed: int = 0) -> "PinholeCamera":
        """Return camera perturbed by subtle deterministic wind/pole vibration at time t."""
        amp = self.params.wind_shake_deg
        if amp <= 1e-4:
            return self

        ph1 = (seed * 0.713) % (2.0 * math.pi)
        ph2 = (seed * 1.379) % (2.0 * math.pi)
        ph3 = (seed * 2.117) % (2.0 * math.pi)
        dpitch = amp * (0.65 * math.sin(2.0 * math.pi * 2.3 * timestamp + ph1) + 0.35 * math.sin(2.0 * math.pi * 5.1 * timestamp + ph2))
        dyaw = amp * 0.75 * math.sin(2.0 * math.pi * 1.9 * timestamp + ph2)
        droll = amp * 0.50 * math.sin(2.0 * math.pi * 3.4 * timestamp + ph3)

        from dataclasses import replace
        shaken_params = replace(
            self.params,
            pitch_deg=self.params.pitch_deg + dpitch,
            yaw_deg=self.params.yaw_deg + dyaw,
            roll_deg=self.params.roll_deg + droll,
        )
        return PinholeCamera(shaken_params)

    @staticmethod
    def aim_at(
        position: np.ndarray,
        target: np.ndarray,
        width: int = 1920,
        height: int = 1080,
        fov_h_deg: float = 55.0,
        roll_deg: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        camera_id: int = 0,
        mount_role: str = "front",
        wind_shake_deg: float = 0.0,
        clock_offset_s: float = 0.0,
        pan_jitter_deg: float = 0.0,
        tilt_jitter_deg: float = 0.0,
    ) -> CameraParams:
        """Construct CameraParams at arbitrary 3D world position aimed at target 3D point."""
        look = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        horiz = max(1e-5, math.hypot(float(look[0]), float(look[1])))
        yaw_deg = math.degrees(math.atan2(float(look[0]), float(look[1]))) + pan_jitter_deg
        pitch_deg = math.degrees(math.atan2(-float(look[2]), horiz)) + tilt_jitter_deg
        return CameraParams(
            width=width,
            height=height,
            fov_h_deg=fov_h_deg,
            position=np.asarray(position, dtype=np.float64),
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            k1=k1,
            k2=k2,
            camera_id=camera_id,
            mount_role=mount_role,
            wind_shake_deg=wind_shake_deg,
            clock_offset_s=clock_offset_s,
        )

    def compute_visibility(self, quad: Quad2D) -> float:
        """Compute polygon visibility ratio (0.0 to 1.0) inside camera frame."""
        pts = quad.points
        w = float(self.params.width)
        h = float(self.params.height)

        # Quick check: if all points inside [0, w] x [0, h]
        if all(0.0 <= x <= w and 0.0 <= y <= h for x, y in pts):
            return 1.0

        # Quick check: completely off-screen
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) < 0.0 or min(xs) > w or max(ys) < 0.0 or min(ys) > h:
            return 0.0

        # Sutherland-Hodgman polygon clipping against viewport [0, w] x [0, h]
        clipped = list(pts)
        for edge in ["left", "right", "top", "bottom"]:
            if not clipped:
                return 0.0
            clipped = self._clip_polygon_edge(clipped, edge, w, h)

        if len(clipped) < 3:
            return 0.0

        area_orig = self._polygon_area(pts)
        if area_orig <= 1e-5:
            return 0.0

        area_clip = self._polygon_area(clipped)
        return float(min(1.0, max(0.0, area_clip / area_orig)))

    @staticmethod
    def _polygon_area(pts: list[tuple[float, float]]) -> float:
        """Shoelace formula for polygon area."""
        n = len(pts)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _clip_polygon_edge(
        pts: list[tuple[float, float]],
        edge: str,
        w: float,
        h: float,
    ) -> list[tuple[float, float]]:
        """Clip polygon against a single rectangular boundary edge."""
        def is_inside(p: tuple[float, float]) -> bool:
            if edge == "left":
                return p[0] >= 0.0
            elif edge == "right":
                return p[0] <= w
            elif edge == "top":
                return p[1] >= 0.0
            else:  # bottom
                return p[1] <= h

        def intersection(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if edge == "left":
                t = (0.0 - p1[0]) / (dx if abs(dx) > 1e-7 else 1e-7)
                return 0.0, p1[1] + t * dy
            elif edge == "right":
                t = (w - p1[0]) / (dx if abs(dx) > 1e-7 else 1e-7)
                return w, p1[1] + t * dy
            elif edge == "top":
                t = (0.0 - p1[1]) / (dy if abs(dy) > 1e-7 else 1e-7)
                return p1[0] + t * dx, 0.0
            else:
                t = (h - p1[1]) / (dy if abs(dy) > 1e-7 else 1e-7)
                return p1[0] + t * dx, h

        out: list[tuple[float, float]] = []
        if not pts:
            return out

        s = pts[-1]
        for p in pts:
            if is_inside(p):
                if is_inside(s):
                    out.append(p)
                else:
                    out.append(intersection(s, p))
                    out.append(p)
            elif is_inside(s):
                out.append(intersection(s, p))
            s = p
        return out
