"""Continuous 3D trajectory models for vehicle motion scenarios."""

from abc import ABC, abstractmethod
import math
import random
from typing import Any
import numpy as np

from vpocr_synth.motion.models import Pose3D, ScenarioType


class BaseTrajectory(ABC):
    """Abstract continuous 3D trajectory evaluable at arbitrary time t."""

    def __init__(self, duration_s: float, seed: int) -> None:
        self.duration_s = duration_s
        self.seed = seed
        self.rng = random.Random(seed)

        # Micro-vibrations from road unevenness and suspension dynamics
        self.pitch_amp_deg = self.rng.uniform(0.2, 0.6)
        self.pitch_freq_hz = self.rng.uniform(1.5, 3.5)
        self.pitch_phase = self.rng.uniform(0, 2 * math.pi)

        self.roll_amp_deg = self.rng.uniform(0.15, 0.45)
        self.roll_freq_hz = self.rng.uniform(1.2, 2.8)
        self.roll_phase = self.rng.uniform(0, 2 * math.pi)

        self.yaw_wobble_deg = self.rng.uniform(0.2, 0.7)
        self.yaw_wobble_freq = self.rng.uniform(0.8, 1.8)
        self.yaw_phase = self.rng.uniform(0, 2 * math.pi)

        # Plate mounting height in meters
        self.plate_height_m = self.rng.uniform(0.38, 0.68)

    @abstractmethod
    def _evaluate_nominal(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute nominal (x, y, z), heading_deg, and velocity [vx, vy, vz] at time t."""
        ...

    def evaluate(self, t: float) -> Pose3D:
        """Evaluate full continuous pose at arbitrary time t (seconds)."""
        # Clamp t for nominal evaluation but allow micro sub-frame evaluations
        t_eval = max(0.0, min(self.duration_s, t))
        nom_pos, nom_yaw, nom_vel = self._evaluate_nominal(t_eval)

        # Add road surface micro-motions
        speed = float(np.linalg.norm(nom_vel))
        speed_factor = min(1.0, speed / 2.0)  # less oscillation when stationary

        pitch_deg = math.sin(t * 2 * math.pi * self.pitch_freq_hz + self.pitch_phase) * self.pitch_amp_deg * speed_factor
        roll_deg = math.sin(t * 2 * math.pi * self.roll_freq_hz + self.roll_phase) * self.roll_amp_deg * speed_factor
        yaw_deg = nom_yaw + math.sin(t * 2 * math.pi * self.yaw_wobble_freq + self.yaw_phase) * self.yaw_wobble_deg * speed_factor

        return Pose3D(
            position=nom_pos,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            velocity=nom_vel,
        )


class StraightApproachTrajectory(BaseTrajectory):
    """Vehicle approaches along lane toward camera gate at steady speed or gentle deceleration."""

    def __init__(self, duration_s: float, seed: int) -> None:
        super().__init__(duration_s, seed)
        # Approaches into LPR recommended range (120..480px)
        self.y0 = self.rng.uniform(12.0, 18.0)
        self.v0 = self.rng.uniform(2.5, 5.0)  # ~9 to 18 km/h
        self.accel = self.rng.uniform(-0.5, 0.1)
        self.x0 = self.rng.uniform(-0.35, 0.35)
        self.heading_offset = self.rng.uniform(-2.5, 2.5)

    def _evaluate_nominal(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        v_current = max(0.5, self.v0 + self.accel * t)
        y = max(2.2, self.y0 - (self.v0 * t + 0.5 * self.accel * t * t))
        x = self.x0 + math.sin(t * 0.5) * 0.04
        z = self.plate_height_m
        vel = np.array([0.0, -v_current, 0.0])
        return np.array([x, y, z]), self.heading_offset, vel


class DecelerateAndStopTrajectory(BaseTrajectory):
    """Vehicle approaches and smoothly brakes to a complete stop at gate."""

    def __init__(self, duration_s: float, seed: int) -> None:
        super().__init__(duration_s, seed)
        self.stop_time = self.duration_s * self.rng.uniform(0.60, 0.85)
        # Stops at gate line (distance 2.8m to 4.5m from camera, prime LPR zone)
        self.y_stop = self.rng.uniform(2.8, 4.5)
        self.v0 = self.rng.uniform(3.0, 5.5)  # 11 to 20 km/h
        d_decel = 0.5 * self.v0 * self.stop_time
        self.y0 = self.y_stop + d_decel
        self.decel = self.v0 / self.stop_time
        self.x0 = self.rng.uniform(-0.3, 0.3)
        self.heading_offset = self.rng.uniform(-2.0, 2.0)

    def _evaluate_nominal(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        if t <= self.stop_time:
            d = self.v0 * t - 0.5 * self.decel * t * t
            y = self.y0 - d
            v = self.v0 - self.decel * t
            vel = np.array([0.0, -v, 0.0])
        else:
            y = self.y_stop
            vel = np.zeros(3)

        return np.array([self.x0, y, self.plate_height_m]), self.heading_offset, vel


class StopAndGoTrajectory(BaseTrajectory):
    """Vehicle begins stationary at gate, waits, then accelerates forward."""

    def __init__(self, duration_s: float, seed: int) -> None:
        super().__init__(duration_s, seed)
        self.wait_time = self.duration_s * self.rng.uniform(0.25, 0.45)
        self.y_start = self.rng.uniform(3.5, 5.2)
        self.target_accel = self.rng.uniform(0.8, 1.8)  # m/s^2
        self.x0 = self.rng.uniform(-0.25, 0.25)
        self.heading_offset = self.rng.uniform(-2.0, 2.0)

    def _evaluate_nominal(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        if t <= self.wait_time:
            y = self.y_start
            vel = np.zeros(3)
        else:
            dt = t - self.wait_time
            v = self.target_accel * dt
            d = 0.5 * self.target_accel * dt * dt
            y = max(1.8, self.y_start - d)
            vel = np.array([0.0, -v, 0.0])

        return np.array([self.x0, y, self.plate_height_m]), self.heading_offset, vel


class CurvingEntryTrajectory(BaseTrajectory):
    """Vehicle enters from an angled approach road, turning into gate lane."""

    def __init__(self, duration_s: float, seed: int) -> None:
        super().__init__(duration_s, seed)
        self.turn_sign = self.rng.choice([-1.0, 1.0])
        self.init_yaw = self.turn_sign * self.rng.uniform(16.0, 32.0)
        self.init_x = self.turn_sign * self.rng.uniform(2.2, 3.8)
        self.init_y = self.rng.uniform(12.0, 17.0)
        self.final_y = self.rng.uniform(2.8, 4.5)
        self.final_x = self.turn_sign * self.rng.uniform(0.0, 0.25)
        self.speed = self.rng.uniform(2.5, 4.2)  # m/s

    def _evaluate_nominal(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        progress = min(1.0, t / self.duration_s)
        s = progress * progress * (3.0 - 2.0 * progress)

        x = self.init_x + s * (self.final_x - self.init_x)
        y = self.init_y + s * (self.final_y - self.init_y)
        yaw = self.init_yaw * (1.0 - s)

        vx = (self.final_x - self.init_x) / self.duration_s * (6.0 * progress * (1.0 - progress))
        vy = (self.final_y - self.init_y) / self.duration_s * (6.0 * progress * (1.0 - progress))
        vel = np.array([vx, vy, 0.0])

        return np.array([x, y, self.plate_height_m]), yaw, vel


class SwitchbackReversingTrajectory(BaseTrajectory):
    """Vehicle stops before gate, shifts to reverse, backs up slightly, then stops."""

    def __init__(self, duration_s: float, seed: int) -> None:
        super().__init__(duration_s, seed)
        self.y_front = self.rng.uniform(2.8, 4.0)
        self.backup_dist = self.rng.uniform(1.2, 2.0)
        self.t_stop1 = self.duration_s * 0.25
        self.t_rev_end = self.duration_s * 0.75
        self.x0 = self.rng.uniform(-0.25, 0.25)
        self.heading_offset = self.rng.uniform(-3.5, 3.5)

    def _evaluate_nominal(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        if t <= self.t_stop1:
            y = self.y_front
            vel = np.zeros(3)
        elif t <= self.t_rev_end:
            dt = t - self.t_stop1
            rev_duration = self.t_rev_end - self.t_stop1
            prog = dt / rev_duration
            s = prog * prog * (3.0 - 2.0 * prog)
            y = self.y_front + s * self.backup_dist
            vy = (self.backup_dist / rev_duration) * (6.0 * prog * (1.0 - prog))
            vel = np.array([0.0, vy, 0.0])
        else:
            y = self.y_front + self.backup_dist
            vel = np.zeros(3)

        return np.array([self.x0, y, self.plate_height_m]), self.heading_offset, vel


class PassThroughRearTrajectory(BaseTrajectory):
    """Camera views rear plate as vehicle departs gate into site."""

    def __init__(self, duration_s: float, seed: int) -> None:
        super().__init__(duration_s, seed)
        self.y0 = self.rng.uniform(2.5, 4.5)
        self.v0 = self.rng.uniform(2.2, 4.8)  # m/s
        self.accel = self.rng.uniform(0.4, 1.2)
        self.x0 = self.rng.uniform(-0.25, 0.25)
        # Rear plate faces backward toward camera at Y=0 (nominal yaw 0 deg toward camera)
        self.heading_offset = self.rng.uniform(-2.5, 2.5)
        self.plate_height_m = self.rng.uniform(0.45, 0.72)

    def _evaluate_nominal(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        dist_traveled = self.v0 * t + 0.5 * self.accel * t * t
        y = self.y0 + dist_traveled
        v = self.v0 + self.accel * t
        vel = np.array([0.0, v, 0.0])
        return np.array([self.x0, y, self.plate_height_m]), self.heading_offset, vel


def create_trajectory(scenario: ScenarioType, duration_s: float, seed: int) -> BaseTrajectory:
    """Factory function to instantiate continuous trajectory model by scenario."""
    match scenario:
        case ScenarioType.STRAIGHT_APPROACH:
            return StraightApproachTrajectory(duration_s, seed)
        case ScenarioType.DECELERATE_AND_STOP:
            return DecelerateAndStopTrajectory(duration_s, seed)
        case ScenarioType.STOP_AND_GO:
            return StopAndGoTrajectory(duration_s, seed)
        case ScenarioType.CURVING_ENTRY:
            return CurvingEntryTrajectory(duration_s, seed)
        case ScenarioType.SWITCHBACK_REVERSING:
            return SwitchbackReversingTrajectory(duration_s, seed)
        case ScenarioType.PASS_THROUGH_REAR:
            return PassThroughRearTrajectory(duration_s, seed)
