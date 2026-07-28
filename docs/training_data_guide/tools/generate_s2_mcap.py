"""
generate_s2_mcap.py — Sinh MỘT file `.mcap` chứa ĐỦ dữ liệu để dựng data train System 2.

=============================================================================
VÌ SAO CÓ FILE NÀY?
-----------------------------------------------------------------------------
File `MCAP/demo_robot.mcap` (bản thử nghiệm) chỉ có `pose / imu / battery / log`
→ KHÔNG có thị giác → KHÔNG thể sinh data System 2.

Script này sinh một `.mcap` "đúng chuẩn đầu vào" của pipeline S2: đủ 5 luồng mà
`mcap2s2.py` cần, mỗi luồng đúng vai trò của nó trong data LeRobot cuối cùng:

  | Topic                       | Schema (Foxglove)        | Sinh ra cái gì trong data S2                |
  |-----------------------------|--------------------------|---------------------------------------------|
  | /camera/front/image_raw     | foxglove.CompressedImage | observation.images.rgb.{H}cm_{pitch_1}deg   |
  | /camera/lookdown/image_raw  | foxglove.CompressedImage | observation.images.rgb.{H}cm_{pitch_2}deg   |
  | /camera/lookdown/depth      | foxglove.CompressedImage | observation.images.depth.{H}cm_{pitch_2}deg |
  | /camera/*/camera_info       | foxglove.CameraCalibration | ma trận K → dùng để CHIẾU pixel-goal      |
  | /robot/pose                 | foxglove.PoseInFrame     | pose.{setting} + action + waypoint 3D       |
  | /task/episode               | (custom JSON)            | meta/episodes.jsonl (câu lệnh + ranh giới)  |

Dữ liệu được sinh bằng một "simulator tí hon": một căn phòng hình hộp có sàn kẻ
ô, 4 bức tường và vài cây cột. Ảnh RGB + ảnh depth được **ray-trace** đúng theo
pose camera, nên ảnh, depth và pose *khớp nhau về hình học* — điều kiện bắt buộc
để bước chiếu pixel-goal ở `mcap2s2.py` cho kết quả đúng.

=============================================================================
QUY ƯỚC HÌNH HỌC (bám đúng data gốc InternData-N1)
-----------------------------------------------------------------------------
- Hệ world: x = trước, y = trái, z = lên (ENU kiểu robot). Sàn ở z = 0.
- Hệ robot (base): x trước, y trái, z lên; base luôn ở z = 0.
- Hệ camera: **quy ước OpenCV** — x phải, y xuống, z là trục quang (nhìn ra trước).
- `pose.{setting}` trong parquet = ma trận 4×4 **camera → world** (cột 4 = vị trí
  camera trong world). Đã đối chiếu số thật của `vln_ce/r2r/17DRP5sb8fy`:
  camera cao 0.6 m, cúi 30° cho đúng ma trận
      [[-0, -0.5, 0.866, 0], [-1, 0, -0, 0], [0, -0.866, -0.5, 0.6], [0,0,0,1]]
  (cột 3 = trục quang = (cos30, 0, -sin30) → cúi 30°; cột 1 = (0,-1,0) = "phải"
  của camera = -y robot). Script này dựng đúng công thức đó.
- Chuyển động rời rạc kiểu R2R: **tiến 0.25 m** hoặc **xoay 15°** mỗi bước
  (đã kiểm chứng bằng cách chạy `get_trajectory_relative_to_frame` trên data thật).

=============================================================================
CÁCH CHẠY
-----------------------------------------------------------------------------
    pip install mcap numpy pillow
    python generate_s2_mcap.py --out demo_s2_robot.mcap

Đổi cấu hình camera (mặc định `125cm_0_30` = 2 camera trên đầu robot hình người):
    python generate_s2_mcap.py --height-cm 60 --pitch1 30 --pitch2 30   # 1 góc

Xem lại file vừa sinh:
    python generate_s2_mcap.py --inspect demo_s2_robot.mcap
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from mcap.writer import CompressionType, Writer
from PIL import Image

# Console Windows mặc định là cp1252 → in tiếng Việt sẽ lỗi. Ép UTF-8 cho chắc.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

NS = 1_000_000_000  # 1 giây = 1e9 nanosecond

# Bước đi rời rạc — giống benchmark R2R mà InternData-N1 dùng
STEP_FORWARD_M = 0.25
STEP_TURN_DEG = 15.0

DEPTH_MAX_MM = 10000  # data gốc clip depth ở 10 m


# =============================================================================
# PHẦN 1 — THẾ GIỚI 3D TÍ HON (để render ảnh + depth)
# =============================================================================
@dataclass
class World:
    """Một căn phòng hình hộp + vài cây cột. Đủ để ảnh có kết cấu và depth có chiều sâu."""

    x_min: float = -2.0
    x_max: float = 16.0
    y_min: float = -5.0
    y_max: float = 5.0
    wall_h: float = 2.6
    # (x, y, bán kính, màu RGB)
    pillars: Tuple[Tuple[float, float, float, Tuple[int, int, int]], ...] = (
        (3.0, 1.2, 0.22, (198, 88, 72)),
        (6.5, -1.4, 0.22, (80, 140, 198)),
        (10.0, 1.6, 0.25, (206, 170, 70)),
        (12.5, -2.2, 0.20, (110, 178, 120)),
    )
    # Màu 4 bức tường: x_min, x_max, y_min, y_max
    wall_colors: Tuple[Tuple[int, int, int], ...] = (
        (150, 150, 158),
        (168, 158, 140),
        (140, 152, 160),
        (158, 148, 152),
    )


def _checker_floor_color(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Sàn kẻ ô 1 m + một dải thảm đỏ chạy dọc hành lang (làm mốc nhìn cho dễ)."""
    ix = np.floor(px).astype(np.int64)
    iy = np.floor(py).astype(np.int64)
    light = ((ix + iy) % 2) == 0

    color = np.empty(px.shape + (3,), dtype=np.float64)
    color[..., 0] = np.where(light, 186.0, 148.0)
    color[..., 1] = np.where(light, 180.0, 142.0)
    color[..., 2] = np.where(light, 170.0, 134.0)

    carpet = np.abs(py) < 0.6  # dải thảm giữa hành lang
    color[carpet] = np.array([150.0, 78.0, 70.0])
    return color


def render_rgbd(cam_pose: np.ndarray, K: np.ndarray, width: int, height: int, world: World):
    """Ray-trace một khung hình.

    Mẹo quan trọng: tia trong hệ camera lấy dạng d_cam = (x_n, y_n, 1) — KHÔNG chuẩn hoá.
    Nhờ vậy tham số giao điểm `t` **chính là depth** (toạ độ Z trong hệ camera), đúng
    định nghĩa depth của ảnh RGB-D. Không cần bước quy đổi nào nữa.

    Trả về: (rgb uint8 [H,W,3], depth_mm uint16 [H,W])
    """
    # Tia song song với một mặt phẳng → chia cho 0 → inf/NaN. Đó là chuyện BÌNH THƯỜNG
    # (mọi giao điểm hỏng đều bị mask loại ở dưới), nên tắt cảnh báo cho đỡ nhiễu log.
    prev_err = np.seterr(all="ignore")
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    xn = (uu - K[0, 2]) / K[0, 0]
    yn = (vv - K[1, 2]) / K[1, 1]
    d_cam = np.stack([xn, yn, np.ones_like(xn)], axis=-1)  # (H, W, 3)

    R = cam_pose[:3, :3]
    C = cam_pose[:3, 3]
    d_w = d_cam @ R.T  # hướng tia trong world

    INF = 1e9
    best_t = np.full((height, width), INF)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    rgb[:] = np.array([224.0, 228.0, 236.0])  # nền (không trúng gì) — coi như trần/khoảng trống

    def _update(t: np.ndarray, hit: np.ndarray, color: np.ndarray):
        """Ghi nhận bề mặt nếu nó gần hơn bề mặt đang giữ."""
        better = hit & (t > 1e-3) & (t < best_t)
        best_t[better] = t[better]
        rgb[better] = color[better] if color.ndim == 3 else color

    # --- (a) sàn: z = 0 ---
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.0 - C[2]) / d_w[..., 2]
    px = C[0] + t * d_w[..., 0]
    py = C[1] + t * d_w[..., 1]
    hit = np.isfinite(t) & (px > world.x_min) & (px < world.x_max) & (py > world.y_min) & (py < world.y_max)
    _update(t, hit, _checker_floor_color(px, py))

    # --- (b) 4 bức tường ---
    walls = [
        (0, world.x_min, world.wall_colors[0]),
        (0, world.x_max, world.wall_colors[1]),
        (1, world.y_min, world.wall_colors[2]),
        (1, world.y_max, world.wall_colors[3]),
    ]
    for axis, offset, base_color in walls:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (offset - C[axis]) / d_w[..., axis]
        pz = C[2] + t * d_w[..., 2]
        other = 1 - axis
        po = C[other] + t * d_w[..., other]
        lo, hi = (world.y_min, world.y_max) if other == 1 else (world.x_min, world.x_max)
        hit = np.isfinite(t) & (pz > 0.0) & (pz < world.wall_h) & (po > lo) & (po < hi)
        # sọc ngang 0.5 m cho tường có kết cấu (để model/mắt người thấy được chuyển động)
        stripe = ((np.floor(pz / 0.5).astype(np.int64) + np.floor(po / 1.0).astype(np.int64)) % 2) == 0
        color = np.empty((height, width, 3))
        color[:] = np.array(base_color, dtype=np.float64)
        color[stripe] *= 0.86
        _update(t, hit, color)

    # --- (c) các cây cột (hình trụ đứng) ---
    for cx, cy, radius, pcolor in world.pillars:
        ox = C[0] - cx
        oy = C[1] - cy
        a = d_w[..., 0] ** 2 + d_w[..., 1] ** 2
        b = 2.0 * (ox * d_w[..., 0] + oy * d_w[..., 1])
        c_ = ox**2 + oy**2 - radius**2
        disc = b**2 - 4 * a * c_
        ok = disc > 0
        sqrt_disc = np.sqrt(np.where(ok, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-b - sqrt_disc) / (2 * a)
        pz = C[2] + t * d_w[..., 2]
        hit = ok & np.isfinite(t) & (pz > 0.0) & (pz < 2.2)
        color = np.empty((height, width, 3))
        color[:] = np.array(pcolor, dtype=np.float64)
        _update(t, hit, color)

    # --- (d) đóng gói kết quả ---
    depth_m = np.where(best_t < INF, best_t, 0.0)
    depth_mm = np.clip(depth_m * 1000.0, 0, DEPTH_MAX_MM).astype(np.uint16)

    # Tô bóng nhẹ theo khoảng cách để ảnh trông tự nhiên hơn (xa thì mờ dần)
    fog = np.clip(depth_m / 14.0, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - 0.45 * fog) + np.array([225.0, 229.0, 236.0]) * (0.45 * fog)
    np.seterr(**prev_err)
    return np.clip(rgb, 0, 255).astype(np.uint8), depth_mm


# =============================================================================
# PHẦN 2 — HÌNH HỌC: pose robot → pose camera (4×4 camera→world)
# =============================================================================
def camera_pose_from_base(x: float, y: float, yaw: float, height_m: float, pitch_deg: float) -> np.ndarray:
    """Dựng ma trận 4×4 camera→world theo ĐÚNG quy ước của `pose.{setting}` trong vln_ce.

    - Vị trí camera = (x, y, height_m).
    - Trục quang (cột 3) = R_yaw · (cos p, 0, -sin p)  → cúi xuống p độ.
    - Trục "phải" (cột 1) = R_yaw · (0, -1, 0)          → phải của camera = -y robot.
    - Trục "xuống"(cột 2) = z_cam × x_cam                → giữ hệ thuận tay phải OpenCV.
    """
    p = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

    z_cam = Rz @ np.array([math.cos(p), 0.0, -math.sin(p)])
    x_cam = Rz @ np.array([0.0, -1.0, 0.0])
    y_cam = np.cross(z_cam, x_cam)

    T = np.eye(4)
    T[:3, 0] = x_cam
    T[:3, 1] = y_cam
    T[:3, 2] = z_cam
    T[:3, 3] = np.array([x, y, height_m])
    return T


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) của phép xoay quanh trục z — dùng cho topic pose."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Ma trận K 3×3 từ góc nhìn ngang. Mặc định 79° ≈ Intel RealSense D435i (kênh màu)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


# =============================================================================
# PHẦN 3 — LỘ TRÌNH: chuỗi lệnh rời rạc → chuỗi pose từng frame
# =============================================================================
@dataclass
class Episode:
    instruction: str
    start: Tuple[float, float, float]  # (x, y, yaw_deg)
    route: List[Tuple[str, int]] = field(default_factory=list)  # [('F', 6), ('R', 90), ...]


DEFAULT_EPISODES: List[Episode] = [
    Episode(
        instruction=(
            "Walk straight down the hallway past the red pillar, turn right at the blue pillar, "
            "then continue forward and stop next to the yellow pillar."
        ),
        start=(0.0, 0.0, 0.0),
        route=[("F", 10), ("R", 45), ("F", 6), ("L", 45), ("F", 10), ("R", 30), ("F", 5)],
    ),
    Episode(
        instruction=(
            "Turn left and walk along the red carpet until you reach the far wall, "
            "then stop in front of the green pillar."
        ),
        start=(1.0, -3.0, 30.0),
        route=[("L", 60), ("F", 8), ("R", 75), ("F", 12), ("L", 30), ("F", 6)],
    ),
]


def expand_route(ep: Episode) -> List[Tuple[float, float, float]]:
    """Bung chuỗi lệnh thành danh sách pose (x, y, yaw) — MỖI BƯỚC LÀ MỘT FRAME.

    Frame 0 = tư thế xuất phát (chưa làm gì) → khớp quy ước `action[0] = -1` của vln_ce.
    """
    x, y, yaw = ep.start[0], ep.start[1], math.radians(ep.start[2])
    poses = [(x, y, yaw)]
    for cmd, amount in ep.route:
        if cmd == "F":
            for _ in range(amount):
                x += STEP_FORWARD_M * math.cos(yaw)
                y += STEP_FORWARD_M * math.sin(yaw)
                poses.append((x, y, yaw))
        elif cmd in ("L", "R"):
            n_turn = int(round(amount / STEP_TURN_DEG))
            sign = 1.0 if cmd == "L" else -1.0
            for _ in range(n_turn):
                yaw += sign * math.radians(STEP_TURN_DEG)
                poses.append((x, y, yaw))
        else:
            raise ValueError(f"Lệnh lộ trình không hiểu: {cmd}")
    return poses


# =============================================================================
# PHẦN 4 — SCHEMA MESSAGE (JSON, theo bộ schema công khai của Foxglove)
# =============================================================================
TIME_SCHEMA = {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}}

COMPRESSED_IMAGE_SCHEMA = {
    "type": "object",
    "title": "foxglove.CompressedImage",
    "properties": {
        "timestamp": TIME_SCHEMA,
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
}

CAMERA_CALIBRATION_SCHEMA = {
    "type": "object",
    "title": "foxglove.CameraCalibration",
    "properties": {
        "timestamp": TIME_SCHEMA,
        "frame_id": {"type": "string"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "distortion_model": {"type": "string"},
        "D": {"type": "array", "items": {"type": "number"}},
        "K": {"type": "array", "items": {"type": "number"}, "minItems": 9, "maxItems": 9},
        "R": {"type": "array", "items": {"type": "number"}, "minItems": 9, "maxItems": 9},
        "P": {"type": "array", "items": {"type": "number"}, "minItems": 12, "maxItems": 12},
    },
}

POSE_IN_FRAME_SCHEMA = {
    "type": "object",
    "title": "foxglove.PoseInFrame",
    "properties": {
        "timestamp": TIME_SCHEMA,
        "frame_id": {"type": "string"},
        "pose": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                },
                "orientation": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "w": {"type": "number"},
                    },
                },
            },
        },
    },
}

# Không có schema chuẩn nào cho "câu lệnh điều hướng" → tự định nghĩa.
EPISODE_SCHEMA = {
    "type": "object",
    "title": "vlnbot.EpisodeMarker",
    "properties": {
        "event": {"type": "string", "enum": ["start", "end"]},
        "episode_id": {"type": "integer"},
        "instruction": {"type": "string"},
    },
}

# Thông số lắp camera — thứ QUYẾT ĐỊNH tên `setting` của data S2.
MOUNT_SCHEMA = {
    "type": "object",
    "title": "vlnbot.CameraMount",
    "properties": {
        "camera": {"type": "string"},
        "height_cm": {"type": "number"},
        "pitch_deg": {"type": "number"},
        "role": {"type": "string", "enum": ["pitch_1", "pitch_2"]},
    },
}


def _stamp(ns: int) -> dict:
    return {"sec": ns // NS, "nsec": ns % NS}


def _jpeg_b64(rgb: np.ndarray, quality: int = 88) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png16_b64(depth_mm: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(depth_mm.astype(np.uint16)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# =============================================================================
# PHẦN 5 — GHI FILE MCAP
# =============================================================================
def build(args) -> None:
    world = World()
    K = intrinsics(args.width, args.height, args.hfov)
    cam_h = args.height_cm / 100.0
    t0 = time.time_ns()

    # --- chuẩn bị các kênh (channel) ---
    with open(args.out, "wb") as f:
        writer = Writer(f, compression=CompressionType.ZSTD)
        writer.start(profile="", library="generate_s2_mcap")

        def channel(topic: str, schema_name: str, schema: dict) -> int:
            sid = writer.register_schema(
                name=schema_name, encoding="jsonschema", data=json.dumps(schema).encode("utf-8")
            )
            return writer.register_channel(topic=topic, message_encoding="json", schema_id=sid)

        ch = {
            "front_rgb": channel("/camera/front/image_raw", "foxglove.CompressedImage", COMPRESSED_IMAGE_SCHEMA),
            "front_info": channel("/camera/front/camera_info", "foxglove.CameraCalibration", CAMERA_CALIBRATION_SCHEMA),
            "down_rgb": channel("/camera/lookdown/image_raw", "foxglove.CompressedImage", COMPRESSED_IMAGE_SCHEMA),
            "down_depth": channel("/camera/lookdown/depth", "foxglove.CompressedImage", COMPRESSED_IMAGE_SCHEMA),
            "down_info": channel(
                "/camera/lookdown/camera_info", "foxglove.CameraCalibration", CAMERA_CALIBRATION_SCHEMA
            ),
            "pose": channel("/robot/pose", "foxglove.PoseInFrame", POSE_IN_FRAME_SCHEMA),
            "episode": channel("/task/episode", "vlnbot.EpisodeMarker", EPISODE_SCHEMA),
            "mount": channel("/robot/camera_mount", "vlnbot.CameraMount", MOUNT_SCHEMA),
        }

        def emit(key: str, ns: int, payload: dict) -> None:
            writer.add_message(
                channel_id=ch[key], log_time=ns, publish_time=ns, data=json.dumps(payload).encode("utf-8")
            )

        # --- thông số lắp camera: ghi 1 lần ở đầu file ---
        for role, cam_name, pitch in (
            ("pitch_1", "front", args.pitch1),
            ("pitch_2", "lookdown", args.pitch2),
        ):
            emit("mount", t0, {"camera": cam_name, "height_cm": args.height_cm, "pitch_deg": pitch, "role": role})

        K_flat = [float(v) for v in K.reshape(-1)]
        calib_common = {
            "width": args.width,
            "height": args.height,
            "distortion_model": "plumb_bob",
            "D": [0.0] * 5,
            "K": K_flat,
            "R": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
            "P": K_flat[:3] + [0.0] + K_flat[3:6] + [0.0] + K_flat[6:9] + [0.0],
        }

        n_frames_total = 0
        clock_ns = t0
        dt_ns = int(NS / args.fps)

        for ep_id, ep in enumerate(args.episodes[: args.num_episodes]):
            frames = expand_route(ep)
            ep_start_ns = clock_ns
            emit("episode", ep_start_ns, {"event": "start", "episode_id": ep_id, "instruction": ep.instruction})

            for i, (x, y, yaw) in enumerate(frames):
                ns = ep_start_ns + i * dt_ns

                # (1) POSE @ 50 Hz — phát dày hơn ảnh để pipeline PHẢI làm bước đồng bộ thời gian.
                #     Giữa 2 frame ảnh, nội suy tuyến tính pose.
                if i + 1 < len(frames):
                    nx, ny, nyaw = frames[i + 1]
                else:
                    nx, ny, nyaw = x, y, yaw
                d_yaw = math.atan2(math.sin(nyaw - yaw), math.cos(nyaw - yaw))
                n_sub = max(1, int(round(50.0 / args.fps)))
                for s in range(n_sub):
                    a = s / n_sub
                    sx, sy_, syaw = x + a * (nx - x), y + a * (ny - y), yaw + a * d_yaw
                    qx, qy, qz, qw = yaw_to_quaternion(syaw)
                    emit(
                        "pose",
                        ns + int(s * dt_ns / n_sub),
                        {
                            "timestamp": _stamp(ns + int(s * dt_ns / n_sub)),
                            "frame_id": "map",
                            "pose": {
                                "position": {"x": sx, "y": sy_, "z": 0.0},
                                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                            },
                        },
                    )

                # (2) ẢNH: render góc cúi (pitch_2, kèm depth) trước, rồi góc thẳng (pitch_1).
                #     Nếu hai góc bằng nhau (cấu hình 1 camera như 60cm_30_30) thì dùng lại
                #     đúng một lần render — vừa nhanh vừa phản ánh đúng thực tế phần cứng.
                pose_down = camera_pose_from_base(x, y, yaw, cam_h, args.pitch2)
                rgb_down, depth_down = render_rgbd(pose_down, K, args.width, args.height, world)

                if abs(args.pitch2 - args.pitch1) < 1e-6:
                    rgb_front = rgb_down
                else:
                    pose_front = camera_pose_from_base(x, y, yaw, cam_h, args.pitch1)
                    rgb_front, _ = render_rgbd(pose_front, K, args.width, args.height, world)

                emit(
                    "front_rgb",
                    ns,
                    {
                        "timestamp": _stamp(ns),
                        "frame_id": "camera_front",
                        "format": "jpeg",
                        "data": _jpeg_b64(rgb_front, args.jpeg_quality),
                    },
                )
                emit(
                    "down_rgb",
                    ns,
                    {
                        "timestamp": _stamp(ns),
                        "frame_id": "camera_lookdown",
                        "format": "jpeg",
                        "data": _jpeg_b64(rgb_down, args.jpeg_quality),
                    },
                )
                emit(
                    "down_depth",
                    ns,
                    {
                        "timestamp": _stamp(ns),
                        "frame_id": "camera_lookdown",
                        "format": "png",  # PNG 16-bit, đơn vị MILIMÉT
                        "data": _png16_b64(depth_down),
                    },
                )

                # (3) camera_info @ ~1 Hz (thực tế robot cũng phát thưa như vậy)
                if i % max(1, int(args.fps)) == 0:
                    for key, frame_id in (("front_info", "camera_front"), ("down_info", "camera_lookdown")):
                        emit(key, ns, {"timestamp": _stamp(ns), "frame_id": frame_id, **calib_common})

                n_frames_total += 1

            ep_end_ns = ep_start_ns + (len(frames) - 1) * dt_ns
            emit("episode", ep_end_ns, {"event": "end", "episode_id": ep_id, "instruction": ep.instruction})
            # nghỉ 2 giây giữa 2 episode (pipeline phải biết bỏ qua khoảng trống này)
            clock_ns = ep_end_ns + 2 * NS
            print(f"  episode {ep_id}: {len(frames)} frames — \"{ep.instruction[:60]}...\"")

        # --- metadata: "hồ sơ" để pipeline đọc mà không cần đoán ---
        writer.add_metadata(
            name="s2_profile",
            data={
                "height_cm": str(args.height_cm),
                "pitch_1": str(args.pitch1),
                "pitch_2": str(args.pitch2),
                "setting": f"{int(args.height_cm)}cm_{int(args.pitch2)}deg",
                "image_width": str(args.width),
                "image_height": str(args.height),
                "fps": str(args.fps),
                "depth_unit": "millimeter",
                "world_frame": "x-forward, y-left, z-up",
                "camera_convention": "OpenCV (x-right, y-down, z-forward)",
            },
        )
        writer.finish()

    size_mb = os.path.getsize(args.out) / 1e6
    print(
        f"\n✅ Đã tạo {args.out} — {n_frames_total} frame ảnh, "
        f"{size_mb:.1f} MB, setting = {int(args.height_cm)}cm_{int(args.pitch2)}deg"
    )


# =============================================================================
# PHẦN 6 — XEM LẠI FILE (kiểm tra nhanh, không cần Foxglove)
# =============================================================================
def inspect(path: str) -> None:
    from mcap.reader import make_reader

    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        stats = summary.statistics
        print(f"=== {path} ===")
        print(f"Tổng message : {stats.message_count}")
        print(f"Thời lượng   : {(stats.message_end_time - stats.message_start_time) / 1e9:.2f} s\n")
        print(f"{'Topic':<32}{'Schema':<28}{'Số message'}")
        print("-" * 74)
        for cid, c in summary.channels.items():
            print(f"{c.topic:<32}{summary.schemas[c.schema_id].name:<28}{stats.channel_message_counts.get(cid, 0)}")
        f.seek(0)
        for record in reader.iter_metadata():
            print(f"\nmetadata '{record.name}':")
            for k, v in record.metadata.items():
                print(f"   {k:<20} = {v}")
        f.seek(0)
        print("\n=== /task/episode ===")
        for _, _, msg in reader.iter_messages(topics=["/task/episode"]):
            p = json.loads(msg.data)
            print(f"  [{(msg.log_time - stats.message_start_time)/1e9:7.2f}s] {p['event']:<5} ep{p['episode_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_s2_robot.mcap", help="file .mcap sẽ tạo")
    ap.add_argument("--inspect", metavar="FILE", help="chỉ xem lại một file .mcap rồi thoát")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=79.0, help="góc nhìn ngang (độ); 79 ≈ RealSense D435i")
    ap.add_argument("--height-cm", type=float, default=125.0, help="chiều cao lắp camera (cm)")
    ap.add_argument("--pitch1", type=float, default=0.0, help="góc cúi camera 'nhìn thẳng' (độ)")
    ap.add_argument("--pitch2", type=float, default=30.0, help="góc cúi camera 'nhìn xuống' (độ)")
    ap.add_argument("--fps", type=float, default=10.0, help="tần số ảnh")
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--num-episodes", type=int, default=2)
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    args.episodes = DEFAULT_EPISODES
    print(f"Đang render {args.num_episodes} episode ở {args.width}×{args.height} ...")
    build(args)


if __name__ == "__main__":
    main()
