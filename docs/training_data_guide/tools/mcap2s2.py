"""
mcap2s2.py — Chuyển một file `.mcap` thành data train System 2 (định dạng LeRobot của `vln_ce`).

=============================================================================
ĐẦU VÀO → ĐẦU RA
-----------------------------------------------------------------------------
    demo_s2_robot.mcap                (log robot thật)
              │
              ▼
    traj_data/<dataset>/<scene_id>/
    ├── meta/{episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, info.json}
    ├── data/chunk-000/episode_000000.parquet
    └── videos/chunk-000/
        ├── observation.images.rgb.{H}cm_{pitch_1}deg/episode_000000_0.jpg
        ├── observation.images.rgb.{H}cm_{pitch_2}deg/episode_000000_0.jpg
        └── observation.images.depth.{H}cm_{pitch_2}deg/episode_000000_0.png

=============================================================================
SÁU GIAI ĐOẠN (mỗi giai đoạn là một hàm, in ra thống kê riêng)
-----------------------------------------------------------------------------
  A. read_mcap        — đọc mọi topic cần dùng, gom theo thời gian
  B. sync_frames      — đồng bộ ảnh/depth/pose theo timestamp + cắt episode + lọc keyframe
  C. make_labels      — sinh `action`, `pose` 4×4, `goal` (u,v), `relative_goal_frame_id`
  D. write_images     — ghi .jpg / .png đúng tên file loader mong đợi
  E. write_lerobot    — ghi parquet (đúng dtype) + 4 file meta
  F. self_check       — đọc lại, mô phỏng đúng logic cắt mẫu của loader để đếm mẫu

=============================================================================
CÁCH CHẠY
-----------------------------------------------------------------------------
    pip install mcap numpy pillow pyarrow
    python mcap2s2.py --mcap demo_s2_robot.mcap --out ./traj_data \
                      --dataset-name myrobot --scene-id demo_scene

Với mcap của robot thật, chỉ cần khai lại tên topic:
    python mcap2s2.py --mcap run_01.mcap \
        --topic-rgb-front /camera/color/image_raw \
        --topic-rgb-down  /camera_down/color/image_raw \
        --topic-depth     /camera_down/aligned_depth_to_color/image_raw \
        --topic-pose      /odom  --topic-caminfo /camera_down/color/camera_info \
        --height-cm 125 --pitch1 0 --pitch2 30 --instruction-file instructions.json

"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

NS = 1_000_000_000
IDX2NAME = {-1: "start", 0: "STOP", 1: "↑ tiến", 2: "← trái", 3: "→ phải", 5: "↓ cúi"}


# =============================================================================
# GIAI ĐOẠN A — ĐỌC MCAP
# =============================================================================
@dataclass
class RawStreams:
    rgb_front: List[Tuple[int, dict]]
    rgb_down: List[Tuple[int, dict]]
    depth: List[Tuple[int, dict]]
    pose: List[Tuple[int, dict]]
    caminfo: List[Tuple[int, dict]]
    episodes: List[Tuple[int, dict]]
    metadata: Dict[str, str]


def read_mcap(path: str, topics: Dict[str, str]) -> RawStreams:
    """Đọc mcap một lượt, gom message của từng topic thành list `(log_time_ns, payload)`.

    Ghi chú: file mcap tự mô tả (self-describing) nên KHÔNG cần ROS để đọc. Ở đây message
    được mã hoá JSON; nếu log của bạn là ROS 2 thật (`message_encoding='cdr'`), cài thêm
    `mcap-ros2-support` và thay `json.loads(msg.data)` bằng decoder tương ứng — phần còn
    lại của script giữ nguyên.
    """
    buckets: Dict[str, List[Tuple[int, dict]]] = {k: [] for k in topics}
    wanted = {v: k for k, v in topics.items()}
    metadata: Dict[str, str] = {}

    with open(path, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(topics=list(wanted.keys())):
            buckets[wanted[channel.topic]].append((message.log_time, json.loads(message.data)))
        f.seek(0)
        for record in reader.iter_metadata():
            if record.name == "s2_profile":
                metadata = dict(record.metadata)

    for v in buckets.values():
        v.sort(key=lambda m: m[0])

    print("── A. Đọc mcap ─────────────────────────────────────────")
    for name, topic in topics.items():
        print(f"   {topic:<34} {len(buckets[name]):>6} message")
    if metadata:
        print(f"   metadata s2_profile: {metadata}")
    return RawStreams(
        rgb_front=buckets["rgb_front"],
        rgb_down=buckets["rgb_down"],
        depth=buckets["depth"],
        pose=buckets["pose"],
        caminfo=buckets["caminfo"],
        episodes=buckets["episodes"],
        metadata=metadata,
    )


# =============================================================================
# GIAI ĐOẠN B — ĐỒNG BỘ THỜI GIAN + CẮT EPISODE
# =============================================================================
@dataclass
class Frame:
    t_ns: int
    rgb_front: bytes
    rgb_down: bytes
    depth_png: bytes
    x: float
    y: float
    yaw: float


@dataclass
class EpisodeData:
    instruction: str
    frames: List[Frame]


def _nearest(stream: List[Tuple[int, dict]], times: np.ndarray, t: int) -> Optional[Tuple[int, dict]]:
    """Tìm message có timestamp gần `t` nhất — tìm nhị phân trên mảng thời gian đã sắp xếp.

    (`times` được tính sẵn MỘT lần ở ngoài; nếu tính lại trong mỗi lần gọi thì với log thật
    hàng trăm nghìn message sẽ chậm bình phương.)
    """
    if not stream:
        return None
    i = int(np.searchsorted(times, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(stream)]
    return min((stream[j] for j in cands), key=lambda m: abs(m[0] - t))


def _quat_to_yaw(q: dict) -> float:
    """Quaternion → góc yaw quanh trục z (chỉ cần yaw vì robot di chuyển trên mặt phẳng)."""
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _payload_bytes(payload: dict) -> bytes:
    """Lấy nội dung ảnh. JSON của Foxglove mã hoá trường bytes bằng base64."""
    return base64.b64decode(payload["data"])


def sync_frames(raw: RawStreams, tol_ms: float, min_move: float, min_turn_deg: float) -> List[EpisodeData]:
    """Ghép các luồng thành chuỗi frame đều đặn, rồi cắt theo ranh giới episode.

    Nguyên tắc: **luồng ảnh RGB góc thẳng là nhịp chính**. Với mỗi ảnh tại thời điểm t,
    đi tìm ảnh cúi / depth / pose gần t nhất. Lệch quá `tol_ms` → bỏ frame (thà mất frame
    còn hơn ghép nhầm pose của thời điểm khác — sai kiểu này KHÔNG báo lỗi lúc train).
    """
    tol_ns = int(tol_ms * 1e6)
    t_down = np.array([m[0] for m in raw.rgb_down])
    t_depth = np.array([m[0] for m in raw.depth])
    t_pose = np.array([m[0] for m in raw.pose])

    # (1) ranh giới episode từ topic /task/episode
    spans: List[Tuple[int, int, str]] = []
    open_ep: Optional[Tuple[int, str]] = None
    for t, p in raw.episodes:
        if p.get("event") == "start":
            open_ep = (t, p.get("instruction", ""))
        elif p.get("event") == "end" and open_ep is not None:
            spans.append((open_ep[0], t, open_ep[1]))
            open_ep = None
    if not spans:  # không có marker → coi cả file là 1 episode
        t_lo = raw.rgb_front[0][0]
        t_hi = raw.rgb_front[-1][0]
        spans = [(t_lo, t_hi, "")]

    episodes: List[EpisodeData] = []
    dropped_sync = 0
    dropped_static = 0

    for ep_start, ep_end, instruction in spans:
        frames: List[Frame] = []
        for t, p_front in raw.rgb_front:
            if not (ep_start - tol_ns <= t <= ep_end + tol_ns):
                continue
            m_down = _nearest(raw.rgb_down, t_down, t)
            m_depth = _nearest(raw.depth, t_depth, t)
            m_pose = _nearest(raw.pose, t_pose, t)
            if m_down is None or m_depth is None or m_pose is None:
                dropped_sync += 1
                continue
            if max(abs(m_down[0] - t), abs(m_depth[0] - t), abs(m_pose[0] - t)) > tol_ns:
                dropped_sync += 1
                continue

            pos = m_pose[1]["pose"]["position"]
            yaw = _quat_to_yaw(m_pose[1]["pose"]["orientation"])

            # (2) lọc keyframe: bỏ frame mà robot gần như đứng yên so với frame trước.
            #     Log robot thật hay có hàng chục frame trùng nhau lúc chờ/đứng im.
            if frames:
                prev = frames[-1]
                d = math.hypot(pos["x"] - prev.x, pos["y"] - prev.y)
                dyaw = abs(math.atan2(math.sin(yaw - prev.yaw), math.cos(yaw - prev.yaw)))
                if d < min_move and math.degrees(dyaw) < min_turn_deg:
                    dropped_static += 1
                    continue

            frames.append(
                Frame(
                    t_ns=t,
                    rgb_front=_payload_bytes(p_front),
                    rgb_down=_payload_bytes(m_down[1]),
                    depth_png=_payload_bytes(m_depth[1]),
                    x=pos["x"],
                    y=pos["y"],
                    yaw=yaw,
                )
            )
        if len(frames) >= 4:  # loader bỏ episode < 4 frame
            episodes.append(EpisodeData(instruction=instruction, frames=frames))

    print("\n── B. Đồng bộ & cắt episode ────────────────────────────")
    print(f"   episode giữ lại : {len(episodes)}")
    print(f"   frame mỗi ep    : {[len(e.frames) for e in episodes]}")
    print(f"   bỏ do lệch giờ  : {dropped_sync}")
    print(f"   bỏ do đứng yên  : {dropped_static}")
    return episodes


# =============================================================================
# GIAI ĐOẠN C — SINH NHÃN (trái tim của pipeline)
# =============================================================================
def camera_pose_from_base(x: float, y: float, yaw: float, height_m: float, pitch_deg: float) -> np.ndarray:
    """Ma trận 4×4 **camera → world**, đúng quy ước cột của `pose.{setting}` trong `vln_ce`."""
    p = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    z_cam = Rz @ np.array([math.cos(p), 0.0, -math.sin(p)])  # trục quang, cúi p độ
    x_cam = Rz @ np.array([0.0, -1.0, 0.0])  # "phải" của camera
    y_cam = np.cross(z_cam, x_cam)  # "xuống" của camera
    T = np.eye(4, dtype=np.float32)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_cam, y_cam, z_cam
    T[:3, 3] = np.array([x, y, height_m])
    return T


def project_to_pixel(cam_pose: np.ndarray, K: np.ndarray, p_world: np.ndarray, w: int, h: int):
    """Chiếu một điểm 3D trong world xuống ảnh. Trả về `(u, v, nhìn_thấy)`.

        P_cam = Rᵀ · (P_world − C)          # đưa điểm về hệ camera
        u = fx·X/Z + cx ,  v = fy·Y/Z + cy   # phép chiếu phối cảnh

    Điểm ở SAU lưng camera (Z ≤ 0) hoặc rơi ra ngoài khung → coi như không thấy.
    """
    R = cam_pose[:3, :3]
    C = cam_pose[:3, 3]
    p_cam = R.T @ (p_world - C)
    if p_cam[2] <= 1e-6:
        return -1, -1, False
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    if not (0 <= u < w and 0 <= v < h):
        return -1, -1, False
    return int(round(u)), int(round(v)), True


def discretize_actions(frames: List[Frame], min_turn_deg: float) -> List[int]:
    """Quỹ đạo liên tục → chuỗi "nút bấm" rời rạc `{-1, 0, 1, 2, 3}`.

    Quy ước của `vln_ce` (đã đối chiếu data thật):
      `action[i]` = việc đã làm để đi từ frame i-1 TỚI frame i; `action[0] = -1` (frame khởi đầu).
      (Loader dịch trái một nhịp: `actions = item['actions'][1:] + [0]` — nên frame cuối thành STOP.)
    """
    actions = [-1]
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        dyaw = math.atan2(math.sin(b.yaw - a.yaw), math.cos(b.yaw - a.yaw))
        if abs(math.degrees(dyaw)) >= min_turn_deg:
            actions.append(2 if dyaw > 0 else 3)  # yaw tăng = quay TRÁI
        else:
            actions.append(1)  # còn lại coi là tiến (kể cả bước rất ngắn)
    return actions


def find_subgoal_frames(actions: List[int]) -> List[int]:
    """Chọn các frame làm "điểm đích trung gian" (sub-goal).

    Quy tắc: mỗi khi robot KẾT THÚC một đoạn đi thẳng (chuyển từ `1` sang xoay) thì frame đó
    là một sub-goal; cộng thêm frame cuối cùng của episode. Ý nghĩa trực quan: *"đi thẳng tới
    chỗ rẽ"* — đúng thứ System 2 phải học chỉ ra trên ảnh.

    (Data gốc R2R dùng các "viewpoint" của đồ thị điều hướng làm sub-goal; ta không có đồ thị
    đó nên dùng điểm rẽ làm xấp xỉ. Đây là chỗ bạn nên thay bằng luật riêng của mình nếu có.)
    """
    subgoals = []
    for i in range(1, len(actions)):
        if actions[i] in (2, 3) and actions[i - 1] == 1:
            subgoals.append(i)
    last = len(actions) - 1
    if not subgoals or subgoals[-1] != last:
        subgoals.append(last)
    return subgoals


def make_labels(ep: EpisodeData, K: np.ndarray, cfg) -> dict:
    """Sinh 4 cột nhãn cho một episode."""
    n = len(ep.frames)
    actions = discretize_actions(ep.frames, cfg.min_turn_deg)
    subgoals = find_subgoal_frames(actions)

    poses = np.zeros((n, 4, 4), dtype=np.float32)
    goals = np.full((n, 2), -1, dtype=np.int32)
    rel_ids = np.full((n,), -1, dtype=np.int32)

    for t, fr in enumerate(ep.frames):
        # pose của camera pitch_2 (camera "nhìn cúi") — đây là camera mà `setting` mô tả
        poses[t] = camera_pose_from_base(fr.x, fr.y, fr.yaw, cfg.height_cm / 100.0, cfg.pitch2)

        g = next((s for s in subgoals if s > t), None)
        if g is None:
            continue
        # waypoint = vị trí chân robot ở frame tương lai `g`, nằm trên SÀN (z = 0)
        p_world = np.array([ep.frames[g].x, ep.frames[g].y, 0.0], dtype=np.float32)
        u, v, visible = project_to_pixel(poses[t], K, p_world, cfg.width, cfg.height)
        if visible:
            goals[t] = (u, v)
            rel_ids[t] = g - t

    return {"action": np.array(actions, dtype=np.int32), "pose": poses, "goal": goals, "rel_id": rel_ids}


# =============================================================================
# GIAI ĐOẠN D — GHI ẢNH
# =============================================================================
def write_images(ep_idx: int, ep: EpisodeData, scene_dir: str, cfg) -> None:
    """Ghi 3 luồng ảnh đúng tên thư mục + tên file mà loader ghép ra.

    Loader ghép đường dẫn ở `internvla_n1_lerobot_dataset.py:1014-1022`:
        videos/chunk-000/observation.images.rgb.{H}cm_{pitch_1}deg/episode_{id:06d}_{frame}.jpg
        (đổi `_{pitch_1}deg`→`_{pitch_2}deg` để ra ảnh cúi;
         thêm đổi `rgb`→`depth`, `.jpg`→`.png` để ra depth)
    """
    chunk = f"chunk-{ep_idx // 1000:03d}"
    base = os.path.join(scene_dir, "videos", chunk)
    d_front = os.path.join(base, f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p1_tag}deg")
    d_down = os.path.join(base, f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p2_tag}deg")
    d_depth = os.path.join(base, f"observation.images.depth.{cfg.h_tag}cm_{cfg.p2_tag}deg")
    for d in (d_front, d_down, d_depth):
        os.makedirs(d, exist_ok=True)

    for i, fr in enumerate(ep.frames):
        stem = f"episode_{ep_idx:06d}_{i}"
        _save_jpeg(fr.rgb_front, os.path.join(d_front, f"{stem}.jpg"))
        _save_jpeg(fr.rgb_down, os.path.join(d_down, f"{stem}.jpg"))
        _save_depth_png(fr.depth_png, os.path.join(d_depth, f"{stem}.png"))


def _save_jpeg(payload: bytes, path: str) -> None:
    """Nếu message đã là JPEG thì ghi thẳng byte (không giải nén → không mất chất lượng)."""
    if payload[:2] == b"\xff\xd8":
        with open(path, "wb") as f:
            f.write(payload)
    else:
        Image.open(io.BytesIO(payload)).convert("RGB").save(path, format="JPEG", quality=90)


def _save_depth_png(payload: bytes, path: str) -> None:
    """Depth PHẢI là PNG uint16 milimét (loader chia 1000 rồi clip 5 m)."""
    img = Image.open(io.BytesIO(payload))
    arr = np.array(img)
    if arr.dtype != np.uint16:
        raise SystemExit(
            f"❌ Depth không phải uint16 (đang là {arr.dtype}). Loader S2 chia 1000 để ra mét — "
            "hãy chuyển depth về uint16 milimét trước khi ghi."
        )
    Image.fromarray(arr).save(path, format="PNG", optimize=True)


# =============================================================================
# GIAI ĐOẠN E — GHI PARQUET + META (đúng dtype LeRobot v2.1)
# =============================================================================
def write_parquet(ep_idx: int, labels: dict, scene_dir: str, cfg, task_index: int, index_offset: int, fps: float):
    """Ghi bảng số của một episode.

    dtype phải khớp bản gốc (đo bằng `pq.read_table(...).schema` trên `vln_ce`):
        action                          : int32
        pose.{setting}                  : list<list<float32>>   (4×4)
        goal.{setting}                  : fixed_size_list<int32>[2]
        relative_goal_frame_id.{setting}: int32
        timestamp float32 · frame_index/episode_index/index/task_index int64
    Lý do phải đúng: loader gọi `.tolist()` trên từng ô (dòng 786-789) — kiểu Python thuần
    (int, list) không có `.tolist()` → vỡ ngay khi nạp.
    """
    n = len(labels["action"])
    s = cfg.setting
    table = pa.table(
        {
            "action": pa.array(labels["action"], type=pa.int32()),
            f"pose.{s}": pa.array([p.tolist() for p in labels["pose"]], type=pa.list_(pa.list_(pa.float32()))),
            f"goal.{s}": pa.array([g.tolist() for g in labels["goal"]], type=pa.list_(pa.int32(), 2)),
            f"relative_goal_frame_id.{s}": pa.array(labels["rel_id"], type=pa.int32()),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / np.float32(fps), type=pa.float32()),
            "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(np.full(n, ep_idx, dtype=np.int64), type=pa.int64()),
            "index": pa.array(np.arange(index_offset, index_offset + n, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64()),
        }
    )
    d = os.path.join(scene_dir, "data", f"chunk-{ep_idx // 1000:03d}")
    os.makedirs(d, exist_ok=True)
    pq.write_table(table, os.path.join(d, f"episode_{ep_idx:06d}.parquet"))
    return table


def write_meta(scene_dir: str, episodes: List[EpisodeData], all_labels: List[dict], cfg, fps: float) -> None:
    """Ghi 4 file `meta/`.

    Loader S2 chỉ đọc **`episodes.jsonl`** (dòng 765) — nhưng ba file kia vẫn nên có để
    dataset đúng chuẩn LeRobot v2.1 và để công cụ khác đọc được.
    """
    meta = os.path.join(scene_dir, "meta")
    os.makedirs(meta, exist_ok=True)
    s = cfg.setting

    with open(os.path.join(meta, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for i, ep in enumerate(episodes):
            f.write(
                json.dumps(
                    {"episode_index": i, "tasks": [ep.instruction], "length": len(ep.frames)}, ensure_ascii=False
                )
                + "\n"
            )

    with open(os.path.join(meta, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for i, ep in enumerate(episodes):
            f.write(json.dumps({"task_index": i, "task": ep.instruction}, ensure_ascii=False) + "\n")

    with open(os.path.join(meta, "episodes_stats.jsonl"), "w", encoding="utf-8") as f:
        for i, lab in enumerate(all_labels):
            n = len(lab["action"])

            def st(a: np.ndarray) -> dict:
                a = np.asarray(a, dtype=np.float64).reshape(n, -1)
                return {
                    "min": a.min(0).tolist(),
                    "max": a.max(0).tolist(),
                    "mean": a.mean(0).tolist(),
                    "std": a.std(0).tolist(),
                    "count": [n],
                }

            f.write(
                json.dumps(
                    {
                        "episode_index": i,
                        "stats": {
                            "action": st(lab["action"]),
                            f"goal.{s}": st(lab["goal"]),
                            f"relative_goal_frame_id.{s}": st(lab["rel_id"]),
                        },
                    }
                )
                + "\n"
            )

    total_frames = sum(len(e.frames) for e in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "custom",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(episodes),
        "total_chunks": (len(episodes) - 1) // 1000 + 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}_{frame}.jpg",
        "features": {
            "action": {"dtype": "int32", "shape": [1], "names": ["action_index"]},
            f"pose.{s}": {"dtype": "float32", "shape": [4, 4], "names": [f"pose.{s}"]},
            f"goal.{s}": {"dtype": "int32", "shape": [2], "names": [f"goal.{s}"]},
            f"relative_goal_frame_id.{s}": {"dtype": "int32", "shape": [1], "names": [f"relative_goal_frame_id.{s}"]},
        },
    }
    with open(os.path.join(meta, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# =============================================================================
# GIAI ĐOẠN F — TỰ KIỂM ĐỊNH (mô phỏng đúng logic cắt mẫu của loader)
# =============================================================================
def self_check(scene_dir: str, cfg, sample_step: int = 4, num_future_steps: int = 4) -> None:
    """Đọc lại dataset vừa ghi và đếm số mẫu train theo ĐÚNG logic của `NavPixelGoalDataset`.

    Bản sao logic ở `internvla_n1_lerobot_dataset.py:857-947`:
      - `actions = actions[1:] + [0]`               (dịch trái, frame cuối = STOP)
      - duyệt `start = n * sample_step`
      - `pixel_goal[0] == -1` → mẫu *turn* (bỏ nếu đang đi thẳng)
      - `pixel_goal[0] >= 3`  → mẫu *pixel_goal*      ← loại quan trọng nhất
      - `0 < pixel_goal[0] < 3` → **bị bỏ**
      - mỗi episode góp thêm 1 mẫu *stop*
    """
    import glob

    s = cfg.setting
    n_goal = n_turn = n_stop = n_short = 0
    missing = 0

    ep_files = sorted(glob.glob(os.path.join(scene_dir, "data", "chunk-*", "episode_*.parquet")))
    for ep_idx, path in enumerate(ep_files):
        df = pq.read_table(path).to_pandas()
        for col in (f"pose.{s}", f"goal.{s}", f"relative_goal_frame_id.{s}"):
            if col not in df.columns:
                raise SystemExit(f"❌ Thiếu cột {col} — loader sẽ bỏ nguyên scene này.")
        actions = df["action"].tolist()[1:] + [0]
        rel = df[f"relative_goal_frame_id.{s}"].tolist()
        n = len(actions)
        if n < 4:
            continue
        for k in range(n // sample_step + 1):
            start = k * sample_step
            if start in (n, n - 1):
                continue
            if rel[start] == -1:
                if actions[start] != 1:
                    n_turn += 1
            elif rel[start] < 3:
                n_short += 1
            else:
                n_goal += 1
                # kiểm tra ảnh của cả cửa sổ [start, start+goal_len] có tồn tại không
                for fid in range(0, start + rel[start] + 1):
                    for sub, ext in (
                        (f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p1_tag}deg", "jpg"),
                        (f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p2_tag}deg", "jpg"),
                        (f"observation.images.depth.{cfg.h_tag}cm_{cfg.p2_tag}deg", "png"),
                    ):
                        p = os.path.join(
                            scene_dir, "videos", f"chunk-{ep_idx//1000:03d}", sub, f"episode_{ep_idx:06d}_{fid}.{ext}"
                        )
                        if not os.path.exists(p):
                            missing += 1
        n_stop += 1

    print("\n── F. Tự kiểm định (mô phỏng logic loader) ─────────────")
    print(f"   mẫu pixel_goal : {n_goal}    ← loại quan trọng nhất")
    print(f"   mẫu turn       : {n_turn}")
    print(f"   mẫu stop       : {n_stop}  (loader nhân 5 khi pixel_goal_only=False)")
    print(f"   bị bỏ (k < 3)  : {n_short}")
    print(f"   file ảnh thiếu : {missing}")
    total_s2 = n_goal + n_turn + n_stop * 5
    print(f"   → train S2 (pixel_goal_only=False): {total_s2} mẫu")
    print(f"   → train dual (pixel_goal_only=True): {n_goal} mẫu")
    if n_goal == 0:
        print("   ⚠️  KHÔNG có mẫu pixel_goal nào! Kiểm tra lại góc cúi pitch_2 (camera có thấy sàn không?)")
    if missing:
        print("   ⚠️  Thiếu file ảnh → loader sẽ crash lúc __getitem__.")


# =============================================================================
# ĐIỀU PHỐI
# =============================================================================
@dataclass
class Config:
    height_cm: float
    pitch1: float
    pitch2: float
    width: int
    height: int
    min_move: float
    min_turn_deg: float

    @property
    def h_tag(self) -> int:
        return int(round(self.height_cm))

    @property
    def p1_tag(self) -> int:
        return int(round(self.pitch1))

    @property
    def p2_tag(self) -> int:
        return int(round(self.pitch2))

    @property
    def setting(self) -> str:
        return f"{self.h_tag}cm_{self.p2_tag}deg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--out", default="./traj_data", help="thư mục gốc chứa dataset sinh ra")
    ap.add_argument("--dataset-name", default="myrobot")
    ap.add_argument("--scene-id", default="scene_0000")
    ap.add_argument("--topic-rgb-front", default="/camera/front/image_raw")
    ap.add_argument("--topic-rgb-down", default="/camera/lookdown/image_raw")
    ap.add_argument("--topic-depth", default="/camera/lookdown/depth")
    ap.add_argument("--topic-pose", default="/robot/pose")
    ap.add_argument("--topic-caminfo", default="/camera/lookdown/camera_info")
    ap.add_argument("--topic-episode", default="/task/episode")
    ap.add_argument("--height-cm", type=float, default=None, help="mặc định lấy từ metadata mcap")
    ap.add_argument("--pitch1", type=float, default=None)
    ap.add_argument("--pitch2", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None, help="ghi vào cột timestamp; mặc định lấy từ metadata mcap")
    ap.add_argument("--tol-ms", type=float, default=60.0, help="ngưỡng lệch thời gian khi đồng bộ")
    ap.add_argument("--min-move", type=float, default=0.05, help="m — nhỏ hơn coi như đứng yên")
    ap.add_argument("--min-turn-deg", type=float, default=5.0, help="độ — nhỏ hơn coi như không xoay")
    ap.add_argument(
        "--instruction-file",
        default=None,
        help='JSON {"0": "câu lệnh ep 0", ...} — dùng khi mcap KHÔNG chứa câu lệnh',
    )
    args = ap.parse_args()

    topics = {
        "rgb_front": args.topic_rgb_front,
        "rgb_down": args.topic_rgb_down,
        "depth": args.topic_depth,
        "pose": args.topic_pose,
        "caminfo": args.topic_caminfo,
        "episodes": args.topic_episode,
    }
    raw = read_mcap(args.mcap, topics)
    if not raw.rgb_front:
        raise SystemExit(f"❌ Không có message nào ở topic {args.topic_rgb_front} — không có ảnh thì không train được.")
    if not raw.caminfo:
        raise SystemExit("❌ Thiếu camera_info (ma trận K). Không có K thì KHÔNG chiếu được pixel-goal.")

    md = raw.metadata
    cfg = Config(
        height_cm=args.height_cm if args.height_cm is not None else float(md.get("height_cm", 125)),
        pitch1=args.pitch1 if args.pitch1 is not None else float(md.get("pitch_1", 0)),
        pitch2=args.pitch2 if args.pitch2 is not None else float(md.get("pitch_2", 30)),
        width=int(raw.caminfo[0][1]["width"]),
        height=int(raw.caminfo[0][1]["height"]),
        min_move=args.min_move,
        min_turn_deg=args.min_turn_deg,
    )
    K = np.array(raw.caminfo[0][1]["K"], dtype=np.float64).reshape(3, 3)
    fps = args.fps if args.fps is not None else float(md.get("fps", 10.0))
    print(f"   setting = {cfg.setting} · ảnh {cfg.width}×{cfg.height} · fx={K[0,0]:.1f} cx={K[0,2]:.1f} · fps={fps}")

    episodes = sync_frames(raw, args.tol_ms, args.min_move, args.min_turn_deg)
    if not episodes:
        raise SystemExit("❌ Không dựng được episode nào. Kiểm tra tên topic và ngưỡng --tol-ms.")

    if args.instruction_file:
        overrides = json.load(open(args.instruction_file, encoding="utf-8"))
        for i, ep in enumerate(episodes):
            ep.instruction = overrides.get(str(i), ep.instruction)
    for i, ep in enumerate(episodes):
        if not ep.instruction.strip():
            raise SystemExit(
                f"❌ Episode {i} không có câu lệnh. System 2 học từ NGÔN NGỮ — thiếu câu lệnh thì "
                "mẫu vô nghĩa. Dùng --instruction-file để bổ sung."
            )

    scene_dir = os.path.join(args.out, args.dataset_name, args.scene_id)
    os.makedirs(scene_dir, exist_ok=True)

    print("\n── C+D+E. Sinh nhãn, ghi ảnh & parquet ─────────────────")
    all_labels = []
    index_offset = 0
    for i, ep in enumerate(episodes):
        labels = make_labels(ep, K, cfg)
        write_images(i, ep, scene_dir, cfg)
        write_parquet(i, labels, scene_dir, cfg, task_index=i, index_offset=index_offset, fps=fps)
        index_offset += len(ep.frames)
        all_labels.append(labels)
        n_goal = int((labels["rel_id"] >= 0).sum())
        print(f"   ep {i}: {len(ep.frames):>3} frame · {n_goal:>3} frame có pixel-goal · action={_hist(labels)}")
    write_meta(scene_dir, episodes, all_labels, cfg, fps)

    print(f"\n   ✅ Đã ghi {scene_dir}")
    self_check(scene_dir, cfg)
    print(
        "\n── Bước tiếp theo ──────────────────────────────────────\n"
        f"   1. Đăng ký dataset vào `data_dict` (internvla_n1_lerobot_dataset.py:127):\n"
        f'        "{args.dataset_name}_{cfg.h_tag}cm_{cfg.p1_tag}_{cfg.p2_tag}": '
        f'{{"data_path": "{args.out}/{args.dataset_name}", "height": {cfg.h_tag}, '
        f'"pitch_1": {cfg.p1_tag}, "pitch_2": {cfg.p2_tag}}},\n'
        f"   2. Trỏ `--vln_dataset_use {args.dataset_name}_{cfg.h_tag}cm_{cfg.p1_tag}_{cfg.p2_tag}` "
        "trong scripts/train/qwenvl_train/train_system2.sh"
    )


def _hist(labels: dict) -> str:
    vals, counts = np.unique(labels["action"], return_counts=True)
    return " ".join(f"{IDX2NAME.get(int(v), v)}×{c}" for v, c in zip(vals, counts))


if __name__ == "__main__":
    main()
