"""
db32s2.py — Chuyển **rosbag2 `.db3`** (log robot ROS 2 thật) → data train System 2 (LeRobot / `vln_ce`).

=============================================================================
CÁCH CHẠY
-----------------------------------------------------------------------------
    pip install numpy pillow pyarrow scipy

    # 1) Khảo sát bag trước (LUÔN làm bước này): topic, cây TF, hình học camera
    python db32s2.py --bag vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s --inspect

    # 2) Sinh data (câu lệnh ngôn ngữ là BẮT BUỘC — S2 học từ ngôn ngữ)
    python db32s2.py --bag vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s \
        --out ./traj_data --dataset-name vinbot --scene-id lab_run01 \
        --instruction "Walk straight along the white walkway past the workbenches and stop at the end of the aisle."

Sáu giai đoạn : A đọc bag · B đồng bộ · C sinh nhãn · D ghi ảnh · E ghi
parquet+meta · F tự kiểm định.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sqlite3
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

NS = 1_000_000_000
DEPTH_MAX_MM = 10000  # data gốc vln_ce clip depth ở 10 m
IDX2NAME = {-1: "start", 0: "STOP", 1: "↑ tiến", 2: "← trái", 3: "→ phải", 5: "↓ cúi"}

# Phép xoay chuẩn ROS giữa "khung thân camera" (x trước, y trái, z lên)
# và "khung quang học" (x phải, y xuống, z trục quang).  p_optical = R_OPT_BODY @ p_body
R_OPT_BODY = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


# =============================================================================
# PHẦN 0 — BỘ GIẢI MÃ CDR (thay cho ROS)
# =============================================================================
class CDR:
    """Đọc một message ROS 2 đã mã hoá CDR mà không cần cài ROS.

    CDR = Common Data Representation. Ba luật cần biết:
      1. 4 byte đầu là *encapsulation header*: `[0x00, endian, 0x00, 0x00]`; `endian` lẻ = little.
      2. Mỗi số nguyên/thực phải **căn lề theo kích thước của nó**, tính từ **sau** 4 byte header.
         (float64 phải nằm ở offset chia hết cho 8 — nếu không, chèn byte đệm.)
      3. `string` = uint32 độ dài (**kể cả ký tự NUL**) + bytes. `sequence<T>` = uint32 số phần tử + phần tử.
    Sai luật căn lề là lỗi phổ biến nhất khi tự viết bộ giải mã → mọi số sau đó lệch hết.
    """

    __slots__ = ("b", "little", "p", "base")

    def __init__(self, buf: bytes):
        self.b = buf
        self.little = bool(buf[1] & 1)
        self.p = 4
        self.base = 4

    def _align(self, n: int) -> None:
        self.p += (-(self.p - self.base)) % n

    def _get(self, fmt: str, n: int):
        self._align(n)
        v = struct.unpack_from(("<" if self.little else ">") + fmt, self.b, self.p)[0]
        self.p += n
        return v

    def u8(self):
        return self._get("B", 1)

    def u32(self):
        return self._get("I", 4)

    def i32(self):
        return self._get("i", 4)

    def f64(self):
        return self._get("d", 8)

    def boolean(self):
        return bool(self._get("B", 1))

    def string(self) -> str:
        n = self.u32()
        s = bytes(self.b[self.p : self.p + max(0, n - 1)]).decode("utf-8", "replace")
        self.p += n
        return s

    def farray(self, n: int) -> np.ndarray:
        self._align(8)
        a = np.frombuffer(self.b, dtype="<f8" if self.little else ">f8", count=n, offset=self.p)
        self.p += 8 * n
        return a.copy()

    def fseq(self) -> np.ndarray:
        return self.farray(self.u32())

    def byteseq(self) -> bytes:
        n = self.u32()
        d = bytes(self.b[self.p : self.p + n])
        self.p += n
        return d

    def stamp_header(self) -> Tuple[int, str]:
        """`std_msgs/Header` → (thời điểm ns, frame_id)."""
        sec = self.i32()
        nsec = self.u32()
        return sec * NS + nsec, self.string()


def msg_camera_info(buf: bytes) -> dict:
    c = CDR(buf)
    t, frame = c.stamp_header()
    h, w = c.u32(), c.u32()
    model = c.string()
    c.fseq()  # D — không dùng (ảnh đã rectify)
    K = c.farray(9).reshape(3, 3)
    return {"t": t, "frame": frame, "h": h, "w": w, "model": model, "K": K}


def msg_compressed_image(buf: bytes) -> dict:
    c = CDR(buf)
    t, frame = c.stamp_header()
    fmt = c.string()
    return {"t": t, "frame": frame, "format": fmt, "data": c.byteseq()}


def msg_odometry(buf: bytes) -> dict:
    c = CDR(buf)
    t, frame = c.stamp_header()
    child = c.string()
    pos = (c.f64(), c.f64(), c.f64())
    quat = (c.f64(), c.f64(), c.f64(), c.f64())
    return {"t": t, "frame": frame, "child": child, "pos": pos, "quat": quat}


def msg_tf(buf: bytes) -> List[dict]:
    c = CDR(buf)
    out = []
    for _ in range(c.u32()):
        t, parent = c.stamp_header()
        child = c.string()
        xyz = (c.f64(), c.f64(), c.f64())
        quat = (c.f64(), c.f64(), c.f64(), c.f64())
        out.append({"t": t, "parent": parent, "child": child, "xyz": xyz, "quat": quat})
    return out


def msg_point_cloud2(buf: bytes) -> dict:
    c = CDR(buf)
    t, frame = c.stamp_header()
    h, w = c.u32(), c.u32()
    fields = []
    for _ in range(c.u32()):
        name = c.string()
        off, dtype, cnt = c.u32(), c.u8(), c.u32()
        fields.append({"name": name, "offset": off, "datatype": dtype, "count": cnt})
    c.boolean()  # is_bigendian
    point_step = c.u32()
    c.u32()  # row_step
    data = c.byteseq()
    return {"t": t, "frame": frame, "h": h, "w": w, "fields": fields, "point_step": point_step, "data": data}


# =============================================================================
# PHẦN 1 — HÌNH HỌC & CÂY TF
# =============================================================================
def quat_to_R(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_of(R: np.ndarray) -> float:
    """Góc quay quanh trục z của một ma trận xoay (robot đi trên mặt phẳng nên chỉ cần yaw)."""
    return math.atan2(R[1, 0], R[0, 0])


def transform_matrix(tr: dict) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = quat_to_R(tr["quat"])
    M[:3, 3] = tr["xyz"]
    return M


class TFBuffer:
    """Cây biến đổi TF tối giản: tra `T_root←frame` tại một thời điểm.

    Vì sao cần? Trong ROS, "camera ở đâu trên robot" **không** nằm trong một trường nào cả — nó là
    tích của một chuỗi biến đổi cha→con (`pelvis → waist_yaw → … → camera`). Biến đổi cố định nằm ở
    `/tf_static`, biến đổi động (khớp cổ, chân) nằm ở `/tf` theo thời gian.
    """

    def __init__(self):
        self.static: Dict[str, dict] = {}
        self.dynamic: Dict[str, List[dict]] = {}
        self._times: Dict[str, np.ndarray] = {}

    def add_static(self, transforms: List[dict]) -> None:
        for tr in transforms:
            self.static[tr["child"]] = tr

    def add_dynamic(self, transforms: List[dict]) -> None:
        for tr in transforms:
            self.dynamic.setdefault(tr["child"], []).append(tr)

    def finalize(self) -> None:
        for k, v in self.dynamic.items():
            v.sort(key=lambda x: x["t"])
            self._times[k] = np.array([x["t"] for x in v])

    def parent_of(self, frame: str, t: int) -> Optional[dict]:
        if frame in self.static:  # biến đổi cố định thắng: đúng cho mọi thời điểm
            return self.static[frame]
        lst = self.dynamic.get(frame)
        if not lst:
            return None
        i = int(np.searchsorted(self._times[frame], t))
        cands = [j for j in (i - 1, i) if 0 <= j < len(lst)]
        return min((lst[j] for j in cands), key=lambda x: abs(x["t"] - t))

    def chain_to_root(self, frame: str, t: int) -> Tuple[np.ndarray, str]:
        """Nhân dồn lên tới gốc cây. Trả về `(T_root←frame, tên_gốc)`."""
        M = np.eye(4)
        f = frame
        seen = set()
        while f not in seen:
            seen.add(f)
            tr = self.parent_of(f, t)
            if tr is None:
                return M, f
            M = transform_matrix(tr) @ M
            f = tr["parent"]
        return M, f

    def lookup(self, target: str, source: str, t: int) -> np.ndarray:
        """`T_target←source` — đưa một điểm từ hệ `source` sang hệ `target`."""
        Mt, rt = self.chain_to_root(target, t)
        Ms, rs = self.chain_to_root(source, t)
        if rt != rs:
            raise SystemExit(f"❌ Frame '{target}' và '{source}' không cùng cây TF ({rt} vs {rs}).")
        return np.linalg.inv(Mt) @ Ms

    def resolve_camera(self, frame_id: str, base: str, t: int) -> Tuple[np.ndarray, str]:
        """`T_base←camera_optical`, tự xử lý trường hợp frame `_optical` KHÔNG có trong cây TF.

        Nhiều driver (ZED là một ví dụ) đặt `frame_id` của ảnh là `..._camera_frame_optical` nhưng
        chỉ phát TF cho `..._camera_frame` (khung thân). Khi đó ta tự nhân thêm phép xoay chuẩn
        thân→quang học `R_OPT_BODY`.
        """
        if frame_id in self.static or frame_id in self.dynamic:
            return self.lookup(base, frame_id, t), frame_id
        if frame_id.endswith("_optical"):
            body = frame_id[: -len("_optical")]
            if body in self.static or body in self.dynamic:
                T = self.lookup(base, body, t).copy()
                T[:3, :3] = T[:3, :3] @ R_OPT_BODY.T  # T_base←optical = T_base←body · T_body←optical
                return T, body + " (+xoay thân→quang học)"
        raise SystemExit(f"❌ Không tìm được frame '{frame_id}' trong cây TF. Dùng --height-cm/--pitch2 để khai tay.")


def camera_pose_synth(x: float, y: float, yaw: float, height_m: float, pitch_deg: float) -> np.ndarray:
    """Ma trận 4×4 camera→world **dựng lại theo đúng quy ước `pose.{setting}` của `vln_ce`**.

    Dùng khi `--pose-mode synth`: bỏ qua roll/lệch nhỏ của camera thật, ép về đúng khuôn mà
    `get_trajectory_relative_to_frame(..., camera_deg=pitch_2)` giả định.
    """
    p = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    z_cam = Rz @ np.array([math.cos(p), 0.0, -math.sin(p)])
    x_cam = Rz @ np.array([0.0, -1.0, 0.0])
    T = np.eye(4, dtype=np.float32)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_cam, np.cross(z_cam, x_cam), z_cam
    T[:3, 3] = (x, y, height_m)
    return T


def pitch_down_deg(T_base_cam: np.ndarray) -> float:
    """Góc cúi (độ) của camera: lấy từ thành phần z của trục quang (cột 3)."""
    return math.degrees(math.asin(float(np.clip(-T_base_cam[2, 2], -1.0, 1.0))))


# =============================================================================
# GIAI ĐOẠN A — ĐỌC BAG
# =============================================================================
@dataclass
class BagTopic:
    tid: int
    name: str
    type: str
    count: int


@dataclass
class Bag:
    """Nhiều file `.db3` của cùng một bag được ghép lại như một."""

    paths: List[str]
    topics: Dict[str, BagTopic] = field(default_factory=dict)

    @staticmethod
    def open(path: str) -> "Bag":
        if os.path.isdir(path):
            paths = sorted(glob.glob(os.path.join(path, "*.db3")))
        else:
            paths = [path]
        if not paths:
            raise SystemExit(f"❌ Không thấy file .db3 nào trong {path}")
        bag = Bag(paths=paths)
        for p in paths:
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            counts = dict(con.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"))
            for tid, name, typ in con.execute("SELECT id, name, type FROM topics"):
                if name in bag.topics:
                    bag.topics[name].count += counts.get(tid, 0)
                else:
                    bag.topics[name] = BagTopic(tid, name, typ, counts.get(tid, 0))
            con.close()
        return bag

    def _iter_raw(self, topic: str, with_data: bool):
        """Duyệt message của một topic qua mọi file .db3, theo thứ tự thời gian."""
        cols = "timestamp, data" if with_data else "timestamp, id"
        for p in self.paths:
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            tid = con.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
            if tid is not None:
                for row in con.execute(f"SELECT {cols} FROM messages WHERE topic_id=? ORDER BY timestamp", (tid[0],)):
                    yield p, row
            con.close()

    def read_all(self, topic: str, parser) -> List[dict]:
        """Giải mã toàn bộ một topic (chỉ dùng cho topic NHẸ: odom, tf, camera_info)."""
        return [parser(row[1]) for _, row in self._iter_raw(topic, True)]

    def index(self, topic: str) -> List[Tuple[int, str, int]]:
        """Chỉ lấy `(timestamp, file, rowid)` — KHÔNG tải blob.

        Đây là mẹo quan trọng: bag mẫu nặng 572 MB, ảnh + point cloud chiếm gần hết. Ta lập chỉ mục
        thời gian trước, chọn keyframe, rồi mới nạp đúng những blob cần → RAM chỉ vài chục MB.
        """
        return [(row[0], p, row[1]) for p, row in self._iter_raw(topic, False)]

    def fetch(self, path: str, rowid: int) -> bytes:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        blob = con.execute("SELECT data FROM messages WHERE id=?", (rowid,)).fetchone()[0]
        con.close()
        return blob


def autodetect_topics(bag: Bag, args) -> Dict[str, Optional[str]]:
    """Đoán tên topic từ *kiểu message* + từ khoá trong tên. In rõ lựa chọn để người dùng kiểm tra."""
    by_type: Dict[str, List[str]] = {}
    for name, t in bag.topics.items():
        by_type.setdefault(t.type, []).append(name)

    imgs = sorted(by_type.get("sensor_msgs/msg/CompressedImage", []) + by_type.get("sensor_msgs/msg/Image", []))
    # chỉ giữ camera bên trái (mắt trái của stereo) — mắt phải là dư thừa cho VLN
    left = [t for t in imgs if "/left/" in t] or imgs

    def pick(cands: List[str], *keys: str) -> Optional[str]:
        for k in keys:
            for c in cands:
                if k in c:
                    return c
        return cands[0] if cands else None

    down = args.rgb_down_topic or pick(left, "waist", "chest", "lower")
    front = args.rgb_front_topic or pick([t for t in left if t != down], "head", "front") or down
    if args.single_camera:
        front = down

    def caminfo_for(img: Optional[str]) -> Optional[str]:
        if img is None:
            return None
        cands = by_type.get("sensor_msgs/msg/CameraInfo", [])
        # chọn camera_info có tiền tố chung dài nhất với topic ảnh
        best, best_len = None, -1
        for c in cands:
            n = len(os.path.commonprefix([c, img]))
            if n > best_len:
                best, best_len = c, n
        return best

    clouds = by_type.get("sensor_msgs/msg/PointCloud2", [])
    cloud = args.depth_topic
    if cloud is None and clouds and down:
        cloud = max(clouds, key=lambda c: len(os.path.commonprefix([c, down])))

    return {
        "rgb_down": down,
        "rgb_front": front,
        "caminfo_down": args.caminfo_down_topic or caminfo_for(down),
        "caminfo_front": args.caminfo_front_topic or caminfo_for(front),
        "cloud": cloud,
        "odom": args.odom_topic or pick(by_type.get("nav_msgs/msg/Odometry", [])),
    }


def inspect(bag: Bag, args) -> None:
    """Chế độ khảo sát: in topic, cây TF, hình học camera suy từ TF, và quỹ đạo."""
    print(f"=== BAG: {', '.join(os.path.basename(p) for p in bag.paths)} ===")
    print(f"{'Topic':<62}{'Type':<36}{'Msg'}")
    print("-" * 108)
    for name in sorted(bag.topics):
        t = bag.topics[name]
        print(f"{name:<62}{t.type:<36}{t.count}")

    picks = autodetect_topics(bag, args)
    print("\n=== TOPIC ĐƯỢC CHỌN (tự đoán) ===")
    for k, v in picks.items():
        print(f"   {k:<14} = {v}")

    tf = load_tf(bag, args)
    print("\n=== CÂY TF: /tf_static ===")
    for child, tr in sorted(tf.static.items()):
        print(f"   {tr['parent']:<46} -> {child:<52} xyz={tuple(round(v, 4) for v in tr['xyz'])}")
    print(f"\n   ({len(tf.dynamic)} frame động từ /tf)")

    odom = bag.read_all(picks["odom"], msg_odometry) if picks["odom"] else []
    if odom:
        xy = np.array([o["pos"][:2] for o in odom])
        yaws = np.array([yaw_of(quat_to_R(o["quat"])) for o in odom])
        dist = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        dur = (odom[-1]["t"] - odom[0]["t"]) / 1e9
        print(f"\n=== QUỸ ĐẠO ({picks['odom']}: {odom[0]['frame']} → {odom[0]['child']}) ===")
        print(f"   {len(odom)} mẫu / {dur:.1f} s (~{len(odom)/max(dur,1e-9):.1f} Hz)")
        print(f"   đường đi {dist:.2f} m · yaw {math.degrees(yaws[0]):+.1f}° → {math.degrees(yaws[-1]):+.1f}°")

    t_mid = odom[len(odom) // 2]["t"] if odom else 0
    print("\n=== HÌNH HỌC CAMERA (suy từ TF) ===")
    pelvis_h = derive_base_height(tf, args.base_frame, t_mid, args.foot_offset)
    print(f"   base_frame '{args.base_frame}' cao ≈ {pelvis_h:.3f} m so với sàn (suy từ khớp chân)")
    for role, topic, ci_topic in (("pitch_1", picks["rgb_front"], picks["caminfo_front"]),
                                  ("pitch_2", picks["rgb_down"], picks["caminfo_down"])):
        if not topic or not ci_topic:
            continue
        ci = msg_camera_info(bag.fetch(*bag.index(ci_topic)[0][1:]))
        T, via = tf.resolve_camera(ci["frame"], args.base_frame, t_mid)
        print(
            f"   {role}: {topic}\n"
            f"         {ci['w']}×{ci['h']}  fx={ci['K'][0,0]:.1f} cx={ci['K'][0,2]:.1f} cy={ci['K'][1,2]:.1f}\n"
            f"         cao {T[2,3] + pelvis_h:.3f} m · cúi {pitch_down_deg(T):.1f}°  (qua TF: {via})"
        )
    print("\n→ Nếu các con số trên hợp lý, chạy lại không có --inspect để sinh data.")


def load_tf(bag: Bag, args) -> TFBuffer:
    tf = TFBuffer()
    if args.tf_static_topic in bag.topics:
        for m in bag.read_all(args.tf_static_topic, msg_tf):
            tf.add_static(m)
    if args.tf_topic in bag.topics:
        for m in bag.read_all(args.tf_topic, msg_tf):
            tf.add_dynamic(m)
    tf.finalize()
    if not tf.static and not tf.dynamic:
        raise SystemExit("❌ Bag không có /tf hay /tf_static → không suy được hình học camera.")
    return tf


def derive_base_height(tf: TFBuffer, base: str, t: int, foot_offset: float) -> float:
    """Chiều cao của `base_link` so với **sàn**, suy từ khớp thấp nhất của chân.

    `base_link` của robot hình người thường ở hông (pelvis) — TF không nói nó cách sàn bao nhiêu.
    Cách suy: tìm frame có tên chứa `ankle/foot/sole`, lấy z thấp nhất so với base, rồi trừ thêm
    `foot_offset` (dày bàn chân, mặc định 4 cm).
    """
    cands = [f for f in list(tf.static) + list(tf.dynamic) if any(k in f for k in ("ankle", "foot", "sole", "toe"))]
    zs = []
    for f in cands:
        try:
            zs.append(tf.lookup(base, f, t)[2, 3])
        except SystemExit:
            continue
    return (-min(zs) + foot_offset) if zs else 0.0


# =============================================================================
# GIAI ĐOẠN B — ĐỒNG BỘ THỜI GIAN
# =============================================================================
@dataclass
class Frame:
    t_ns: int
    rgb_front: Tuple[str, int]  # (file .db3, rowid) — nạp blob sau
    rgb_down: Tuple[str, int]
    cloud: Optional[Tuple[str, int]]
    x: float
    y: float
    yaw: float
    T_world_cam: np.ndarray  # 4×4 camera(pitch_2) → world, sàn ở z = 0


def _nearest_idx(times: np.ndarray, t: int) -> int:
    i = int(np.searchsorted(times, t))
    if i == 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def sync_frames(bag: Bag, tf: TFBuffer, picks: dict, args, pelvis_h: float, cam_frame: str) -> List[Frame]:
    """Chọn keyframe và ghép đủ 4 thứ cho mỗi frame: ảnh thẳng · ảnh cúi · point cloud · pose.

    Nhịp chính = **luồng ảnh cúi (`pitch_2`)**, vì đó là ảnh mà nhãn `goal` gắn vào.
    """
    idx_down = bag.index(picks["rgb_down"])
    idx_front = bag.index(picks["rgb_front"]) if picks["rgb_front"] != picks["rgb_down"] else idx_down
    idx_cloud = bag.index(picks["cloud"]) if (picks["cloud"] and args.depth_source == "pointcloud") else []
    odom = bag.read_all(picks["odom"], msg_odometry)
    if not odom:
        raise SystemExit(f"❌ Topic odometry '{picks['odom']}' rỗng → không có quỹ đạo thì không train được.")

    t_front = np.array([r[0] for r in idx_front])
    t_cloud = np.array([r[0] for r in idx_cloud]) if idx_cloud else np.empty(0)
    t_odom = np.array([o["t"] for o in odom])
    tol = int(args.tol_ms * 1e6)

    frames: List[Frame] = []
    n_late = n_static = 0
    for t, path, rowid in idx_down:
        j = _nearest_idx(t_odom, t)
        if abs(t_odom[j] - t) > tol:
            n_late += 1
            continue
        k = _nearest_idx(t_front, t)
        if abs(t_front[k] - t) > tol:
            n_late += 1
            continue

        o = odom[j]
        R_wb = quat_to_R(o["quat"])
        yaw = yaw_of(R_wb)
        x, y = o["pos"][0], o["pos"][1]

        # bỏ frame robot gần như đứng yên so với frame trước (log thật rất nhiều frame trùng)
        if frames:
            prev = frames[-1]
            d = math.hypot(x - prev.x, y - prev.y)
            dyaw = abs(math.atan2(math.sin(yaw - prev.yaw), math.cos(yaw - prev.yaw)))
            if d < args.min_move and math.degrees(dyaw) < args.min_turn_deg:
                n_static += 1
                continue

        # ---- pose camera trong hệ world, SÀN Ở z = 0 ----
        if args.pose_mode == "synth":
            T_wc = camera_pose_synth(x, y, yaw, args.height_cm / 100.0, args.pitch2)
        else:
            T_wb = np.eye(4)
            T_wb[:3, :3] = R_wb
            T_wb[:3, 3] = o["pos"]
            T_wb[2, 3] += pelvis_h  # dịch gốc world lên: sàn thành z = 0
            T_wc = (T_wb @ tf.resolve_camera(cam_frame, args.base_frame, t)[0]).astype(np.float32)

        cl = None
        if len(t_cloud):
            m = _nearest_idx(t_cloud, t)
            if abs(t_cloud[m] - t) <= max(tol, int(120e6)):  # cloud thường 10 Hz → nới ngưỡng
                cl = (idx_cloud[m][1], idx_cloud[m][2])

        frames.append(Frame(t, (idx_front[k][1], idx_front[k][2]), (path, rowid), cl, x, y, yaw, T_wc))

    print("\n── B. Đồng bộ & chọn keyframe ──────────────────────────")
    print(f"   ảnh cúi trong bag : {len(idx_down)}")
    print(f"   keyframe giữ lại  : {len(frames)}")
    print(f"   bỏ do lệch giờ    : {n_late}")
    print(f"   bỏ do đứng yên    : {n_static}  (ngưỡng {args.min_move} m / {args.min_turn_deg}°)")
    if len(frames) >= 2:
        d = [math.hypot(frames[i].x - frames[i - 1].x, frames[i].y - frames[i - 1].y) for i in range(1, len(frames))]
        print(f"   bước giữa keyframe: trung bình {np.mean(d):.3f} m (data gốc R2R = 0.25 m)")
    return frames


# =============================================================================
# GIAI ĐOẠN C — SINH NHÃN
# =============================================================================
def discretize_actions(frames: List[Frame], min_turn_deg: float) -> List[int]:
    """Quỹ đạo liên tục → "nút bấm" `{-1, 1, 2, 3}`.

    `action[i]` = việc đã làm để đi từ frame `i-1` TỚI frame `i`; `action[0] = -1` (mốc khởi đầu).
    Loader tự dịch trái một nhịp (`actions[1:] + [0]`) nên frame cuối thành STOP.
    """
    actions = [-1]
    for i in range(1, len(frames)):
        dyaw = math.degrees(
            math.atan2(math.sin(frames[i].yaw - frames[i - 1].yaw), math.cos(frames[i].yaw - frames[i - 1].yaw))
        )
        actions.append((2 if dyaw > 0 else 3) if abs(dyaw) >= min_turn_deg else 1)
    return actions


def find_subgoal_frames(frames: List[Frame], actions: List[int], subgoal_dist: float) -> List[int]:
    """Chọn các frame làm "đích trung gian" (sub-goal).

    Hai luật cộng lại (khác `mcap2s2.py`, vốn chỉ có luật 1):
      1. **Điểm rẽ** — frame kết thúc một đoạn đi thẳng (ý nghĩa: "đi thẳng tới chỗ rẽ").
      2. **Mỗi `subgoal_dist` mét đường đi** — vì log thật có thể đi thẳng rất dài mà không rẽ
         (bag mẫu: đi thẳng 4.58 m, KHÔNG có cú rẽ nào → nếu chỉ dùng luật 1 thì cả episode chỉ có
         một sub-goal ở cuối, `k` lên tới cả trăm frame → cửa sổ ảnh khổng lồ, lệch hẳn data gốc
         nơi `k` chỉ cỡ 4–25).
      + luôn thêm frame cuối cùng.
    Data gốc R2R dùng "viewpoint" của đồ thị điều hướng (~1–2 m một điểm) — nên mặc định 1.5 m.
    """
    subgoals = set()
    for i in range(1, len(actions)):
        if actions[i] in (2, 3) and actions[i - 1] == 1:
            subgoals.add(i)
    acc = 0.0
    for i in range(1, len(frames)):
        acc += math.hypot(frames[i].x - frames[i - 1].x, frames[i].y - frames[i - 1].y)
        if acc >= subgoal_dist:
            subgoals.add(i)
            acc = 0.0
    subgoals.add(len(frames) - 1)
    return sorted(subgoals)


def project_to_pixel(T_world_cam: np.ndarray, K: np.ndarray, p_world: np.ndarray, w: int, h: int):
    """Chiếu điểm 3D (hệ world) xuống ảnh. Trả về `(u, v, nhìn_thấy)`.

        P_cam = Rᵀ · (P_world − C);  u = fx·X/Z + cx;  v = fy·Y/Z + cy
    """
    p_cam = T_world_cam[:3, :3].T @ (p_world - T_world_cam[:3, 3])
    if p_cam[2] <= 1e-6:  # sau lưng camera
        return -1, -1, False
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    if not (0 <= u < w and 0 <= v < h):
        return -1, -1, False
    return int(round(u)), int(round(v)), True


def make_labels(frames: List[Frame], K_out: np.ndarray, cfg, args) -> dict:
    n = len(frames)
    actions = discretize_actions(frames, args.min_turn_deg)
    subgoals = find_subgoal_frames(frames, actions, args.subgoal_dist)

    poses = np.stack([f.T_world_cam for f in frames]).astype(np.float32)
    goals = np.full((n, 2), -1, dtype=np.int32)
    rel = np.full((n,), -1, dtype=np.int32)

    for t in range(n):
        g = next((s for s in subgoals if s > t), None)
        if g is None or (g - t) > args.max_goal_frames:
            continue
        # waypoint = vị trí chân robot ở frame tương lai `g`, TRÊN SÀN (z = 0)
        p_world = np.array([frames[g].x, frames[g].y, 0.0], dtype=np.float64)
        u, v, ok = project_to_pixel(poses[t], K_out, p_world, cfg.out_w, cfg.out_h)
        if ok:
            goals[t] = (u, v)
            rel[t] = g - t

    return {"action": np.array(actions, np.int32), "pose": poses, "goal": goals, "rel_id": rel,
            "subgoals": subgoals}


# =============================================================================
# GIAI ĐOẠN D — GHI ẢNH (+ dựng depth từ point cloud)
# =============================================================================
def decode_jpeg(blob: bytes, bgr_swap: bool) -> Image.Image:
    msg = msg_compressed_image(blob)
    im = Image.open(io.BytesIO(msg["data"])).convert("RGB")
    if bgr_swap:
        im = Image.fromarray(np.array(im)[:, :, ::-1])
    return im


def cloud_to_depth_mm(blob: bytes, K: np.ndarray, cfg, fill: bool) -> np.ndarray:
    """`PointCloud2` → ảnh depth uint16 **milimét** khớp với ảnh RGB.

    Vì sao phải làm vậy? Log robot thật thường **không có topic ảnh depth**, chỉ có point cloud của
    ZED. Nhưng loader S2 **bắt buộc file depth tồn tại** (`04_data_train_s2.md` mục 6.3).

    Ba bước: (1) đọc x,y,z (hệ *thân* camera: x trước, y trái, z lên) → (2) xoay sang hệ *quang học*
    (x phải, y xuống, z trước) → (3) chiếu bằng `K` (đã hiệu chỉnh theo crop/resize), giữ điểm GẦN
    NHẤT cho mỗi pixel.
    """
    w, h = cfg.out_w, cfg.out_h
    pc = msg_point_cloud2(blob)
    step = pc["point_step"]
    arr = np.frombuffer(pc["data"], dtype=np.uint8).reshape(-1, step)
    off = {f["name"]: f["offset"] for f in pc["fields"]}
    xyz = np.stack([arr[:, off[c] : off[c] + 4].copy().view(np.float32).ravel() for c in ("x", "y", "z")], axis=1)

    ok = np.isfinite(xyz).all(axis=1)
    p = xyz[ok] @ R_OPT_BODY.T  # thân → quang học
    z = p[:, 2]
    keep = z > 0.05
    p, z = p[keep], z[keep]
    u = np.round(K[0, 0] * p[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * p[:, 1] / z + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[inside]

    depth = np.full((h, w), np.inf, np.float32)
    np.minimum.at(depth, (v, u), z)  # điểm gần nhất thắng (che khuất đúng cách)
    valid = np.isfinite(depth)
    depth[~valid] = 0.0

    if fill and valid.any():
        from scipy.ndimage import distance_transform_edt

        _, (iy, ix) = distance_transform_edt(~valid, return_distances=True, return_indices=True)
        depth = depth[iy, ix]  # lấp lỗ bằng pixel hợp lệ gần nhất

    return np.clip(depth * 1000.0, 0, DEPTH_MAX_MM).astype(np.uint16)


def center_crop_box(src_w: int, src_h: int, out_w: int, out_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Hộp cắt GIỮA đưa ảnh (src_w × src_h) về đúng tỉ lệ đích. None = đã đúng tỉ lệ, không cần cắt.

    Cắt giữa chứ không cắt từ góc: phần bỏ đi chia đều hai bên (hoặc trên/dưới) nên trục quang
    vẫn nằm giữa khung, ảnh không bị lệch.
    """
    if abs(src_w / src_h - out_w / out_h) < 1e-6:
        return None
    if src_w / src_h > out_w / out_h:  # nguồn rộng hơn → cắt hai bên
        cw = int(round(src_h * out_w / out_h))
        x0 = (src_w - cw) // 2
        return (x0, 0, x0 + cw, src_h)
    ch = int(round(src_w * out_h / out_w))  # nguồn cao hơn → cắt trên/dưới
    y0 = (src_h - ch) // 2
    return (0, y0, src_w, y0 + ch)


def fit_image(im: Image.Image, cfg) -> Image.Image:
    """Đưa ảnh về kích thước đích theo đúng chế độ `--out-fit`.

    🔴 Hộp cắt phải tính lại theo kích thước THẬT của từng ảnh. Hai camera có độ phân giải khác
    nhau (đầu 1920×1080, bụng 960×600); dùng chung một hộp cắt cố định — vốn tính cho camera bụng
    — sẽ cắt trúng góc trái-trên của ảnh camera đầu, tức lệch tâm và sai góc nhìn.
    """
    if cfg.out_fit == "crop":
        box = center_crop_box(im.width, im.height, cfg.out_w, cfg.out_h)
        if box is not None:
            im = im.crop(box)
    return im.resize((cfg.out_w, cfg.out_h), Image.BILINEAR)


def write_images(bag: Bag, ep_idx: int, frames: List[Frame], scene_dir: str, cfg, args, K_down: np.ndarray) -> int:
    """Ghi 3 luồng ảnh đúng tên thư mục/file mà loader ghép ra (dòng 1014-1022)."""
    chunk = f"chunk-{ep_idx // 1000:03d}"
    base = os.path.join(scene_dir, "videos", chunk)
    d_front = os.path.join(base, f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p1_tag}deg")
    d_down = os.path.join(base, f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p2_tag}deg")
    d_depth = os.path.join(base, f"observation.images.depth.{cfg.h_tag}cm_{cfg.p2_tag}deg")
    for d in (d_front, d_down, d_depth):
        os.makedirs(d, exist_ok=True)

    n_zero_depth = 0
    for i, fr in enumerate(frames):
        stem = f"episode_{ep_idx:06d}_{i}"
        im_down = fit_image(decode_jpeg(bag.fetch(*fr.rgb_down), args.bgr_swap), cfg)
        im_down.save(os.path.join(d_down, f"{stem}.jpg"), quality=args.jpeg_quality)
        # Cấu hình 1 camera (pitch_1 == pitch_2): loader vẫn mở CẢ HAI đường dẫn → ghi cùng nội dung.
        im_front = im_down if fr.rgb_front == fr.rgb_down else fit_image(
            decode_jpeg(bag.fetch(*fr.rgb_front), args.bgr_swap), cfg
        )
        im_front.save(os.path.join(d_front, f"{stem}.jpg"), quality=args.jpeg_quality)

        if fr.cloud is not None:
            depth = cloud_to_depth_mm(bag.fetch(*fr.cloud), K_down, cfg, args.depth_fill)
        else:
            depth = np.zeros((cfg.out_h, cfg.out_w), np.uint16)
            n_zero_depth += 1
        Image.fromarray(depth).save(os.path.join(d_depth, f"{stem}.png"), optimize=True)
    return n_zero_depth


# =============================================================================
# GIAI ĐOẠN E — GHI PARQUET + META
# =============================================================================
def write_parquet(ep_idx: int, labels: dict, scene_dir: str, cfg, task_index: int, index_offset: int,
                  timestamps: np.ndarray):
    """dtype phải khớp bản gốc `vln_ce` — loader gọi `.tolist()` trên từng ô nên kiểu Python thuần vỡ ngay.

    `timestamp` ghi **thời gian thật lấy từ bag** (giây, tính từ đầu episode), KHÔNG phải `i/fps`:
    keyframe đã bị lọc thưa nên khoảng cách giữa các frame không đều. Chính vì vậy tài liệu luôn dặn
    "dùng cột `timestamp`, đừng suy từ `fps`".
    """
    n = len(labels["action"])
    s = cfg.setting
    table = pa.table(
        {
            "action": pa.array(labels["action"], type=pa.int32()),
            f"pose.{s}": pa.array([p.tolist() for p in labels["pose"]], type=pa.list_(pa.list_(pa.float32()))),
            f"goal.{s}": pa.array([g.tolist() for g in labels["goal"]], type=pa.list_(pa.int32(), 2)),
            f"relative_goal_frame_id.{s}": pa.array(labels["rel_id"], type=pa.int32()),
            "timestamp": pa.array(np.asarray(timestamps, dtype=np.float32), type=pa.float32()),
            "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(np.full(n, ep_idx, np.int64), type=pa.int64()),
            "index": pa.array(np.arange(index_offset, index_offset + n, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array(np.full(n, task_index, np.int64), type=pa.int64()),
        }
    )
    d = os.path.join(scene_dir, "data", f"chunk-{ep_idx // 1000:03d}")
    os.makedirs(d, exist_ok=True)
    pq.write_table(table, os.path.join(d, f"episode_{ep_idx:06d}.parquet"))


def write_meta(scene_dir: str, instructions: List[str], lengths: List[int], all_labels: List[dict], cfg, fps: float):
    """Loader S2 chỉ đọc `episodes.jsonl`; ba file kia để dataset đúng chuẩn LeRobot v2.1."""
    meta = os.path.join(scene_dir, "meta")
    os.makedirs(meta, exist_ok=True)
    s = cfg.setting

    with open(os.path.join(meta, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for i, (ins, n) in enumerate(zip(instructions, lengths)):
            f.write(json.dumps({"episode_index": i, "tasks": [ins], "length": n}, ensure_ascii=False) + "\n")
    with open(os.path.join(meta, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for i, ins in enumerate(instructions):
            f.write(json.dumps({"task_index": i, "task": ins}, ensure_ascii=False) + "\n")
    with open(os.path.join(meta, "episodes_stats.jsonl"), "w", encoding="utf-8") as f:
        for i, lab in enumerate(all_labels):
            n = len(lab["action"])

            def st(a):
                a = np.asarray(a, np.float64).reshape(n, -1)
                return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
                        "std": a.std(0).tolist(), "count": [n]}

            f.write(json.dumps({"episode_index": i, "stats": {
                "action": st(lab["action"]), f"goal.{s}": st(lab["goal"]),
                f"relative_goal_frame_id.{s}": st(lab["rel_id"])}}) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "custom",
        "total_episodes": len(lengths),
        "total_frames": int(sum(lengths)),
        "total_tasks": len(instructions),
        "total_chunks": (len(lengths) - 1) // 1000 + 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(lengths)}"},
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
# GIAI ĐOẠN F — TỰ KIỂM ĐỊNH
# =============================================================================
def self_check(scene_dir: str, cfg, sample_step: int = 4) -> None:
    """Mô phỏng ĐÚNG logic cắt mẫu của `NavPixelGoalDataset` (dòng 857-947) để đếm mẫu train."""
    s = cfg.setting
    n_goal = n_turn = n_stop = n_short = missing = 0
    for ep_idx, path in enumerate(sorted(glob.glob(os.path.join(scene_dir, "data", "chunk-*", "episode_*.parquet")))):
        df = pq.read_table(path).to_pandas()
        for col in (f"pose.{s}", f"goal.{s}", f"relative_goal_frame_id.{s}"):
            if col not in df.columns:
                raise SystemExit(f"❌ Thiếu cột {col} → loader sẽ bỏ nguyên scene.")
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
                n_turn += int(actions[start] != 1)
            elif rel[start] < 3:
                n_short += 1
            else:
                n_goal += 1
                for fid in range(0, start + rel[start] + 1):
                    for sub, ext in ((f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p1_tag}deg", "jpg"),
                                     (f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p2_tag}deg", "jpg"),
                                     (f"observation.images.depth.{cfg.h_tag}cm_{cfg.p2_tag}deg", "png")):
                        if not os.path.exists(os.path.join(scene_dir, "videos", f"chunk-{ep_idx//1000:03d}", sub,
                                                           f"episode_{ep_idx:06d}_{fid}.{ext}")):
                            missing += 1
        n_stop += 1

    print("\n── F. Tự kiểm định (mô phỏng logic loader) ─────────────")
    print(f"   mẫu pixel_goal : {n_goal}    ← loại quan trọng nhất")
    print(f"   mẫu turn       : {n_turn}")
    print(f"   mẫu stop       : {n_stop}  (loader nhân 5 khi pixel_goal_only=False)")
    print(f"   bị bỏ (k < 3)  : {n_short}")
    print(f"   file ảnh thiếu : {missing}")
    print(f"   → train S2  (pixel_goal_only=False): {n_goal + n_turn + n_stop * 5} mẫu")
    print(f"   → train dual (pixel_goal_only=True): {n_goal} mẫu")
    if n_goal == 0:
        print("   ⚠️  KHÔNG có mẫu pixel_goal! Camera pitch_2 có thấy sàn không? Thử tăng --subgoal-dist.")
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
    out_w: int
    out_h: int
    # Hộp cắt (x0, y0, x1, y1) của riêng camera CÚI — chỉ dùng để hiệu chỉnh `K`.
    # Ảnh thật được cắt trong `fit_image`, tính lại theo kích thước của từng ảnh.
    crop_box: Optional[Tuple[int, int, int, int]] = None
    # Chế độ đưa ảnh về kích thước đích: "crop" (cắt giữa, giữ tỉ lệ) hoặc "stretch" (kéo giãn).
    out_fit: str = "crop"

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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="thư mục bag (có metadata.yaml) hoặc một file .db3")
    ap.add_argument("--inspect", action="store_true", help="chỉ khảo sát bag rồi thoát (LÀM TRƯỚC TIÊN)")
    ap.add_argument("--out", default="./traj_data")
    ap.add_argument("--dataset-name", default="myrobot")
    ap.add_argument("--scene-id", default=None, help="mặc định lấy tên bag")
    ap.add_argument("--instruction", default=None, help="câu lệnh tiếng Anh cho cả bag (BẮT BUỘC)")
    ap.add_argument("--instruction-file", default=None, help='JSON {"0": "câu lệnh ep 0", ...}')

    g = ap.add_argument_group("topic (để trống = tự đoán)")
    g.add_argument("--rgb-down-topic", default=None, help="ảnh camera NHÌN CÚI (pitch_2)")
    g.add_argument("--rgb-front-topic", default=None, help="ảnh camera NHÌN THẲNG (pitch_1)")
    g.add_argument("--caminfo-down-topic", default=None)
    g.add_argument("--caminfo-front-topic", default=None)
    g.add_argument("--depth-topic", default=None, help="PointCloud2 dùng để dựng depth")
    g.add_argument("--odom-topic", default=None)
    g.add_argument("--tf-topic", default="/tf")
    g.add_argument("--tf-static-topic", default="/tf_static")
    g.add_argument("--single-camera", action="store_true", help="dùng camera cúi cho CẢ hai góc")

    g = ap.add_argument_group("hình học (để trống = suy từ TF)")
    g.add_argument("--base-frame", default="base_link")
    g.add_argument("--foot-offset", type=float, default=0.04, help="m — dày bàn chân, để suy chiều cao base")
    g.add_argument("--height-cm", type=float, default=None)
    g.add_argument("--pitch1", type=float, default=None)
    g.add_argument("--pitch2", type=float, default=None)
    g.add_argument("--pose-mode", choices=["tf", "synth"], default="tf",
                   help="tf = pose thật đo từ TF · synth = dựng lại theo đúng khuôn vln_ce")

    g = ap.add_argument_group("lấy mẫu & nhãn")
    g.add_argument("--out-size", default="640x480", help="'640x480' (như data gốc) hoặc 'native'")
    g.add_argument("--out-fit", choices=["crop", "stretch"], default="crop",
                   help="crop = cắt giữa giữ tỉ lệ (không méo, mất chút góc nhìn) · stretch = kéo giãn")
    g.add_argument("--min-move", type=float, default=0.25, help="m — khoảng cách tối thiểu giữa 2 keyframe")
    g.add_argument("--min-turn-deg", type=float, default=10.0)
    g.add_argument("--subgoal-dist", type=float, default=1.5, help="m — đặt một sub-goal mỗi bấy nhiêu mét")
    g.add_argument("--max-goal-frames", type=int, default=30, help="chặn trên cho relative_goal_frame_id")
    g.add_argument("--tol-ms", type=float, default=60.0)
    g.add_argument("--split-sec", type=float, default=None, help="cắt bag dài thành nhiều episode mỗi N giây")

    g = ap.add_argument_group("ảnh")
    g.add_argument("--depth-source", choices=["pointcloud", "zeros"], default="pointcloud")
    g.add_argument("--depth-fill", action="store_true", default=True, help="lấp lỗ depth bằng pixel gần nhất")
    g.add_argument("--no-depth-fill", dest="depth_fill", action="store_false")
    g.add_argument("--bgr-swap", action="store_true",
                   help="đảo kênh R↔B. Mặc định KHÔNG: cv::imencode đã ghi JPEG đúng RGB dù format ghi 'bgr8'")
    g.add_argument("--jpeg-quality", type=int, default=90)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    bag = Bag.open(args.bag)

    if args.inspect:
        inspect(bag, args)
        return

    picks = autodetect_topics(bag, args)
    print("── A. Đọc bag ──────────────────────────────────────────")
    print(f"   {len(bag.paths)} file .db3 · {len(bag.topics)} topic")
    for k, v in picks.items():
        n = bag.topics[v].count if v in bag.topics else 0
        print(f"   {k:<14} = {v}  ({n} msg)")
    for k in ("rgb_down", "caminfo_down", "odom"):
        if not picks[k]:
            raise SystemExit(f"❌ Không tìm được topic cho '{k}'. Chạy --inspect rồi khai tay bằng --{k.replace('_','-')}-topic.")
    if not picks["cloud"] and args.depth_source == "pointcloud":
        print("   ⚠️  Không có PointCloud2 → depth sẽ là ảnh 0 (chỉ dùng được để train S2 thuần).")

    tf = load_tf(bag, args)
    ci_down = msg_camera_info(bag.fetch(*bag.index(picks["caminfo_down"])[0][1:]))
    ci_front = (
        msg_camera_info(bag.fetch(*bag.index(picks["caminfo_front"])[0][1:]))
        if picks["caminfo_front"] and picks["rgb_front"] != picks["rgb_down"]
        else ci_down
    )

    odom_all = bag.read_all(picks["odom"], msg_odometry)
    t_mid = odom_all[len(odom_all) // 2]["t"] if odom_all else 0
    pelvis_h = derive_base_height(tf, args.base_frame, t_mid, args.foot_offset)
    T_down, via_down = tf.resolve_camera(ci_down["frame"], args.base_frame, t_mid)
    T_front, _ = tf.resolve_camera(ci_front["frame"], args.base_frame, t_mid)

    height_cm = args.height_cm if args.height_cm is not None else (T_down[2, 3] + pelvis_h) * 100.0
    pitch2 = args.pitch2 if args.pitch2 is not None else pitch_down_deg(T_down)
    pitch1 = args.pitch1 if args.pitch1 is not None else (pitch2 if args.single_camera else pitch_down_deg(T_front))

    src_w, src_h = ci_down["w"], ci_down["h"]
    if args.out_size == "native":
        out_w, out_h = src_w, src_h
    else:
        out_w, out_h = (int(v) for v in args.out_size.lower().split("x"))

    # ---- Đưa ảnh về kích thước đích: CẮT giữ tỉ lệ, hay KÉO GIÃN? ----
    # `crop` (mặc định): cắt giữa cho đúng tỉ lệ đích rồi resize → hình học tự nhiên, không méo,
    #   đánh đổi là mất một phần góc nhìn ngang. Với bag mẫu: 960×600 (1.60) → cắt 800×600 (1.333)
    #   → hfov 105.7° còn 95.4°, GẦN HƠN với data gốc vln_ce (~90°).
    # `stretch`: giữ toàn bộ góc nhìn nhưng ảnh bị méo. Vẫn đúng về hình học pinhole (fx,cx ×sx và
    #   fy,cy ×sy là biến đổi affine hợp lệ) nên `goal` vẫn chính xác — chỉ khác phân bố ảnh.
    # Mỗi camera có hộp cắt RIÊNG (tính trong `fit_image` theo kích thước từng ảnh). Ở đây chỉ
    # tính hộp của camera CÚI, vì `K` — và do đó nhãn `goal` — gắn với camera đó.
    crop_box = center_crop_box(src_w, src_h, out_w, out_h) if args.out_fit == "crop" else None
    cfg = Config(height_cm, pitch1, pitch2, out_w, out_h, crop_box, args.out_fit)

    # K phải đi theo đúng phép cắt + resize, nếu không phép chiếu pixel-goal sẽ lệch.
    K_out = ci_down["K"].copy()
    if crop_box is not None:
        K_out[0, 2] -= crop_box[0]  # cắt = dịch tâm ảnh
        K_out[1, 2] -= crop_box[1]
        src_w, src_h = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    K_out[0, :] *= out_w / src_w
    K_out[1, :] *= out_h / src_h

    print("\n   ── hình học suy từ TF ──")
    print(f"   base '{args.base_frame}' cao {pelvis_h:.3f} m so với sàn (từ khớp chân + {args.foot_offset} m)")
    print(f"   pitch_2 ({ci_down['frame']}): cúi {pitch2:.1f}° · cao {height_cm/100:.3f} m · qua TF {via_down}")
    print(f"   pitch_1: cúi {pitch1:.1f}°")
    print(f"   setting = {cfg.setting} · ảnh {ci_down['w']}×{ci_down['h']} → {out_w}×{out_h}")
    if args.out_fit == "crop":
        for role, ci in (("pitch_2", ci_down), ("pitch_1", ci_front)):
            b = center_crop_box(ci["w"], ci["h"], out_w, out_h)
            how = f"cắt giữa {b} → {b[2] - b[0]}×{b[3] - b[1]}" if b else "đã đúng tỉ lệ, không cắt"
            print(f"   {role} {ci['w']}×{ci['h']}: {how}")
    print(f"   K sau resize: fx={K_out[0,0]:.1f} fy={K_out[1,1]:.1f} cx={K_out[0,2]:.1f} cy={K_out[1,2]:.1f}")
    if cfg.p2_tag == 0:
        print("   ⚠️  pitch_2 = 0° → camera không thấy sàn → goal sẽ toàn -1. Data gốc KHÔNG dùng cấu hình này.")

    frames = sync_frames(bag, tf, picks, args, pelvis_h, ci_down["frame"])
    if len(frames) < 4:
        raise SystemExit("❌ Chỉ dựng được <4 keyframe. Giảm --min-move hoặc kiểm tra lại --tol-ms.")

    # cắt thành episode
    episodes: List[List[Frame]] = [frames]
    if args.split_sec:
        episodes, cur, t0 = [], [], frames[0].t_ns
        for f in frames:
            if (f.t_ns - t0) / 1e9 >= args.split_sec and len(cur) >= 4:
                episodes.append(cur)
                cur, t0 = [], f.t_ns
            cur.append(f)
        if len(cur) >= 4:
            episodes.append(cur)

    instructions = []
    overrides = json.load(open(args.instruction_file, encoding="utf-8")) if args.instruction_file else {}
    for i in range(len(episodes)):
        ins = overrides.get(str(i)) or args.instruction or ""
        if not ins.strip():
            raise SystemExit(
                "❌ Chưa có câu lệnh ngôn ngữ. System 2 học từ NGÔN NGỮ — thiếu câu lệnh thì mẫu vô nghĩa.\n"
                '   Thêm:  --instruction "Walk straight along the aisle and stop at the end."'
            )
        instructions.append(ins.strip())

    scene_id = args.scene_id or os.path.basename(os.path.normpath(args.bag)).replace(".db3", "")
    scene_dir = os.path.join(args.out, args.dataset_name, scene_id)
    if "rgb" in os.path.abspath(scene_dir).replace("\\", "/").rsplit("/videos", 1)[0]:
        print("   ⚠️  Đường dẫn có chứa chữ 'rgb' → loader dùng .replace('rgb','depth') sẽ hỏng. Hãy đổi tên.")
    os.makedirs(scene_dir, exist_ok=True)

    fps = 1e9 * (len(frames) - 1) / max(1, frames[-1].t_ns - frames[0].t_ns)
    print("\n── C+D+E. Sinh nhãn, ghi ảnh & parquet ─────────────────")
    all_labels, lengths, offset, zero_depth = [], [], 0, 0
    for i, ep in enumerate(episodes):
        labels = make_labels(ep, K_out, cfg, args)
        zero_depth += write_images(bag, i, ep, scene_dir, cfg, args, K_out)
        stamps = np.array([(f.t_ns - ep[0].t_ns) / 1e9 for f in ep])
        write_parquet(i, labels, scene_dir, cfg, task_index=i, index_offset=offset, timestamps=stamps)
        offset += len(ep)
        lengths.append(len(ep))
        all_labels.append(labels)
        vals, cnts = np.unique(labels["action"], return_counts=True)
        hist = " ".join(f"{IDX2NAME.get(int(v), v)}×{c}" for v, c in zip(vals, cnts))
        print(f"   ep {i}: {len(ep):>3} frame · {int((labels['rel_id'] >= 0).sum()):>3} frame có pixel-goal · "
              f"{len(labels['subgoals'])} sub-goal · {hist}")
    write_meta(scene_dir, instructions, lengths, all_labels, cfg, fps)
    if zero_depth:
        print(f"   ⚠️  {zero_depth} frame không ghép được point cloud → depth = 0")
    print(f"\n   ✅ Đã ghi {scene_dir}  (fps ghi vào parquet: {fps:.2f})")

    self_check(scene_dir, cfg)

    name = f"{args.dataset_name}_{cfg.h_tag}cm_{cfg.p1_tag}_{cfg.p2_tag}"
    print(
        "\n── Bước tiếp theo ──────────────────────────────────────\n"
        "   1. Đăng ký vào `data_dict` (internvla_n1_lerobot_dataset.py:127):\n"
        f'        "{name}": {{"data_path": "{os.path.join(args.out, args.dataset_name)}", '
        f'"height": {cfg.h_tag}, "pitch_1": {cfg.p1_tag}, "pitch_2": {cfg.p2_tag}}},\n'
        f"   2. Trỏ `--vln_dataset_use {name}` trong scripts/train/qwenvl_train/train_system2.sh"
    )


if __name__ == "__main__":
    main()
