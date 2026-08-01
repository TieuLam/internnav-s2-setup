# 06 — Pipeline: từ file `.mcap` → data train System 2

> **File này để làm gì:** hướng dẫn **triển khai đầy đủ** đường ống biến log robot `.mcap` thành
> dataset LeRobot mà `NavPixelGoalDataset` nạp được. Kèm **hai script chạy được** trong
> [tools/](tools/), mọi con số dưới đây là **kết quả chạy thật**.
>
> Bộ tài liệu: [04_data_train_s2](04_data_train_s2.md) (hợp đồng dữ liệu) ·
> [03_code_train_s2](03_code_train_s2.md) (loader đọc gì)

---

## 0. Kết luận phải nắm trước khi làm gì cả

### 0.1. Một log "chỉ có pose/imu" là **không đủ**

Đây là bẫy đầu tiên: nhiều bản ghi robot chỉ có các topic điều khiển/cảm biến quán tính
(`pose / imu / battery / log`) — **không có thị giác**. Với loại log đó thì **không thể** sinh data S2:

| Thứ S2 bắt buộc cần | Log chỉ-có-pose/imu |
|---|---|
| Ảnh RGB theo thời gian | ❌ |
| Ảnh Depth | ❌ |
| Intrinsics camera (`fx, fy, cx, cy`) | ❌ — **không có thì không chiếu được pixel-goal** |
| Pose/quỹ đạo robot | ✅ |
| Câu lệnh ngôn ngữ | ❌ |

### 0.2. Một `.mcap` "đủ dùng" cho S2 phải có 5 luồng

| Luồng | Dùng để sinh ra | Bắt buộc? |
|---|---|---|
| **RGB góc `pitch_1`** (nhìn thẳng) | `observation.images.rgb.{H}cm_{pitch_1}deg/` | ✅ |
| **RGB góc `pitch_2`** (nhìn cúi) | `observation.images.rgb.{H}cm_{pitch_2}deg/` | ✅ (trùng luồng trên nếu `pitch_1==pitch_2`) |
| **Depth ở `pitch_2`** | `observation.images.depth.{H}cm_{pitch_2}deg/` | ✅ phải tồn tại |
| **`camera_info`** (ma trận `K`) | phép chiếu 3D→pixel để ra `goal` | ✅ |
| **Pose/TF theo thời gian** | `action`, `pose.{setting}`, waypoint 3D | ✅ |
| **Câu lệnh + ranh giới episode** | `meta/episodes.jsonl` | ✅ (log robot thường **không** có → phải bổ sung ngoài) |

👉 Việc đầu tiên với một file `.mcap` lạ là **đối chiếu nó với bảng trên** — dùng
[tools/mcap_inspect.py](tools/mcap_inspect.py) (mục 2).

> 📌 **Nếu log robot của bạn là `.db3` (rosbag2)** — định dạng mà `ros2 bag record` sinh ra theo mặc
> định — hãy dùng [06c_pipeline_db3_to_s2](06c_pipeline_db3_to_s2.md) + [tools/db32s2.py](tools/db32s2.py)
> thay cho tài liệu này. Ý tưởng 6 giai đoạn giống nhau; khác ở tầng đọc file (SQLite + CDR) và ở chỗ
> hình học camera phải suy từ cây TF.

---

## 1. Toàn cảnh đường ống

```
            log robot thật  (ROS 2 → .mcap)
                   │
                   ▼   tools/mcap2s2.py
   ┌───────────────────────────────────────────────────────────────┐
   │ A. read_mcap    đọc mọi topic → list (log_time, payload)      │
   │ B. sync_frames  đồng bộ thời gian + cắt episode + lọc keyframe│
   │ C. make_labels  action · pose 4×4 · goal (u,v) · rel_id       │  ← trái tim
   │ D. write_images ghi .jpg/.png đúng tên loader mong đợi        │
   │ E. write_lerobot ghi parquet đúng dtype + 4 file meta         │
   │ F. self_check   mô phỏng logic loader để đếm mẫu              │
   └───────────────────────────────────────────────────────────────┘
                   │
                   ▼
    traj_data/<dataset>/<scene>/{meta,data,videos}
                   │
                   ▼  đăng ký data_dict + --vln_dataset_use
              train_system2.sh
```

---

## 2. Bước 1 — Khảo sát file `.mcap` đầu vào

### 2.1. Chạy `mcap_inspect.py`

```bash
pip install mcap
cd docs/training_data_guide/tools
python mcap_inspect.py --mcap log_robot.mcap
```

Nó trả lời đúng ba câu hỏi bạn cần trước khi viết bất cứ dòng code nào: **có topic gì · message type
gì · mã hoá kiểu gì**, kèm nhịp (Hz) từng topic và cây field của từng schema. Không giải mã nội dung
nên chạy trong vài mili giây kể cả file 10 GB (thêm `--deep` nếu muốn nhịp và cỡ byte chính xác).

### 2.2. Đối chiếu với 6 luồng bắt buộc

Lấy bảng ở mục 0.2 làm danh sách kiểm. Với mỗi luồng, ghi lại **tên topic** và **nhịp**:

| Luồng | Topic của bạn | Nhịp | Ghi chú khi thiếu |
|---|---|---|---|
| RGB `pitch_1` | | | thiếu là **không train được** |
| RGB `pitch_2` | | | dùng chung `pitch_1` nếu robot 1 camera (mục 7.4) |
| Depth ở `pitch_2` | | | không có sensor → mục 7.3 |
| `camera_info` (`K`) | | | thiếu là **không chiếu được pixel-goal** |
| Pose / TF | | | |
| Câu lệnh + ranh giới episode | | | log robot hầu như luôn thiếu → `--instruction-file` |

Ba thứ nữa cần chốt ngay ở bước này, vì chúng quyết định **tên thư mục** của dataset:
**chiều cao camera (cm)**, **góc cúi `pitch_1`**, **góc cúi `pitch_2`**.

### 2.3. Quy ước hình học (bám sát data gốc)

Đây là công thức mà `mcap2s2.py` dùng để dựng cột `pose.{setting}` từ `(x, y, yaw)` của robot:

```python
def camera_pose_from_base(x, y, yaw, height_m, pitch_deg):
    z_cam = Rz(yaw) @ (cos p, 0, -sin p)   # trục quang, cúi p độ
    x_cam = Rz(yaw) @ (0, -1, 0)           # "phải" của camera = -y robot
    y_cam = cross(z_cam, x_cam)            # "xuống" (hệ OpenCV, thuận tay phải)
    T[:3,0:3] = [x_cam, y_cam, z_cam] ; T[:3,3] = (x, y, height_m)
```

Kiểm chứng: với `h=0.6, p=30°, yaw=0` công thức cho ra **đúng** ma trận đo thật trong `vln_ce`
([04](04_data_train_s2.md) mục 5.1).

### 2.4. Nếu chưa có log thật để thử

Cách gọn nhất để chạy trọn vòng đời dữ liệu là dùng **bag ROS 2 thật có sẵn trong repo**:
[06c_pipeline_db3_to_s2](06c_pipeline_db3_to_s2.md) — đầy đủ ảnh 2 camera, `camera_info`, odometry,
point cloud, và đã được chạy thử end-to-end.

---

## 3. Bước 2 — Chuyển `.mcap` → dataset S2

```bash
pip install mcap numpy pillow pyarrow
python mcap2s2.py --mcap log_robot.mcap --out ./traj_data \
                  --dataset-name myrobot --scene-id demo_scene
```

Dưới đây là **từng giai đoạn** của [tools/mcap2s2.py](tools/mcap2s2.py).

> 📖 Cần mức chi tiết hơn — **từng hàm một** (nhận gì, trả gì, vì sao viết thế, sửa ở đâu khi dùng
> mcap robot thật)? Xem [09_giai_thich_ham_mcap2s2](09_giai_thich_ham_mcap2s2.md).

### Giai đoạn A — `read_mcap()`: đọc log

Đọc một lượt, gom mỗi topic thành `list[(log_time_ns, payload)]` đã sắp xếp theo thời gian, và lấy
metadata `s2_profile`.

**Kết quả thật:**
```
/camera/front/image_raw          78 message
/camera/lookdown/image_raw       78 message
/camera/lookdown/depth           78 message
/robot/pose                     390 message      ← dày hơn ảnh 5 lần
/camera/lookdown/camera_info      8 message
/task/episode                     4 message
setting = 125cm_30deg · ảnh 640×480 · fx=388.2 cx=320.0
```

Script **dừng ngay với thông báo rõ ràng** nếu thiếu ảnh hoặc thiếu `camera_info` — vì không có `K`
thì không thể chiếu pixel-goal.

### Giai đoạn B — `sync_frames()`: đồng bộ thời gian

> **Vì sao phải có bước này?** Các luồng chạy ở tần số khác nhau. Ghép nhầm ảnh của thời điểm này với
> pose của thời điểm khác → **nhãn sai mà không có lỗi nào báo ra** — kiểu hỏng nguy hiểm nhất.

Nguyên tắc: **luồng RGB `pitch_1` là nhịp chính**. Với mỗi ảnh tại `t`, tìm ảnh cúi / depth / pose
**gần `t` nhất** (tìm nhị phân trên mảng thời gian). Nếu lệch quá `--tol-ms` (mặc định 60 ms) →
**bỏ frame** (thà mất frame còn hơn ghép sai).

Hai việc nữa trong giai đoạn này:
1. **Cắt episode** theo marker `start`/`end` của `/task/episode`. Không có marker → coi cả file là 1 episode.
2. **Lọc keyframe**: bỏ frame mà robot gần như đứng yên so với frame trước (`--min-move 0.05` m,
   `--min-turn-deg 5`). Log robot thật hay có hàng chục frame trùng nhau lúc chờ.

Episode < 4 frame bị loại (khớp điều kiện `if actions_len < 4: continue` của loader).

**Kết quả thật:** `episode giữ lại: 2 · frame mỗi ep: [40, 38] · bỏ do lệch giờ: 0 · bỏ do đứng yên: 0`

### Giai đoạn C — `make_labels()`: sinh nhãn (trái tim)

#### C.1. `action` — quỹ đạo liên tục → "nút bấm"

```python
actions = [-1]                       # frame 0 luôn là -1 (mốc khởi đầu)
for i in 1..n-1:
    dyaw = góc quay giữa frame i-1 và i
    actions.append(2 if dyaw > +ngưỡng else 3 if dyaw < -ngưỡng else 1)
```
Quy ước: **`action[i]` = việc đã làm để đi từ frame `i-1` TỚI frame `i`**. Loader sẽ tự dịch trái một
nhịp (`actions[1:] + [0]`) nên frame cuối thành `STOP`.

#### C.2. `pose.{setting}` — ma trận 4×4

Dùng `camera_pose_from_base(x, y, yaw, height_cm/100, pitch_2)` (mục 2.3). **Chú ý dùng `pitch_2`**,
vì `setting` mô tả camera nhìn cúi.

#### C.3. `goal` + `relative_goal_frame_id` — phần khó nhất

> **🎯 Pixel goal là gì?** System 2 không xuất toạ độ 3D. Nó nhìn ảnh hiện tại và chỉ ra **"điểm nên
> đi tới nằm ở đâu TRÊN TẤM ẢNH"**. Điểm đó thực chất là **vị trí tương lai của robot** được **chiếu
> ngược vào khung ảnh hiện tại**.

**Bước 1 — chọn sub-goal.** Quy tắc dùng trong script: mỗi khi robot **kết thúc một đoạn đi thẳng**
(chuyển từ `1` sang xoay) thì frame đó là một sub-goal; cộng thêm frame cuối episode.
Ý nghĩa trực quan: *"đi thẳng tới chỗ rẽ"*.

> 📌 Data gốc R2R dùng các **viewpoint của đồ thị điều hướng** làm sub-goal. Ta không có đồ thị đó nên
> dùng điểm rẽ làm xấp xỉ. **Đây là chỗ bạn nên thay bằng luật riêng nếu có thông tin tốt hơn**
> (ví dụ: các điểm dừng do người vận hành đánh dấu).

**Bước 2 — chiếu xuống ảnh.** Waypoint = **vị trí chân robot ở frame tương lai `g`, nằm trên sàn
(z = 0)**:

```python
P_cam = Rᵀ · (P_world − C)                    # đưa điểm về hệ camera
if P_cam.z <= 0:            → không thấy      # điểm ở SAU lưng camera
u = fx·X/Z + cx ;  v = fy·Y/Z + cy
if không nằm trong [0,W)×[0,H):  → không thấy # rơi ra ngoài khung
```

**Bước 3 — ghi nhãn.**

| Tình huống | `goal` | `relative_goal_frame_id` |
|---|---|---|
| Thấy waypoint | `[u, v]` | `k = g − t` |
| Không thấy (sau lưng / ngoài khung / quá gần) | `[-1, -1]` | `-1` |

Ba giá trị `k` và số phận của chúng trong loader:

| `k` | Ý nghĩa | Loader làm gì |
|---|---|---|
| `-1` | không có waypoint nhìn thấy | mẫu **turn** (hoặc bỏ nếu đang đi thẳng) |
| `k ≥ 3` | waypoint hợp lệ | mẫu **pixel_goal** — cửa sổ `[t, t+k+1]` |
| `0 < k < 3` | waypoint quá gần | **bị bỏ** (`if goal_len < 3: continue`) |

**Kết quả thật — nhãn của episode 0** (đọc lại từ parquet đã ghi):

```
action : [-1, 1,1,1,1,1,1,1,1,1,1, 3,3,3, 1,1,1,1,1,1, 2,2,2, 1,...]
rel_id : [11,10, 9, 8, 7, 6, 5, 4, -1,-1,-1,  9, 8, 7,  6, 5, 4, -1,-1,-1, 13,12,11,10,...]
goal   : [[320,217],[320,234],[320,254],[320,278],[320,307],[320,344],[320,392],[320,455],
          [-1,-1],[-1,-1],[-1,-1],[486,336],[400,314],[320,307],...]
```

Đọc bảng này để hiểu hành vi:
- `rel_id` **giảm dần** `11,10,9,…` = nhiều frame liên tiếp cùng nhắm **một** sub-goal.
- `u = 320` (giữa ảnh) khi robot đi thẳng về phía đích — đúng hình học.
- `v` **tăng dần** (217 → 455): càng tới gần, điểm đích càng **trôi xuống dưới ảnh**.
- Ba `-1` liên tiếp trước mỗi cú rẽ: waypoint đã **quá gần**, rơi khỏi mép dưới khung hình → đúng
  tình huống "không còn gì để nhắm, chuẩn bị rẽ".
- Sau khi rẽ, `goal` lệch sang phải (`486, 400`) rồi về giữa — vì robot đang xoay dần về hướng mới.

→ Mẫu hình này **trùng với dạng của data gốc** (`[25,24,23,11,10,9,8,-1,7,-1,15,…]`).

### Giai đoạn D — `write_images()`

Ghi đúng tên thư mục/file mà loader ghép ra
([dòng 1014-1022](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L1014)):

```
videos/chunk-000/observation.images.rgb.125cm_0deg/episode_000000_0.jpg
videos/chunk-000/observation.images.rgb.125cm_30deg/episode_000000_0.jpg
videos/chunk-000/observation.images.depth.125cm_30deg/episode_000000_0.png
```

Mẹo nhỏ: nếu message đã là JPEG thì **ghi thẳng byte gốc**, không giải nén–nén lại → nhanh hơn và
không mất chất lượng. Depth được kiểm tra **phải là uint16**, không thì script dừng với thông báo rõ.

### Giai đoạn E — `write_parquet()` + `write_meta()`

Ghi parquet với **đúng dtype** (lý do: [04](04_data_train_s2.md) mục 3.3):

```python
"action":                       pa.int32()
f"pose.{s}":                    pa.list_(pa.list_(pa.float32()))   # 4×4
f"goal.{s}":                    pa.list_(pa.int32(), 2)
f"relative_goal_frame_id.{s}":  pa.int32()
"timestamp": float32 · "frame_index"/"episode_index"/"index"/"task_index": int64
```

`write_meta()` ghi cả 4 file `meta/` (loader S2 chỉ cần `episodes.jsonl`, ba file kia để dataset đúng
chuẩn LeRobot v2.1).

### Giai đoạn F — `self_check()`

Đọc lại dataset vừa ghi và **mô phỏng đúng logic cắt mẫu của loader** để đếm số mẫu, đồng thời kiểm
tra **mọi file ảnh trong cửa sổ `[0, start+k]` có tồn tại không**.

**Kết quả thật:**
```
mẫu pixel_goal : 14    ← loại quan trọng nhất
mẫu turn       : 3
mẫu stop       : 2   (loader nhân 5 khi pixel_goal_only=False)
bị bỏ (k < 3)  : 0
file ảnh thiếu : 0
→ train S2  (pixel_goal_only=False): 27 mẫu
→ train dual (pixel_goal_only=True): 14 mẫu
```

---

## 4. Bước 3 — Kiểm định bằng **chính loader thật**

Đây là bước quan trọng nhất: đừng tin script của mình, hãy để code của dự án phán quyết. Loader import
`decord`/`torchcodec` (nặng, hay thiếu) → **giả lập hai module đó** là import được:

```python
import sys, types
for m in ['decord', 'torchcodec', 'torchcodec.decoders']:
    sys.modules[m] = types.ModuleType(m)
sys.modules['decord'].VideoReader = object
sys.modules['torchcodec.decoders'].VideoDecoder = object
sys.modules['torchcodec'].decoders = sys.modules['torchcodec.decoders']

import internnav.dataset.internvla_n1_lerobot_dataset as L
L.data_dict['myrobot_125cm_0_30'] = {"data_path": "traj_data/myrobot",
                                     "height": 125, "pitch_1": 0, "pitch_2": 30}
class A: pass
a = A(); a.vln_dataset_use='myrobot_125cm_0_30'; a.model_type='qwen2.5vl'
a.sample_step=4; a.predict_step_num=32; a.pixel_goal_only=False
a.num_future_steps=4; a.num_history=8; a.image_processor=None; a.transform_train=None
print(len(L.NavPixelGoalDataset(tokenizer=None, data_args=a)))
```

**Kết quả thật:** `27` (và `14` khi `pixel_goal_only=True`) — **trùng khớp** con số của giai đoạn F.
Nghĩa là dataset sinh ra **đúng hợp đồng dữ liệu**.

Kiểm định thứ hai — **round-trip quy ước hệ toạ độ**:

```python
from internnav.dataset.internvla_n1_lerobot_dataset import get_trajectory_relative_to_frame
poses = np.array([np.stack(p) for p in df['pose.125cm_30deg'].tolist()])
print(get_trajectory_relative_to_frame(poses, camera_deg=30)[:12])
# [[0. 0. 0.] [0.25 0. 0.] [0.5 0. 0.] ... [2.5 0. -0.262]]   ← 0.25 m/bước, 15°/nấc ✓
```

Kiểm định thứ ba — **nhìn bằng mắt**: vẽ chấm `goal` lên ảnh và xem nó có nằm đúng hướng đi không.
(Đã làm: chấm đỏ nằm trên sàn phía trước, trôi dần xuống dưới khi robot tiến lại gần — đúng như mong đợi.)

---

## 5. Bước 4 — Đăng ký & train

Script in sẵn hai dòng cần làm ở cuối:

**(1)** Thêm entry vào `data_dict`
([internvla_n1_lerobot_dataset.py:127](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L127)):

```python
MYROBOT_125CM_0_30 = {"data_path": "traj_data/myrobot", "height": 125, "pitch_1": 0, "pitch_2": 30}
data_dict = {..., "myrobot_125cm_0_30": MYROBOT_125CM_0_30}
```

**(2)** Trỏ dataset trong [train_system2.sh](../../../code/scripts/train/qwenvl_train/train_system2.sh):

```bash
vln_datasets=myrobot_125cm_0_30
```

Rồi chạy vài chục step trên 1 GPU để xem loss có hữu hạn và giảm không. Cách fine-tune từ checkpoint
`InternVLA-N1-System2` thay vì Qwen gốc: [03](03_code_train_s2.md) mục 7.1.

---

## 6. Chạy trọn vẹn hai bước (copy-paste)

```bash
cd InternNav/internnav-s2-setup/docs/training_data_guide/tools
pip install mcap numpy pillow pyarrow

python mcap_inspect.py --mcap log_robot.mcap            # bước 1: khảo sát, đối chiếu mục 2.2
python mcap2s2.py --mcap log_robot.mcap --out ./traj_data \
                  --dataset-name myrobot --scene-id run01 \
                  --instruction-file instructions.json
```

Rồi kiểm định bằng loader thật (mục 4) trước khi tin dataset.

> 💡 Muốn chạy thử ngay trên dữ liệu có sẵn trong repo (bag ROS 2 thật, đã kiểm định end-to-end):
> [06c_pipeline_db3_to_s2](06c_pipeline_db3_to_s2.md) mục 10.

---

## 7. Áp dụng cho `.mcap` của **robot thật**

### 7.1. Ánh xạ topic

```bash
python mcap2s2.py --mcap run_01.mcap \
    --topic-rgb-front /camera/color/image_raw \
    --topic-rgb-down  /camera_down/color/image_raw \
    --topic-depth     /camera_down/aligned_depth_to_color/image_raw \
    --topic-caminfo   /camera_down/color/camera_info \
    --topic-pose      /odom \
    --height-cm 125 --pitch1 0 --pitch2 30 \
    --instruction-file instructions.json
```

### 7.2. Bốn chỗ **phải sửa code** khi dùng log ROS 2 thật

| Vấn đề | Vì sao | Sửa thế nào |
|---|---|---|
| Message mã hoá **CDR** (ROS 2), không phải JSON | `json.loads(msg.data)` sẽ vỡ | `pip install mcap-ros2-support`, thay bằng `Ros2Reader`/decoder tương ứng trong `read_mcap()` |
| Ảnh là `sensor_msgs/Image` **thô** (không nén) | không có trường `data` base64 | dựng numpy từ `msg.data` + `msg.encoding` (`rgb8`/`bgr8`/`16UC1`) rồi encode JPEG/PNG trong `write_images()` |
| Pose nằm ở `/tf` chứ không phải một topic pose | `/tf` là cây biến đổi, phải tra chuỗi `map → base_link` | dùng `tf2` để tổng hợp, hoặc dùng `/odom` nếu độ chính xác đủ |
| Không có `/task/episode` | log robot không biết "câu lệnh" là gì | dùng `--instruction-file` (JSON `{"0": "câu lệnh ep 0", …}`); ranh giới episode thì cắt tay theo thời gian hoặc theo lần bấm nút nhiệm vụ |

### 7.3. Nếu robot **không có camera depth**

Loader **bắt buộc file depth tồn tại** nhưng (khi chỉ train S2) **không dùng giá trị**
([04](04_data_train_s2.md) mục 6.3). Ba lựa chọn:

| Cách | Ưu | Nhược |
|---|---|---|
| Chạy **DepthAnythingV2** (chế độ metric) sinh depth uint16 mm | dùng lại được cho train dual sau này | tốn GPU, sai số |
| Ghi **ảnh uint16 giữ chỗ** (toàn 0 hoặc hằng số) | rẻ nhất, đủ để train S2 | **không dùng được cho train dual** |
| Vá loader bỏ qua depth khi `pixel_goal_only=False` | sạch nhất về mặt đĩa | xâm lấn code lõi — không khuyến khích |

### 7.4. Nếu robot chỉ có **một camera**

Chọn cấu hình `pitch_1 == pitch_2` (như `60cm_30_30`): loader sẽ dùng **cùng một file** cho cả ảnh
thẳng lẫn ảnh cúi ([04](04_data_train_s2.md) mục 2.1) → chỉ cần **2 thư mục ảnh**, **1 camera thật**.
Nhớ gắn camera **chúc xuống** đủ để thấy sàn phía trước, nếu không `goal` sẽ toàn `-1`.

---

## 8. Bảng rủi ro & cách phát hiện

| Rủi ro | Hậu quả | Cách phát hiện sớm |
|---|---|---|
| Sai tên cột (dùng `pitch_1` thay `pitch_2`) | scene bị bỏ **im lặng**, dataset rỗng | giai đoạn F báo lỗi thiếu cột; hoặc `len(NavPixelGoalDataset)==0` |
| Sai dtype parquet | `AttributeError` bị `try/except` nuốt → dataset rỗng | chạy kiểm định mục 4 |
| Depth không phải uint16 | loader chia 1000 ra số vô nghĩa | `mcap2s2.py` **dừng ngay** với thông báo |
| Đơn vị depth là mét thay vì milimét | độ sâu lệch 1000 lần | in `min/max` của ảnh depth: phải cỡ hàng trăm–hàng nghìn |
| Camera nhìn thẳng, không thấy sàn | `goal` toàn `-1` → **0 mẫu pixel_goal** | giai đoạn F cảnh báo `⚠️ KHÔNG có mẫu pixel_goal nào!` |
| Lệch thời gian giữa ảnh và pose | nhãn sai **không báo lỗi** | giai đoạn B in số frame "bỏ do lệch giờ"; giảm `--tol-ms` để siết |
| Sai quy ước hệ toạ độ pose | nhãn quỹ đạo dual sai | round-trip test (mục 4) |
| Thư mục cha chứa chữ `rgb` | đường dẫn depth hỏng | giai đoạn F báo "file ảnh thiếu" |
| Episode quá ngắn (< 4 frame) | bị loại | giai đoạn B in số frame mỗi episode |

---

## 9. Lộ trình triển khai theo phase

| Phase | Việc | Thời lượng ước tính | Xong khi… |
|---|---|---|---|
| **0. Khảo sát** | Liệt kê topic/schema của mcap **thật**; xác nhận có RGB / depth / camera_info / pose; xác định nguồn câu lệnh; lấy `K` và thông số lắp camera | 0.5 ngày | điền xong bảng ở mục 0.2 |
| **1. Đọc + đồng bộ** | Ánh xạ topic vào `mcap2s2.py`; chạy tới hết giai đoạn B | 1 ngày | in ra số frame mỗi episode hợp lý |
| **2. Nhãn hình học** | Chỉnh luật chọn sub-goal cho hợp lộ trình của bạn; kiểm biên (sau lưng camera, `k<3`) | 1–2 ngày | round-trip test đạt; vẽ goal lên ảnh thấy hợp lý |
| **3. Ghi & kiểm định** | Chạy trọn A→F; kiểm bằng loader thật | 0.5 ngày | `len(NavPixelGoalDataset) > 0` |
| **4. Train thử** | Đăng ký `data_dict`, chạy ~50 step trên 1 GPU | 0.5 ngày | loss hữu hạn và giảm |
| **5. Mở rộng** | Chạy hàng loạt nhiều mcap; cân bằng tỉ lệ turn/stop/goal; thêm cấu hình camera | tuỳ dữ liệu | đủ lượng data để fine-tune |

---

## 10. Ghi chú kỹ thuật

- **`sample_step=4` lúc train** nghĩa là cứ 4 frame mới lấy 1 mẫu. Nếu log của bạn 30 Hz mà robot đi
  chậm, hãy **giảm tần số keyframe lúc sinh data** (tăng `--min-move`) thay vì để loader bỏ phí.
- **Kích thước ảnh**: giữ nguyên độ phân giải gốc (640×480 như data gốc). Model tự resize về 384×384.
- **Nhiều mcap → nhiều scene**: mỗi lần chạy sinh **một scene**; đặt `--scene-id` khác nhau, cùng
  `--dataset-name`. Loader tự quét mọi thư mục con của `data_path`
  ([dòng 761](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L761)).
- **Nhiều câu lệnh cho cùng một lượt đi**: nối bằng `<INSTRUCTION_SEP>` trong `episodes.jsonl` —
  loader sẽ nhân thành nhiều episode train (tăng dữ liệu miễn phí).
- **Không dùng lại `scripts/dataset_converters/vlnce2lerobot.py`**: converter đó cần thư viện
  `lerobot` và chỉ ghi `observation.images.rgb` + `action` — **thiếu** `pose/goal/relative_goal_frame_id`
  và thiếu quy ước tên `{setting}`. Ghi thẳng bằng `pyarrow` như `mcap2s2.py` vừa ít phụ thuộc vừa
  kiểm soát được dtype.

---

Bản song song cho System 1 (`.mcap` → `vln_n1`): [06b_pipeline_mcap_to_s1](06b_pipeline_mcap_to_s1.md).

*Quay lại mục lục: [00_README](00_README.md).*
