# Ba cấu trúc dữ liệu của InternNav: `vln_n1` · `vln_pe` · `vln_ce`

> Tài liệu tổng hợp — **train system nào theo cấu trúc nào**, bằng chứng từ source code repo
> `InternRobotics/InternNav`, và kiến trúc chi tiết của từng cấu trúc.
> Mọi số liệu là **đo thật** trên `InternRobotics/InternData-N1` (21–22/07/2026), không lấy từ docs.
> Xem thêm: [`data_contract.md`](data_contract.md) (chi tiết schema) · `SETUP_NOTES.md` (nhật ký đo).

---

## 0. TL;DR

- `InternData-N1` ở gốc có **đúng 3 thư mục**: `vln_ce/`, `vln_n1/`, `vln_pe/` (tổng 20 829 file).
- Ba hậu tố **không phải 3 định dạng của cùng một thứ** — chúng là **3 trục khác nhau**: benchmark
  (`ce`), thân xác robot (`pe`), model đích (`n1`).
- InternVLA-N1 là **dual-system** → **không có một schema chung**. Mỗi thành phần ăn một cấu trúc khác:
  - **System 1** (NavDP, diffusion, quỹ đạo liên tục) → cấu trúc **`vln_n1`**
  - **System 2** (Qwen2.5-VL 7B, chấm pixel goal) → cấu trúc **`vln_ce`**
  - **Baseline CMA/RDP** (đối chứng, dự phòng eval) → cấu trúc **`vln_pe`**

---

## 1. Bảng tổng hợp: train system nào → theo cấu trúc nào

| Train cho | Cấu trúc data | Loader (repo) | Train script (repo) | Action | Đã đo? |
|---|---|---|---|---|---|
| **System 1** — NavDP diffusion policy | **`vln_n1/`** | `internnav/dataset/navdp_lerobot_dataset.py` | `scripts/train/base_train/configs/navdp.py` | **4×4 SE(3) liên tục** | ✅ |
| **System 2** — VLM (Qwen2.5-VL 7B) | **`vln_ce/`** | `internnav/dataset/internvla_n1_lerobot_dataset.py` | `scripts/train/qwenvl_train/train_system2.sh` | int rời rạc `{1,2,3,5}` + **pixel goal** | ✅ |
| Baseline **CMA** | **`vln_pe/`** | `internnav/dataset/cma_lerobot_dataset.py` | `scripts/train/base_train/configs/cma.py` | int rời rạc (4 lớp) | ✅ |
| Baseline **RDP** | **`vln_pe/`** | `internnav/dataset/rdp_lerobot_dataset.py` | `scripts/train/base_train/configs/rdp.py` | liên tục `(dx,dy,dyaw)` | ✅ (chung `vln_pe`) |

> ⚠️ **Baseline CMA/RDP KHÔNG nằm trong dual-system.** Chúng là các agent end-to-end độc lập, chọn
> thay thế `internvla_n1_agent` lúc eval. Vai trò: đối chứng SR/SPL và **lưới an toàn cho eval Ngày 4**
> (chạy được mà không cần checkpoint 16.79GB / không cần S2).

### Ý nghĩa ba hậu tố

| Hậu tố | Viết tắt | Trục thay đổi | Sinh ra ở đâu |
|---|---|---|---|
| `vln_ce` | **C**ontinuous **E**nvironment | **benchmark** — R2R/RxR gốc, so số với paper | Habitat, agent là camera bay tự do |
| `vln_pe` | **P**hysical **E**mbodiment | **thân xác robot** — cùng task, đặt lên robot có vật lý | Isaac Sim, robot cụ thể (Aliengo…) |
| `vln_n1` | **InternVLA-N1** | **model** — data train chính System 1 | 6 simulator × 2 camera, render hàng loạt |

---

## 2. Tại sao lại nhận định như vậy — bằng chứng từ code

Mỗi mapping ở Mục 1 được chứng minh bằng **loader đọc đúng cột nào** + **train script trỏ vào subset nào**.

### 2.1. System 1 → `vln_n1` (loader `navdp_lerobot_dataset.py`)

**Bằng chứng A — loader đọc đúng 3 trường đặc trưng của `vln_n1`:**

📄 `internnav/dataset/navdp_lerobot_dataset.py:198-201`
```python
camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
trajectory_length = len(df['action'].tolist())
camera_trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64).reshape(-1, 4, 4)
```
→ Ba cột `camera_intrinsic` (3,3), `camera_extrinsic` (4,4), `action` (4,4 SE(3)) — **chỉ `vln_n1` có
đúng cả ba** (đo ở Mục 3.1). `vln_pe` khai báo có nhưng parquet thật không có (Mục 3.2).

**Bằng chứng B — checkpoint chứa NavDP = System 1.** Load `InternVLA-N1` bằng class HF thuần
(`Qwen2_5_VLForConditionalGeneration`) thì ~120 tensor báo `UNEXPECTED`, tất cả cùng tiền tố
`model.language_model.navdp.*` (chi tiết `SETUP_NOTES.md` 2.4). **NavDP = System 1**, và `vln_n1/` là
data huấn luyện nó.

### 2.2. System 2 → `vln_ce` (loader `internvla_n1_lerobot_dataset.py`)

Chuỗi 3 mắt xích, khép kín:

**Mắt xích 1 — loader ghép tên cột `{setting}`:**

📄 `internnav/dataset/internvla_n1_lerobot_dataset.py:850`
```python
setting = f'{height}cm_{pitch_2}deg'          # vd: "125cm_30deg"
```
📄 `internnav/dataset/internvla_n1_lerobot_dataset.py:781-783`
```python
pose_key = f"pose.{setting}"
goal_key = f"goal.{setting}"
relative_goal_frame_id_key = f"relative_goal_frame_id.{setting}"
```

**Mắt xích 2 — registry trong cùng file trỏ `data_path` vào 3 thư mục con của `vln_ce`:**

📄 `internnav/dataset/internvla_n1_lerobot_dataset.py:50-51` (và các block kế tiếp tới ~dòng 121)
```python
R2R_125CM_0_30 = {
    "data_path": "traj_data/r2r",     # cũng có traj_data/rxr, traj_data/scalevln
    "height": 125, "pitch_1": 0, "pitch_2": 30,
}
```
→ `traj_data/{r2r, rxr, scalevln}` khớp **đúng 3 thư mục con** của `vln_ce/traj_data/`
(r2r 61 file · rxr 59 · scalevln 794).

**Mắt xích 3 — train script dùng đúng bộ tên đó:**

📄 `scripts/train/qwenvl_train/train_system2.sh`
```bash
llm=Qwen/Qwen2.5-VL-7B-Instruct
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,\
             rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30
```

**Xác nhận bằng data thật:** mở `vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz` (16.16 MB) → parquet có
đúng `pose.125cm_30deg`, `goal.125cm_30deg`, `relative_goal_frame_id.125cm_30deg`… (Mục 3.3).

**Action space là rời rạc** (không phải SE(3) như S1):

📄 `internnav/dataset/internvla_n1_lerobot_dataset.py:950`
```python
self.idx2actions = {0: 'STOP', 1: "↑", 2: "←", 3: "→", 5: "↓"}
```

### 2.3. Baseline CMA/RDP → `vln_pe` (loader `cma_lerobot_dataset.py`)

📄 `internnav/dataset/cma_lerobot_dataset.py:112-115`
```python
data['robot_info']['yaw']         = data['robot_info']['yaw'][:-drop_last_frame_nums]
data['robot_info']['position']    = data['robot_info']['position'][:-drop_last_frame_nums]
data['robot_info']['orientation'] = data['robot_info']['orientation'][:-drop_last_frame_nums]
data['progress']                  = data['progress'][:-drop_last_frame_nums]
```
→ Bộ từ vựng `position` / `orientation` / `yaw` / `progress` — **đúng các cột `vln_pe` có** và
`vln_n1`/`vln_ce` **không** có (đo ở Mục 3.2).

### 2.4. Kết luận: repo có ÍT NHẤT 3 schema LeRobot song song

| Loader | Cột chính | Action | Subset khớp |
|---|---|---|---|
| `navdp_lerobot_dataset.py` | `camera_intrinsic` (3,3), `camera_extrinsic` (4,4) | 4×4 SE(3) liên tục | **`vln_n1`** |
| `internvla_n1_lerobot_dataset.py` | `pose.{s}`, `goal.{s}`, `relative_goal_frame_id.{s}` | int rời rạc + pixel goal | **`vln_ce`** |
| `cma_lerobot_dataset.py` | `position`, `orientation`, `yaw`, `progress`, `step` | int rời rạc | **`vln_pe`** |

Ba subset **không thay thế nhau**: đưa `vln_pe` cho `navdp_lerobot_dataset.py` → nó tìm
`camera_intrinsic` không thấy → crash. Đây cũng là **nguyên nhân gốc** của lỗi `info.json` ở `vln_pe`
(features là template `vln_n1` bị copy nhầm — `data_contract.md` 3.c.1).

---

## 3. Kiến trúc từng cấu trúc (schema đo thật)

Cả ba đều theo khung **LeRobotDataset v2.1**: mỗi scene là một dataset độc lập.

```
<scene>/
├── data/chunk-000/episode_XXXXXX.parquet     ← bảng số (cột khác nhau tuỳ subset)
├── videos/chunk-000/<video_key>/...          ← RGB/depth (định dạng khác nhau)
└── meta/{info,episodes,tasks,episodes_stats}.jsonl
```

Điểm khác nhau nằm ở **cột parquet**, **định dạng ảnh**, và **mức tuân thủ LeRobot chuẩn**.

### 3.1. `vln_n1` — cấu trúc **System 1** (NavDP)

**Cây thư mục — mỗi scene render 2 lần, 2 camera:**
```
vln_n1/traj_data/<simulator>_<camera>/<scene_uuid>.tar.gz
  simulator : 3dfront | gibson | hm3d | hssd | matterport3d | replica
  camera    : d435i | zed
```
Mẫu đo: `vln_n1/traj_data/matterport3d_d435i/pLe4wQe7qrG.tar.gz` (248.7 MB), 13 episode / 1811 frame.

**Cột parquet — đúng 4 cột:**

| Cột | dtype | shape | Nội dung |
|---|---|---|---|
| `index` | int64 | scalar | 0,1,2,… |
| `observation.camera_intrinsic` | float32 | (3,3) | **hằng số qua mọi frame** |
| `observation.camera_extrinsic` | float32 | (4,4) | pose camera |
| `action` | float32 | **(4,4) SE(3)** | đổi mỗi frame → quỹ đạo camera |

**Intrinsic đo được** (`_d435i`): `fx=355.81, fy=351.69, cx=240, cy=135` → ảnh **480×270**, FOV
**68°×42°** → khớp spec Intel RealSense D435i. Hậu tố `_d435i`/`_zed` là **camera thật**.

**Đặc điểm cấu trúc:**
- `action` là **ma trận thuần hình học** (SE(3)) — không có nhãn ngôn ngữ trong loader.
- ⚠️ **THIẾU** `timestamp`, `frame_index`, `episode_index`, `task_index` — bốn trường **bắt buộc** của
  LeRobot v2.1. Chỉ loader riêng của InternNav đọc được, LeRobot gốc **không**.
- Video: `.mp4`.
- `info.json` **khớp** parquet (khai đúng 3 feature).

### 3.2. `vln_pe` — cấu trúc **Baseline CMA/RDP** (Physical Embodiment)

Mẫu đo: `vln_pe/traj_data/r2r_aliengo/17DRP5sb8fy/`, 23 episode / 418 frame. Thư mục tên `r2r_aliengo`
= episode R2R render trên **chó robot Aliengo** (Isaac Sim).

**Cột parquet — 14 cột (khác hẳn khai báo):**

| Cột | dtype | Nội dung |
|---|---|---|
| `observation.camera_position` | object | xyz (3) |
| `observation.camera_orientation` | object | quaternion (4) |
| `observation.camera_yaw` | float64 | radian |
| `observation.robot_position` | object | xyz (3) — **tách khỏi camera** |
| `observation.robot_orientation` | object | quaternion (4) |
| `observation.robot_yaw` | float64 | radian |
| `observation.progress` | float64 | tiến độ |
| `observation.step` | int64 | bước sim (0,50,100…), **không** phải frame index |
| `observation.action` | int64 | **rời rạc** (thấy 3,3,1) |
| `timestamp`,`frame_index`,`episode_index`,`index`,`task_index` | | ✅ có đủ (LeRobot chuẩn) |

**Đặc điểm cấu trúc — 3 mâu thuẫn khai-báo vs thật (ứng viên PR fix docs):**
- 🚨 `info.json.features` khai `camera_intrinsic/extrinsic/action (4,4)` — **parquet không có cột nào**.
  Đây là **template `vln_n1` bị copy nhầm**. Pose lưu dạng **position + quaternion tách rời**, không
  có intrinsic → không dựng lại phép chiếu 3D→2D chỉ từ parquet.
- 🚨 `fps: 30` nhưng `timestamp` bước đều 1/6 s → thật ra **6 Hz**. Tính thời gian bằng `frame/fps` sai 5×.
- 🚨 `splits: {"train":"0:1"}` nhưng có 23 episode → trường `splits` vô dụng.
- **`camera` và `robot` có pose riêng** → có offset camera-so-với-thân (đặc trưng "physical embodiment").
- Video: `.npy` (không mp4).

### 3.3. `vln_ce` — cấu trúc **System 2** (VLM)

**Cây thư mục:**
```
vln_ce/traj_data/<subset>/<scene>.tar.gz
  subset : r2r | rxr | scalevln
```
Mẫu đo: `vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz` (**16.16 MB — nhỏ nhất repo**), 3 episode / 16 frame.

**Cột parquet — 21 cột:**

| Cột | dtype | shape | Nội dung |
|---|---|---|---|
| `action` | int32 | (1) | **rời rạc** `{1:↑,2:←,3:→,5:↓}`, `-1`=start. Đo: `[-1,2,2,2,2,2,2,2,2,2,1,1,2,1,2,1]` |
| `pose.{setting}` | float32 | (4,4) | pose camera · **×5 setting** |
| `goal.{setting}` | int32 | (2) | **pixel goal `[u,v]`**; `(-1,-1)`=không có · **×5 setting**. Đo: `[230,372]`,`[280,356]` |
| `relative_goal_frame_id.{setting}` | int32 | (1) | số frame còn lại tới đích; `-1`=không · **×5** |
| `timestamp`,`frame_index`,`episode_index`,`index`,`task_index` | | ✅ **CÓ ĐỦ** (khác `vln_n1`) |

→ `1 + 5×3 + 5 = 21` cột.

**5 cấu hình camera** (`{setting} = f'{height}cm_{pitch_2}deg'`):

| setting | height | pitch | trong `train_system2.sh`? |
|---|---|---|---|
| `125cm_0deg` | 125 | 0° | ❌ **loại** — goal toàn `(-1,-1)` |
| `125cm_30deg` | 125 | 30° | ✅ |
| `125cm_45deg` | 125 | 45° | ✅ |
| `60cm_15deg` | 60 | 15° | ✅ |
| `60cm_30deg` | 60 | 30° | ✅ |

**RGB + depth — CÓ SẴN trong `vln_ce`** (khác `vln_n1`/`vln_pe`), lưu **PNG từng frame** tại `videos/`:

| Loại | Kích thước | dtype | Đơn vị |
|---|---|---|---|
| RGB | **640×480** | uint8 | 0–255 |
| Depth | **640×480** | uint16 (`I;16`) | **milimet**, clip 10000 (=10 m) |

**Đặc điểm cấu trúc:**
- Cùng một quỹ đạo được render lại ở **5 độ cao/góc cúi camera** → cơ chế chống lệ thuộc vị trí lắp camera.
- **`info.json` khớp parquet** → lỗi 3.c.1 chỉ cục bộ ở `vln_pe`, không phải lỗi toàn dataset.
- `episodes.jsonl`: `tasks[0].split("<INSTRUCTION_SEP>")` → **nhiều instruction/episode** (giải thích
  `total_tasks > total_episodes`). `assert len(actions) == length` → `length` phải = số dòng parquet.
- Lỗi nhỏ (ứng viên PR): `info.json.video_path` ghi `.mp4` nhưng file thật là `.png`; `total_videos: 0`.

### 3.4. Bảng so sánh nhanh 3 cấu trúc

| | `vln_n1` (S1) | `vln_pe` (baseline) | `vln_ce` (S2) |
|---|---|---|---|
| Action | 4×4 SE(3) liên tục | int rời rạc | int rời rạc + pixel goal |
| Pose lưu dạng | `camera_extrinsic` (4,4) | position + quaternion | `pose.{setting}` (4,4) ×5 |
| Pixel goal | ❌ | ❌ | ✅ `goal.{setting}` |
| RGB/Depth | mp4 (480×270) | npy | **PNG (640×480), có sẵn** |
| 4 trường LeRobot chuẩn | ❌ thiếu | ✅ | ✅ |
| `info.json` khớp parquet | ✅ | ❌ (copy nhầm) | ✅ |
| Cần ngôn ngữ | ❌ | có instruction | ✅ instruction |
| Camera | D435i / ZED | Aliengo (offset thân) | 5 setting height×pitch |

---

## 4. Hệ quả cho nhóm SIM / robot VinRobotics

1. **Chốt với nhóm SIM: sinh data cho System 1 (`vln_n1`) hay System 2 (`vln_ce`)?** — mỗi system
   một schema khác nhau. Trả lời sai câu này thì tới W4 fine-tune mới lộ.
2. **Camera phải khớp:**
   - Train S1 → intrinsic phải là **camera thật của robot**, không copy `fx=355.8` của D435i.
   - Train S2 → camera phải **có góc cúi (pitch ≥ ~15°)**; setting `0deg` bị loại vì pixel goal toàn `-1`.
3. **Khuyến nghị hướng train:** ưu tiên **System 1 theo `vln_n1`** — data rẻ (chỉ lái + ghi, không cần
   annotate), phần cứng vừa server ≥24GB. **System 2** cần full fine-tune Qwen 7B (`train_system2.sh`
   khai `-N 8 --gres=gpu:8` = 64 GPU) → ngoài tầm thực tập; thay bằng chỉnh prompt/few-shot.
4. **Baseline `vln_pe`** chỉ để đối chứng + dự phòng eval Ngày 4 — không phải hướng train.

---

## Phụ lục — vị trí file trong repo `InternRobotics/InternNav`

| Vai trò | Đường dẫn |
|---|---|
| Loader S1 | `internnav/dataset/navdp_lerobot_dataset.py` |
| Loader S2 | `internnav/dataset/internvla_n1_lerobot_dataset.py` |
| Loader baseline CMA | `internnav/dataset/cma_lerobot_dataset.py` |
| Loader baseline RDP | `internnav/dataset/rdp_lerobot_dataset.py` |
| Train S2 | `scripts/train/qwenvl_train/train_system2.sh` |
| Train S1 / baseline | `scripts/train/base_train/configs/{navdp,cma,rdp}.py` |
| Converter (sinh LeRobot) | `scripts/dataset_converters/vlnce2lerobot.py` |
| Agent dual-system | `internnav/agent/internvla_n1_agent.py` |
| **Load checkpoint S1+S2** (điểm duy nhất, ~dòng 33–40) | `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` (`InternVLAN1Net.__init__`) |
| Dựng S1 (`self.navdp`) + `latent_queries` — rẽ theo `config.system1` | `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` (`InternVLAN1MetaModel.__init__`, `build_navdp`) |
| Loader NavDP standalone (`navdp_pretrained`) | `internnav/model/basemodel/internvla_n1/navdp.py` (`NavDP_Policy_DPT_CriticSum_DAT.load_model`) |

*Số dòng trích dẫn theo nhánh `main` tại thời điểm đo 22/07/2026 — có thể lệch nếu repo cập nhật.*
