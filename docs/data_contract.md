# Data contract — LeRobot schema cho **InternVLA-N1**

Nguyên liệu để chốt schema với nhóm SIM (Ngày 5). Ghi lại **số liệu đo được**, không phải số liệu trong docs.

## 1. 🎯 Model đích: **InternVLA-N1** — nhưng nó là **dual-system**

**Kết luận quan trọng nhất của tài liệu này:** không tồn tại "**một** schema data của InternVLA-N1".
Mỗi system ăn một loại data khác nhau. Contract với nhóm SIM **bắt buộc** phải nói rõ đang sinh data
cho system nào — nếu không, lỗi chỉ lộ ra ở W4 lúc fine-tune.

Bằng chứng từ chính checkpoint (`SETUP_NOTES.md` 2.4): load `InternVLA-N1` bằng
`Qwen2_5_VLForConditionalGeneration` thì ~120 tensor bị báo `UNEXPECTED`, tất cả cùng tiền tố
`model.language_model.navdp.*` + `model.language_model.latent_queries`. **NavDP = System 1**,
diffusion policy — và `vln_n1/` chính là data huấn luyện nó.

| Nguồn | Trạng thái | Vai trò |
|---|---|---|
| **`vln_n1/`** | ✅ **ĐÃ ĐO 21/07** (mục 2), `info.json` khớp parquet | **Schema System 1** — quỹ đạo SE(3) liên tục |
| **`vln_ce/`** | ✅ **ĐÃ ĐO 22/07** (mục 4.b), `info.json` khớp parquet | **Schema System 2** — `pose`/`goal`/`action` rời rạc, **kèm sẵn RGB+depth** |
| `vln_pe/r2r_aliengo` | ✅ đã đo (mục 3), `info.json` **sai** (3.c.1) | Baseline CMA/RDP + vật chứng PR A |

**✅ Đã đủ để viết contract cho CẢ HAI system.** Mảnh cuối (`pose.{setting}`/`goal.{setting}`) đã tìm ra:
`vln_ce/traj_data/{r2r,rxr,scalevln}` chính là data `internvla_n1_lerobot_dataset.py` đọc — có đủ ba cột
`pose.{setting}` / `goal.{setting}` / `relative_goal_frame_id.{setting}`, với
**`{setting} = f'{height}cm_{pitch_2}deg'`** (5 cấu hình camera). Chi tiết đo được ở mục 4.b.

> Mục 3 (`vln_pe`) giữ lại làm **tham chiếu đối chiếu** và bằng chứng PR — không phải contract.

---

## 2. ✅ `vln_n1/` — schema data của **System 1** (đã đo 21/07)

### 2.1. Cách lấy (không lấy được bằng `--include "meta/*"`)

`fnmatch.filter(files, "vln_n1/**/meta/info.json")` → **0 file**. Toàn bộ `vln_n1/` là **3774 file
`.tar.gz`**, không có file rời. Phải tải archive rồi giải nén.

**Cây thư mục — mỗi scene render 2 lần, 2 camera khác nhau:**

```
vln_n1/traj_data/<simulator>_<camera>/<scene_uuid>.tar.gz
  simulator : 3dfront | gibson | hm3d | hssd | matterport3d | replica
  camera    : d435i | zed
```

Mẫu đã đo: `vln_n1/traj_data/matterport3d_d435i/pLe4wQe7qrG.tar.gz` (248.7 MB).

```python
from huggingface_hub import hf_hub_download
import tarfile, glob, json, pandas as pd

tgz = hf_hub_download("InternRobotics/InternData-N1",
                      filename="vln_n1/traj_data/matterport3d_d435i/pLe4wQe7qrG.tar.gz",
                      repo_type="dataset", local_dir="/kaggle/temp/n1dl")
with tarfile.open(tgz) as t:
    t.extractall("/kaggle/temp/n1x")     # KHONG giai nen vao /kaggle/working
```

### 2.2. `meta/info.json` — và nó **KHỚP** parquet

```json
{
  "codebase_version": "v2.1",  "robot_type": "unknown",
  "total_episodes": 13, "total_frames": 1811, "total_tasks": 26, "total_videos": 13,
  "total_chunks": 1, "chunks_size": 1000, "fps": 30,
  "splits": { "train": "0:1" },
  "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "observation.camera_intrinsic": { "dtype": "float32", "shape": [3, 3] },
    "observation.camera_extrinsic": { "dtype": "float32", "shape": [4, 4] },
    "action":                       { "dtype": "float32", "shape": [4, 4] }
  }
}
```

**Cột parquet thật** (`episode_000000`, 151 frame):

| Cột | dtype | Nội dung |
|---|---|---|
| `index` | int64 | 0, 1, 2, … |
| `observation.camera_intrinsic` | object → 9 float | ma trận 3×3 phẳng, **hằng số qua mọi frame** |
| `observation.camera_extrinsic` | object → 16 float | ma trận 4×4 phẳng, hằng số ở 3 frame đầu |
| `action` | object → 16 float | **ma trận 4×4 SE(3)**, thay đổi từng frame → quỹ đạo camera |

✅ **3 feature khai báo = 3 cột thật.** Đây là bằng chứng dứt điểm cho nguyên nhân lỗi ở 3.c.1:
`info.json` của `vln_pe` là **template của `vln_n1` bị copy mà không sửa `features`**.

### 2.3. 🎯 Intrinsic xác nhận đúng camera D435i — và cho luôn độ phân giải

```
fx = 355.81464   fy = 351.687
cx = 240.0       cy = 135.0
```

- `cx=240, cy=135` → ảnh **480 × 270**
- FOV ngang = 2·atan(240/355.81) = **68°**
- FOV dọc  = 2·atan(135/351.69) = **42°**

Spec Intel RealSense **D435i: 69° × 42°** → **khớp**. Vậy hậu tố `_d435i` / `_zed` là camera thật,
và nhóm SIM phải mô phỏng đúng model camera, không phải camera tuỳ ý.

⬜ **Phải đo thêm `_zed`** — intrinsic khác hẳn (ZED FOV rộng hơn, có baseline stereo). Nếu nhóm SIM
render bằng camera khác thì System 1 nhận intrinsic lệch → quỹ đạo dự đoán sai tỉ lệ.

### 2.4. Khác biệt so với `vln_pe` — đáng chú ý

| | `vln_n1` | `vln_pe` |
|---|---|---|
| `action` | **4×4 SE(3) liên tục** | int rời rạc {0,1,2,3,5} |
| `info.json` khớp parquet | ✅ có | ❌ không (3.c.1) |
| Video | `.mp4` | `.npy` |
| Cột LeRobot chuẩn (`timestamp`, `frame_index`, `episode_index`, `task_index`) | ❌ **KHÔNG CÓ**, chỉ có `index` | ✅ có đủ |
| Frame/episode | 1811/13 ≈ 139 | 418/23 ≈ 18 |

⚠️ **`vln_n1` thiếu `timestamp`, `frame_index`, `episode_index`, `task_index`** — đây là các trường
**bắt buộc** của LeRobotDataset v2.1. Nghĩa là `vln_n1` không phải LeRobot chuẩn dù khai
`codebase_version: v2.1`. Nhóm SIM cần biết: sinh đúng theo `vln_n1` thì loader LeRobot gốc không đọc
được, chỉ loader riêng của InternNav đọc được.

⚠️ **`fps: 30` và `splits: {"train": "0:1"}`** lặp lại y hệt `vln_pe` → hai trường này là **giá trị
template ở mọi subset**, không đáng tin ở đâu cả.

---

## 3. Tham chiếu — schema `vln_pe/r2r_aliengo` (đã đo, KHÔNG phải contract)

- **Nguồn:** `InternRobotics/InternData-N1` (HF dataset)
- **Đường dẫn:** `vln_pe/traj_data/r2r_aliengo/17DRP5sb8fy/`
- **Ngày đo:** 21/07/2026
- **Cách lấy:** xem `SETUP_NOTES.md` 3.11 (chỉ tải `meta/*` + 1 parquet, vài trăm KB)

> ⚠️ Đây là **một** trong 122 scene. Mỗi scene là một LeRobotDataset độc lập, có `meta/` riêng.
> Nhóm SIM cũng sẽ sinh nhiều dataset chứ không phải một — điểm phải thống nhất trong contract.
>
> Giữ lại vì: (a) đối chiếu với `vln_n1` để biết chỗ nào là quy ước chung của repo, chỗ nào riêng
> subset; (b) là bằng chứng cho PR fix docs (`SETUP_NOTES.md` 3.13).

---

### 3.a. `meta/info.json` — khai báo

```json
{
  "codebase_version": "v2.1",
  "robot_type": "unknown",
  "total_episodes": 23,
  "total_frames": 418,
  "total_tasks": 69,
  "total_videos": 23,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 30,
  "splits": { "train": "0:1" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.npy"
}
```

`features` khai báo 8 key:

| Feature | dtype | shape |
|---|---|---|
| `observation.camera_intrinsic` | float32 | (3, 3) |
| `observation.camera_extrinsic` | float32 | (4, 4) |
| `action` | float32 | (4, 4) |
| `timestamp` | float32 | (1,) |
| `frame_index` | int64 | (1,) |
| `episode_index` | int64 | (1,) |
| `index` | int64 | (1,) |
| `task_index` | int64 | (1,) |

---

### 3.b. `episode_000000.parquet` — dữ liệu thật

14 cột, 12 frame:

| Cột | dtype | Ghi chú |
|---|---|---|
| `observation.camera_position` | object | list float, 3 phần tử (xyz) |
| `observation.camera_orientation` | object | list float, **4** phần tử → quaternion (w,x,y,z?) |
| `observation.camera_yaw` | float64 | scalar, radian |
| `observation.robot_position` | object | list float, 3 phần tử (xyz) |
| `observation.robot_orientation` | object | list float, 4 phần tử → quaternion |
| `observation.robot_yaw` | float64 | scalar, radian |
| `observation.progress` | float64 | 0.0 ở 3 frame đầu |
| `observation.step` | int64 | 0, 50, 100 → bước sim, **không phải** frame index |
| `observation.action` | int64 | **rời rạc**: thấy giá trị 3, 3, 1 |
| `timestamp` | float32 | 0.0, 0.166667, 0.333333 |
| `frame_index` | int64 | 0, 1, 2 |
| `episode_index` | int64 | 0 |
| `index` | int64 | 0, 1, 2 |
| `task_index` | int64 | 0 |

---

### 3.c. 🚨 Ba mâu thuẫn giữa khai báo và dữ liệu thật

Phải nêu rõ với nhóm SIM. Đừng bê nguyên `info.json` sang. **Kiểm tra lại y hệt trên `vln_n1`** —
nếu lặp lại thì đây là lỗi hệ thống của cả dataset, PR sẽ mạnh hơn nhiều.

#### 3.c.1. `features` trong `info.json` KHÔNG khớp cột parquet

| `info.json` khai báo | Parquet thật |
|---|---|
| `observation.camera_intrinsic` (3,3) | ❌ không có |
| `observation.camera_extrinsic` (4,4) | ❌ không có |
| `action` float32 (4,4) | ❌ không có — có `observation.action` **int64 scalar** |
| — | ✅ `camera_position` / `camera_orientation` / `camera_yaw` |
| — | ✅ `robot_position` / `robot_orientation` / `robot_yaw` |
| — | ✅ `progress`, `step` |

**Hệ quả:** pose camera được lưu dạng **position + quaternion tách rời**, không phải ma trận extrinsic 4×4 như khai báo. Không có intrinsic ở đâu cả → **không dựng lại được phép chiếu 3D→2D** từ dữ liệu này nếu chỉ dựa vào parquet.

**Action space là rời rạc** (int64: 1, 3), **không phải** SE(3) 4×4 liên tục như `info.json` nói.

#### Nguyên nhân gốc — đã truy ra từ source code InternNav (21/07)

**Parquet đúng, `info.json` sai.** Không phải ta tải nhầm data. `features` trong `info.json` là
**template của NavDP bị copy sang `vln_pe` mà không sửa**. Bằng chứng trực tiếp —
`internnav/dataset/navdp_lerobot_dataset.py` đọc đúng ba trường đó, đúng từng shape:

```python
camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
camera_trajectory = np.array([np.stack(frame) for frame in df['action']], ...).reshape(-1, 4, 4)
```

Trong khi action rời rạc của parquet khớp với `internvla_n1_lerobot_dataset.py` / `vlln_lerobot_dataset.py`:

```python
self.idx2actions = {0: 'STOP', 1: "↑", 2: "←", 3: "→", 5: "↓"}
```

→ Giá trị `3, 3, 1` của ta = **quay phải, quay phải, tiến**. Dữ liệu hoàn toàn hợp lệ.

Và `cma_lerobot_dataset.py` đọc đúng bộ từ vựng của parquet này: `position`, `orientation`, `yaw`,
`progress`, `step`.

**Ba dấu hiệu khác cùng chỉ vào "template chưa điền":** `robot_type: "unknown"` (dữ liệu rõ ràng là
Aliengo), `splits: {"train": "0:1"}` (giá trị mặc định), `fps: 30` (xem 3.2).

→ **Việc cần làm:** đọc parquet của scene khác + của subset `vln_n1` để xác nhận phạm vi lỗi.

#### 3.c.2. `fps: 30` nhưng khoảng thời gian thật là 1/6 s

`timestamp` bước đều 0.166667 s → **6 Hz**, không phải 30 Hz. Cùng lúc `observation.step` nhảy 0 → 50 → 100.

Giải thích khả dĩ: sim chạy ở tần số cao, trajectory được **hạ mẫu** khi ghi, nhưng `fps` trong `info.json` vẫn giữ giá trị của sim. Nếu ai đó tính thời gian bằng `frame_index / fps` sẽ ra **sai 5 lần**.

→ **Contract phải ghi rõ:** dùng cột `timestamp`, không tự tính từ `fps`.

#### 3.c.3. `splits: {"train": "0:1"}` nhưng có 23 episode

Khai báo split chỉ chứa 1 episode trong khi `total_episodes: 23`. Trường `splits` ở đây **không dùng được**.

Ngoài ra `total_tasks: 69` > `total_episodes: 23` → khoảng 3 task/episode, cần xác nhận bằng `tasks.jsonl`.

---

## 4. 🚨 Repo có ÍT NHẤT 3 schema LeRobot khác nhau — đây là lý do phải đổi sang `vln_n1`

Phát hiện quan trọng nhất khi truy nguyên nhân 3.1. Đọc `internnav/dataset/*_lerobot_dataset.py`:

**Bảng đã sửa sau khi đo `vln_n1` (bản trước map sai `vln_n1` ↔ loader `internvla_n1`):**

| Loader | Cột chính | Action | Subset — đã xác minh? |
|---|---|---|---|
| `navdp_lerobot_dataset.py` | `observation.camera_intrinsic` (3,3), `observation.camera_extrinsic` (4,4) | **4×4 SE(3) liên tục** | ✅ **`vln_n1/`** — khớp từng shape (mục 2) |
| `cma_lerobot_dataset.py` | `position`, `orientation`, `yaw`, `progress`, `step` | int rời rạc | ✅ `vln_pe/` (mục 3) |
| `internvla_n1_lerobot_dataset.py`, `vlln_lerobot_dataset.py` | `pose.{setting}`, `goal.{setting}`, `relative_goal_frame_id.{setting}` | int rời rạc | ✅ **`vln_ce/`** — khớp từng shape (mục 4.b) |

**Vì sao `vln_n1` lại là schema NavDP — và vì sao điều đó hợp lý:**

InternVLA-N1 là **dual-system**. System 1 chính là **diffusion policy kiểu NavDP**, sinh quỹ đạo
liên tục. Vậy `vln_n1/` là data huấn luyện **System 1**: `action` 4×4 SE(3) là chuỗi pose camera,
không phải lệnh điều khiển rời rạc.

→ **Không có "một schema của InternVLA-N1".** Mỗi system ăn một loại data. Contract phải nói rõ
nhóm SIM đang sinh data cho system nào:

| Sinh data cho | Schema theo | Trạng thái |
|---|---|---|
| **System 1** (diffusion policy, quỹ đạo liên tục) | `vln_n1/` — mục 2 | ✅ đo xong, viết contract được ngay |
| **System 2** (VLM, action rời rạc + pixel goal) | `vln_ce/` — mục 4.b | ✅ **đo xong 22/07**, viết contract được ngay |
| Baseline CMA/RDP (eval Ngày 4) | `vln_pe/` — mục 3 | ✅ đo xong, nhưng là data **đọc vào**, ngoài phạm vi contract |

### 4.b. ✅ `vln_ce/` — schema data của **System 2** (đã đo 22/07)

**Đã tìm ra.** `vln_ce/traj_data/` chứa đúng ba cột `internvla_n1_lerobot_dataset.py` đọc. Chuỗi bằng
chứng khép kín (data thật + source code + train script):

1. **Loader ghép `{setting}` thế nào** — `internvla_n1_lerobot_dataset.py:850`:
   ```python
   setting = f'{height}cm_{pitch_2}deg'      # vd "125cm_30deg"
   ```
2. **Registry** cùng file (dòng ~80–145) khai `data_path` trần: `traj_data/r2r`, `traj_data/rxr`,
   `traj_data/scalevln` — khớp đúng 3 thư mục con của `vln_ce/traj_data/` (r2r 61 file, rxr 59,
   scalevln 794).
3. **`train_system2.sh`** dùng đúng bộ tên: `vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,
   r2r_60cm_30_30,rxr_...,scalevln_...`.

#### Cách lấy (file nhỏ nhất cả repo, 16.16 MB — rẻ hơn `vln_n1`)

`vln_ce/traj_data/` toàn `.tar.gz`, không có file rời. File nhỏ nhất: `vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz`.

```python
from huggingface_hub import hf_hub_download
import tarfile, glob, pandas as pd

tgz = hf_hub_download("InternRobotics/InternData-N1",
        filename="vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz",
        repo_type="dataset", local_dir="cedl")
tarfile.open(tgz).extractall("cex")
df = pd.read_parquet(sorted(glob.glob("cex/**/*.parquet", recursive=True))[0])
```

#### Cây thư mục

```
vln_ce/traj_data/<subset>/<scene>.tar.gz
  subset : r2r | rxr | scalevln
  scene/
    data/chunk-000/episode_000000.parquet          <- 21 cot
    videos/chunk-000/observation.images.{rgb,depth}.{setting}/episode_XXXXXX_<frame>.png
    meta/{info,episodes,tasks,episodes_stats}.jsonl
```

#### 5 cấu hình camera (`{setting}`)

`{setting} = f'{height}cm_{pitch_2}deg'`. Data thật có **5 setting** (không phải 4):

| setting | height | pitch | có trong `train_system2.sh`? |
|---|---|---|---|
| `125cm_0deg` | 125 | 0° | ❌ **không** (xem ghi chú dưới) |
| `125cm_30deg` | 125 | 30° | ✅ `r2r_125cm_0_30` |
| `125cm_45deg` | 125 | 45° | ✅ `r2r_125cm_0_45` |
| `60cm_15deg` | 60 | 15° | ✅ `r2r_60cm_15_15` |
| `60cm_30deg` | 60 | 30° | ✅ `r2r_60cm_30_30` |

🚨 **`125cm_0deg` (camera nhìn thẳng) có trong data nhưng bị loại khỏi train** — vì camera không cúi
thì hầu như không thấy điểm sàn để chấm pixel goal: đo được **goal toàn `(-1,-1)`** ở setting này. Hệ
quả cho robot VinRobotics: **camera phải có góc cúi (pitch ≥ ~15°)**, không thì pixel-goal của S2 vô dụng.

#### Cột parquet — 21 cột (đo trên `rxr/YmJkqBEsHnH`, 3 episode × 16 frame)

| Cột | dtype | shape | Nội dung |
|---|---|---|---|
| `action` | int32 | (1) | **rời rạc** `{1:↑, 2:←, 3:→, 5:↓}`, `-1` = frame start. Đo: `[-1,2,2,2,2,2,2,2,2,2,1,1,2,1,2,1]` |
| `pose.{setting}` | float32 | (4,4) | pose camera từng frame · **×5 setting** |
| `goal.{setting}` | int32 | (2) | pixel goal `[u,v]`; `(-1,-1)` = không có · **×5 setting**. Đo: `[230,372] [280,356] [275,392]` |
| `relative_goal_frame_id.{setting}` | int32 | (1) | số frame còn lại tới đích; `-1` = không có · **×5 setting** |
| `timestamp` | float32 | (1) | ✅ có |
| `frame_index` | int64 | (1) | ✅ có |
| `episode_index` | int64 | (1) | ✅ có |
| `index` | int64 | (1) | ✅ có |
| `task_index` | int64 | (1) | ✅ có |

→ 1 (`action`) + 5×3 (pose/goal/rel) + 5 (chuẩn LeRobot) = **21 cột**.

✅ **`vln_ce` CÓ ĐỦ 4 trường LeRobot chuẩn** (`timestamp/frame_index/episode_index/task_index`) —
**ngược với `vln_n1`** (mục 2.4 — thiếu hết). Vậy hai subset không cùng mức chuẩn LeRobot.

#### RGB + depth — CÓ SẴN trong `vln_ce` (khác `vln_n1`/`vln_pe`)

Điểm đọc code không ra. Ảnh lưu ở `videos/`, dạng **PNG từng frame** (không mp4, không npy):

| Loại | Kích thước | dtype / mode | Đơn vị |
|---|---|---|---|
| RGB | **640×480** | uint8, RGB | 0–255 |
| Depth | **640×480** | uint16, mode `I;16` | **milimet**, clip ở 10000 (=10 m) |

10 thư mục = 5 setting × {rgb, depth}. ⚠️ RGB ở đây **640×480**, khác `vln_n1` (480×270 của D435i)
→ hai subset dùng model camera khác nhau, nhóm SIM phải render đúng cho từng system.

#### `meta/` — đo được

- **`info.json` KHỚP parquet** (features khai đúng: `action` int32 [1], `pose` (4,4), `goal` [2],
  `relative_goal_frame_id` [1]). → Lỗi 3.c.1 **chỉ cục bộ ở `vln_pe`**, không phải lỗi toàn dataset.
  Điều này thu hẹp phạm vi PR fix docs.
- **`episodes.jsonl`**: `{"episode_index", "tasks": [...], "length"}`. Loader lấy
  `tasks[0].split("<INSTRUCTION_SEP>")` → **nhiều instruction/episode**, giải thích luôn thắc mắc
  3.c.3 (`total_tasks > total_episodes`). Có `assert len(actions) == length` → `length` **phải** =
  số dòng parquet.
- **`tasks.jsonl`**: câu lệnh tiếng Anh dạng tự nhiên, vd *"You are facing toward the entrance of the
  church, turn around and stop in between the steps..."*

#### Lỗi nhỏ mới (ứng viên PR, không chặn)

- `info.json.video_path` ghi `.mp4` nhưng file thật là `.png`; `total_videos: 0`. Nhẹ.
- ⬜ Trên scene `rxr` nhỏ này, `relative_goal_frame_id` **toàn `-1`** dù `goal` có giá trị ở vài frame
  của setting 45°/60cm — trong khi loader route pixel-goal theo `relative_goal_frame_id[0]==-1`. Cần
  kiểm lại trên 1 scene `r2r` lớn hơn trước khi kết luận (có thể do scene rxr edge-case).

### 4.c. Nguồn schema dự phòng — đọc thẳng từ code loader

Nếu 4.b không tìm ra data, contract System 2 **vẫn viết được** từ
`internnav/dataset/internvla_n1_lerobot_dataset.py`. Đây là nguồn không phụ thuộc download, không
phụ thuộc quyền HF, và **chắc hơn `info.json`** — vì `info.json` đã được chứng minh là sai (3.c.1),
còn code thì phải đúng, nếu không model không train được.

Các cột loader đọc:

| Cột | Cách dùng trong code |
|---|---|
| `action` | `ep_actions = df["action"].tolist()`; map qua `{0:'STOP', 1:'↑', 2:'←', 3:'→', 5:'↓'}` |
| `pose.{setting}` | `df[pose_key].apply(lambda x: x.tolist()).tolist()` |
| `goal.{setting}` | kiểm tra `goal_key in df.columns` → **có thể optional** |
| `relative_goal_frame_id.{setting}` | kiểm tra `in df.columns` → **có thể optional** |

✅ **`{setting}` đã xác nhận (22/07):** `internvla_n1_lerobot_dataset.py:850` ghép
**`setting = f'{height}cm_{pitch_2}deg'`** → 5 giá trị thật: `125cm_0deg`, `125cm_30deg`,
`125cm_45deg`, `60cm_15deg`, `60cm_30deg` (xem 4.b). Lưu ý `pitch_1` **không** vào tên `{setting}` —
nó được truyền riêng (nghi là camera thứ hai), chưa xác minh.

---

## 5. Các điểm khác phải nêu khi chốt schema

*(quan sát từ `vln_pe` — phải kiểm tra lại từng điểm trên `vln_n1`)*

- **Không có RGB / depth trong parquet.** `features` cũng không khai báo image. Ảnh nằm ở `videos/` dưới dạng **`.npy`** (xem `video_path`), **không phải** mp4 như LeRobot chuẩn. Chưa tải, chưa biết shape/dtype → **việc cần làm tiếp theo**.
- **`robot_type: "unknown"`** — nhưng thư mục tên `r2r_aliengo` → render bằng robot **Aliengo** (chó 4 chân, Isaac Sim), **không phải** camera agent VLN-CE. Chiều cao camera, hệ toạ độ, action space có thể lệch với thứ nhóm SIM sinh từ Habitat.
- **`camera` và `robot` có pose riêng biệt** → có offset camera-so-với-thân robot. `camera_yaw` và `robot_yaw` ở 3 frame đầu **bằng nhau** (-0.040159, -0.259446, -0.503138) nhưng position khác nhau → camera gắn cứng, chỉ lệch tịnh tiến. Cần xác nhận trên nhiều frame hơn.
- **Quaternion convention chưa rõ** (wxyz hay xyzw; frame Z-up hay Y-up). Giá trị đầu ~0.9997 với `yaw` ~-0.04 rad gợi ý **wxyz** (w đứng đầu). Phải chốt dứt khoát với nhóm SIM — đây là loại lệch gây bug im lặng.
- **`codebase_version: "v2.1"`** — phiên bản LeRobot format. Nhóm SIM dùng bản nào? Nếu lệch major thì loader không đọc được.

---

## 6. TODO trước Ngày 5

**Chặn (phải xong mới viết được contract đầy đủ):**
- [x] Đo schema **`vln_n1/`** → xong (mục 2): schema **System 1**, `info.json` khớp parquet
- [x] Xác định vai trò từng subset → xong (mục 1 & 4)
- [x] 🚨 **Mở 1 tar.gz của `vln_ce/traj_data/`** → xong (mục 4.b): schema **System 2**, có đủ
      `pose.{setting}`/`goal.{setting}`/`relative_goal_frame_id.{setting}`, kèm RGB+depth. **Cả hai
      system giờ đều viết được contract.**

**Sau đó:**
- [x] Tìm `{setting}` là gì → xong: `f'{height}cm_{pitch_2}deg'` (mục 4.b/4.c)
- [x] Đọc `meta/tasks.jsonl` của `vln_ce` → format instruction (câu lệnh tiếng Anh tự nhiên, mục 4.b)
- [x] Shape/dtype ảnh RGB & depth → xong với `vln_ce`: RGB 640×480 uint8, depth 640×480 uint16 mm (mục 4.b)
- [ ] Đọc `meta/episodes_stats.jsonl` của `vln_n1` **và** `vln_ce` → min/max từng cột, biết range hợp lệ
- [ ] Đo `meta/tasks.jsonl` + video của `vln_n1` (System 1) để đối chiếu — hiện mới đo kỹ `vln_ce`
- [ ] Hỏi nhóm SIM: convention quaternion (wxyz/xyzw, Z-up/Y-up); LeRobot version; số dataset sẽ sinh;
      **sinh cho System 1 (`vln_n1`) hay System 2 (`vln_ce`)** — mỗi system một schema khác nhau

**Đối chiếu (giá trị cho PR, không chặn contract):**
- [ ] Kiểm tra `vln_n1` có lặp lại lỗi 3.c.1 (`features` lệch parquet) không → nếu có thì là lỗi hệ thống
- [ ] Đọc parquet của scene khác trong `vln_pe` → xác nhận 3.c.1 không phải cá biệt
