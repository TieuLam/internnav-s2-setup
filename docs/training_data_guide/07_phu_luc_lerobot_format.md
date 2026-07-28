# 07 — Phụ lục: bên trong một dataset LeRobot — `chunk`, `parquet`, và bốn file `meta/`

> **File này để làm gì:** giải thích **cấu trúc vật lý** của một dataset (chuẩn LeRobotDataset v2.1):
> vì sao có thư mục `chunk-XXX`, trong `.parquet` có những cột gì, và **từng trường trong bốn file
> `meta/` nghĩa là gì, sinh ra thế nào**.
>
> Mọi con số ví dụ **đo thật** trên scene `vln_ce/traj_data/r2r/17DRP5sb8fy` (75 episode) có sẵn
> trong `InternNav/data/`.
> Bộ tài liệu: [04_data_train_s2](04_data_train_s2.md) · [05_data_train_s1](05_data_train_s1.md) ·
> [00_README](00_README.md)

---

## 0. Bức tranh một scene

```
17DRP5sb8fy/                                  ← một scene (một căn nhà)
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet            ← bảng số của episode 0 (46 hàng = 46 frame)
│       ├── episode_000001.parquet            ← episode 1 (54 frame)
│       └── … (tới episode_000074)
├── videos/chunk-000/observation.images.{rgb,depth}.{setting}/…   ← ảnh
└── meta/
    ├── info.json              ← bản kê khai toàn dataset
    ├── tasks.jsonl            ← bảng tra: mã số → câu lệnh
    ├── episodes.jsonl         ← danh mục episode (câu lệnh + độ dài)
    └── episodes_stats.jsonl   ← thống kê min/max/mean/std từng cột
```

Ba khối: **`data/`** (số), **`videos/`** (ảnh), **`meta/`** (mô tả).

---

## 0.1. Cây thư mục từ `traj_data` xuống tới scene (4 tầng)

```
vln_ce/traj_data/
├── r2r/                          ← TẦNG 1: subset (nguồn dữ liệu)
│   ├── 17DRP5sb8fy/              ← TẦNG 2: scene (một căn nhà, tên = UUID scene 3D)
│   │   ├── data/ meta/ videos/   ← TẦNG 3: 3 khối trong scene
│   │   │   └── (data→chunk, videos→10 folder)   ← TẦNG 4
│   │   └── 17DRP5sb8fy.tar.gz    ← bản nén để tải lẻ từng scene
│   └── <scene khác>/ …
├── rxr/  …                       (các subset khác)
└── scalevln/ …
```

| Tầng | Chia theo | Vì sao |
|---|---|---|
| **1. subset** (`r2r`, `rxr`, `scalevln`) | ba **benchmark điều hướng** gốc | mỗi bộ có phong cách câu lệnh riêng: r2r ngắn từng bước; rxr dài chi tiết; scalevln sinh tự động quy mô lớn (~794 scene) |
| **2. scene** (`17DRP5sb8fy`…) | từng **căn nhà**; tên = UUID scene quét 3D (Matterport3D) | mỗi scene là **một LeRobotDataset độc lập** — `meta/` riêng, episode đánh số lại từ 0; không trộn chung được (index đụng nhau) |
| **3. `data/ meta/ videos/`** | loại nội dung | tách **số / mô tả / ảnh** ra ba nơi |
| **4a. `data/` → `chunk-XXX/`** | nhóm ≤ 1000 episode | chống quá tải thư mục (mục 1) |
| **4b. `videos/chunk-000/` → 10 folder** | 2 loại ảnh × 5 setting camera | mỗi "luồng quan sát" để riêng (mục 2b) |

---

## 1. Vì sao có `chunk-XXX`? — "ngăn kéo đựng hồ sơ"

`chunk` là **thư mục nhóm**, không mang ý nghĩa nội dung. Hình dung tủ hồ sơ: mỗi **ngăn kéo** (chunk)
đựng tối đa **1000 hồ sơ** (episode); mỗi hồ sơ vẫn là **một tờ riêng** (một file `.parquet`).

Quy tắc (`info.json` ghi `chunks_size = 1000`):

```
số_ngăn = episode_index // 1000
```

| `episode_index` | nằm ở |
|---|---|
| 0 → 999 | `chunk-000` |
| 1000 → 1999 | `chunk-001` |
| 2000 → 2999 | `chunk-002` |

Scene này chỉ có 75 episode → tất cả trong `chunk-000`, `info.json` ghi `total_chunks: 1`.

**Vì sao phải chia?** Nhét cả trăm nghìn file vào **một** thư mục thì hệ điều hành liệt kê/sao chép
rất chậm. **Đây thuần là mẹo kỹ thuật** — "mỗi chunk chứa nhiều episode" nghĩa là *nhiều file nằm
chung một thư mục*, chứ **không phải** nhiều episode bị trộn vào một file.

> Reader không dò thư mục, nó **tính** đường dẫn: `chunk = ep // chunks_size`.

---

## 2. File `episode_XXXXXX.parquet` — bảng số của một episode

Mỗi hàng = một frame. Episode 0 có **46 hàng × 21 cột**:

```
action,
pose.125cm_0deg,  goal.125cm_0deg,  relative_goal_frame_id.125cm_0deg,
pose.125cm_30deg, goal.125cm_30deg, relative_goal_frame_id.125cm_30deg,
pose.125cm_45deg, goal.125cm_45deg, relative_goal_frame_id.125cm_45deg,
pose.60cm_15deg,  goal.60cm_15deg,  relative_goal_frame_id.60cm_15deg,
pose.60cm_30deg,  goal.60cm_30deg,  relative_goal_frame_id.60cm_30deg,
timestamp, frame_index, episode_index, index, task_index
```

**1 cột `action` + 5 setting × 3 cột + 5 cột quản lý = 21 cột.**

### 2.1. `action` — nút bấm mỗi frame
`int32`. Giá trị `1=↑`, `2=←`, `3=→`, `5=↓`, `0=STOP`, **`-1` = frame khởi đầu**. Đo thật 16 frame đầu:
```
[-1, 3, 3, 3, 1, 3, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2]
 ▲start ▲quay phải 3 lần  ▲tiến  …
```

### 2.2. `pose.{setting}` — camera đang ở đâu (4×4)
`list<list<float32>>` shape `(4,4)`. Đo thật `pose.60cm_30deg` frame 0:
```
[[-0.000  -0.500   0.866   0.000]
 [-1.000   0.000  -0.000   0.000]
 [ 0.000  -0.866  -0.500   0.600]  ←  z = 0.6 m = 60 cm  → khớp "60cm"!
 [ 0.000   0.000   0.000   1.000]]
```
Cột 3 = trục quang = `(cos30°, 0, −sin30°)` → **cúi 30°** → khớp "30deg".
Giải nghĩa đầy đủ quy ước: [04](04_data_train_s2.md) mục 5.

### 2.3. `goal.{setting}` — điểm đích trên ảnh `[u, v]`
`fixed_size_list<int32>[2]`. `u` = cột, `v` = hàng, tính **riêng cho góc nhìn của setting đó**.
`(-1,-1)` = frame này không có đích. Đo thật `goal.60cm_30deg` (ảnh 640×480):
```
frame 0: [500,  93]      frame 3: [300, 178]
frame 1: [368,  86]      …
frame 2: [241,  87]      frame 7: [-1, -1]   ← không có đích
```

### 2.4. `relative_goal_frame_id.{setting}` — còn mấy frame tới đích
`int32`. `-1` = không có. Đo thật:
```
[25, 24, 23, 11, 10, 9, 8, -1, 7, -1, 15, 14, 13, …]
  ▲còn 25 frame              ▲không có   ▲đích mới, còn 15 frame
```
Loader dùng số này để **cắt cửa sổ** `[t, t+k+1]` khi tạo mẫu train.

### 2.5. Năm cột quản lý (chuẩn LeRobot)

| Cột | Nghĩa | Ví dụ thật |
|---|---|---|
| `timestamp` | thời điểm frame (giây) | `0.0, 0.0333, 0.0667, …` (bước 1/30 s) |
| `frame_index` | thứ tự frame **trong episode** | 0..45 |
| `episode_index` | frame này thuộc episode nào | 0 |
| `index` | thứ tự frame **trong cả dataset** (không reset) | |
| `task_index` | câu lệnh nào (tra `tasks.jsonl`) | `0, 0, 0, …` |

> ⚠️ Luôn dùng cột **`timestamp`** để tính thời gian, đừng suy từ `fps` (ở subset khác `fps` là giá
> trị template sai). Riêng loader S2 **không đọc** 5 cột này ([04](04_data_train_s2.md) mục 3.2).

---

## 2b. Thư mục `videos/` — vì sao có 10 folder con

```
observation.images.rgb.125cm_0deg/     observation.images.depth.125cm_0deg/
observation.images.rgb.125cm_30deg/    observation.images.depth.125cm_30deg/
observation.images.rgb.125cm_45deg/    observation.images.depth.125cm_45deg/
observation.images.rgb.60cm_15deg/     observation.images.depth.60cm_15deg/
observation.images.rgb.60cm_30deg/     observation.images.depth.60cm_30deg/
```

= **2 loại ảnh × 5 cấu hình camera**. Mỗi scene được simulator render đồng thời ở 5 cấu hình (cao
125/60 cm, cúi 0/15/30/45°). Tiền tố `observation.images.` là quy ước LeRobot: mỗi "kênh quan sát"
là một *feature* riêng.

Bên trong: ảnh từng frame `episode_XXXXXX_<frame>.{jpg|png}` — ví dụ `episode_000000_12.jpg`
(**số frame không đệm 0**; khác `vln_n1` đệm 3 chữ số — lý do ở [05](05_data_train_s1.md) mục 2.1).

> ❓ **"Quay thật có cần 5 camera không?"** — KHÔNG. 5 setting là do simulator render dư (miễn phí);
> lúc train chỉ dùng 1 (đôi khi 1 cặp) setting. Robot thật nhiều nhất cần 2 camera. Chi tiết:
> [08](08_phu_luc_thu_thap_data.md) mục 4.

---

## 3. `meta/info.json` — bản kê khai toàn dataset

| Trường | Giá trị thật | Nghĩa |
|---|---|---|
| `codebase_version` | `"v2.1"` | phiên bản chuẩn LeRobot |
| `total_episodes` | `75` | tổng số episode |
| `total_frames` | `3237` | tổng số frame |
| `total_tasks` | `75` | số câu lệnh khác nhau |
| `total_chunks` | `1` | số ngăn kéo |
| `chunks_size` | `1000` | mỗi chunk tối đa 1000 episode |
| `fps` | `30` | khung hình/giây khai báo (⚠️ hay sai ở subset khác) |
| `splits` | `{"train": "0:75"}` | episode 0–74 dùng để train |
| `data_path` | `"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"` | **khuôn** ghép đường dẫn |
| `features` | `{"action": {dtype,shape,names}, "pose.125cm_0deg": {…}, …}` | **schema mọi cột** |

**Cảnh báo:** `info.json` của bộ N1 **hay khai sai** (vd `video_path` ghi đuôi `.mp4` nhưng thực tế
lưu JPG/PNG từng frame). Quy tắc: **tin `df.dtypes` của parquet thật, không tin `info.json`.**

---

## 4. `meta/tasks.jsonl` — bảng tra "mã số → câu lệnh"

```json
{"task_index": 0, "task": "Exit the bedroom, enter the bathroom, wait at the toilet. "}
{"task_index": 1, "task": "Walk out of the dining area and walk straight into the bedroom..."}
```

**Vì sao tồn tại:** cột `task_index` trong parquet chỉ lưu **số** cho gọn. File này là **từ điển tra
ngược** số → chữ (kỹ thuật *chuẩn hoá*, tránh lưu trùng câu dài ở mọi frame).

```python
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open("meta/tasks.jsonl")}
cau_lenh = tasks[df["task_index"][0]]
```

---

## 5. `meta/episodes.jsonl` — danh mục episode (**file duy nhất loader S2 đọc trong `meta/`**)

```json
{"episode_index": 0, "tasks": ["Exit the bedroom, enter the bathroom, wait at the toilet. "], "length": 46}
{"episode_index": 1, "tasks": ["Walk out of the dining area..."], "length": 54}
```

| Trường | Nghĩa |
|---|---|
| `episode_index` | số thứ tự episode → dùng ghép đường dẫn parquet + tên file ảnh |
| `tasks` | danh sách câu lệnh; nhiều câu ngăn bằng `<INSTRUCTION_SEP>` |
| `length` | **số frame** = số hàng parquet (loader có `assert` để bắt lệch) |

```python
ep = json.loads(open("meta/episodes.jsonl").readline())
instruction = ep["tasks"][0].split("<INSTRUCTION_SEP>")[0]
n_frames    = ep["length"]      # 46
```

> 📌 Khác biệt cần biết: ở `vln_n1`, trường `tasks` là **danh sách dict** (`{"sub_instruction": …}`)
> chứ không phải danh sách chuỗi — nhưng loader S1 không đọc file này nên không ảnh hưởng.

---

## 6. `meta/episodes_stats.jsonl` — thống kê từng episode

Mỗi dòng chứa thống kê của **một episode cho mọi cột**, mỗi cột 5 số: `min`, `max`, `mean`, `std`, `count`.

```json
{"episode_index": 0, "stats": {
    "action":          {"min":[-1], "max":[3], "mean":[1.543], "std":[0.902], "count":[46]},
    "goal.125cm_0deg": {"min":[-1,-1], "max":[-1,-1], "mean":[-1.0,-1.0], "std":[0.0,0.0], "count":[46]},
    "pose.125cm_30deg":{"min":[[4×4]], "max":[[4×4]], …},
    …}}
```

**Dùng để làm gì:**
1. **Chuẩn hoá khi train** — biết range để co giãn dữ liệu.
2. **Kiểm tra nhanh chất lượng cột mà không cần mở parquet** (xem 6.1).
3. **Loader S1 dùng để cắt frame:** đọc `image_index` min/max để biết episode dùng ảnh nào
   ([05](05_data_train_s1.md) mục 7.1). *(Ở `vln_n1`, file này chỉ có 3 trường:
   `episode_index`, `task_index`, `image_index`.)*

### 6.1. Ví dụ mổ xẻ: vì sao `goal.125cm_0deg` toàn `-1`

```json
"goal.125cm_0deg": {"min":[-1,-1], "max":[-1,-1], "mean":[-1,-1], "std":[0,0], "count":[46]}
```

`min = max = [-1,-1]` → **cả 46 frame đều không có đích**.

**`goal` sinh ra thế nào?** Không phải người chấm tay — nó được tính bằng **phép chiếu hình học**:
1. Chọn một **điểm đích tương lai** (sub-goal) trong không gian 3D.
2. Dùng **pose camera** + **intrinsic `K`** để chiếu điểm đó xuống mặt phẳng ảnh → ra pixel `[u,v]`.
3. Nếu pixel **rơi ra ngoài khung** (hoặc ở sau lưng camera) → ghi `[-1,-1]`.

**Vì sao `125cm_0deg` không có đích?** Camera cao 1.25 m và **nhìn thẳng (pitch 0°)** → không chúc
xuống sàn → điểm đích trên sàn phía trước khi chiếu lên ảnh rơi khỏi khung. Ngược lại `60cm_30deg`
(thấp hơn, cúi 30°, thấy sàn) cho ra pixel thật `[500,93]`, `[368,86]`…

→ Đây chính là lý do **`data_dict` không bao giờ dùng `pitch_2 = 0`** — mọi cấu hình train đều có góc
cúi ([04](04_data_train_s2.md) mục 8). Chi tiết cách tự sinh `goal`: [06](06_pipeline_mcap_to_s2.md)
giai đoạn C.3.

---

## 7. Giải mã tên setting `{H}cm_{pitch}deg`

| Setting | Chiều cao | Góc cúi | Thấy sàn? | Có `goal`? |
|---|---|---|---|---|
| `125cm_0deg` | 125 cm | 0° | ❌ | ❌ toàn `-1` → **tránh** |
| `125cm_30deg` | 125 cm | 30° | ✅ | ✅ |
| `125cm_45deg` | 125 cm | 45° | ✅ | ✅ |
| `60cm_15deg` | 60 cm | 15° | ✅ | ✅ |
| `60cm_30deg` | 60 cm | 30° | ✅ | ✅ |

Bằng chứng ở mục 2.2: `pose.60cm_30deg` có `z = 0.6` và phần xoay mã hoá 30°.

---

## 8. Tóm tắt: đọc & tạo

| Thành phần | Là gì | Đọc bằng | Sinh bởi (nếu dùng thư viện `lerobot`) |
|---|---|---|---|
| `chunk-XXX/` | ngăn nhóm ≤1000 episode; `chunk = ep//1000` | tính từ `chunks_size` | `get_data_file_path` |
| `episode_*.parquet` | bảng số 1 episode | `pd.read_parquet` | `_save_episode_table` |
| `info.json` | kê khai schema + khuôn đường dẫn | `json.load` | `write_info` |
| `tasks.jsonl` | mã số → câu lệnh | duyệt dòng | `add_task` |
| `episodes.jsonl` | episode → câu lệnh + `length` | duyệt dòng | `write_episode` |
| `episodes_stats.jsonl` | episode → min/max/mean/std | duyệt dòng | `compute_episode_stats` |

> 💡 **Không nhất thiết phải dùng thư viện `lerobot`.** Script [tools/mcap2s2.py](tools/mcap2s2.py)
> ghi thẳng bằng `pyarrow` + `json` — ít phụ thuộc hơn và **kiểm soát được dtype**, thứ mà loader S2
> rất khó tính ([04](04_data_train_s2.md) mục 3.3).

---

*Quay lại mục lục: [00_README](00_README.md).*
