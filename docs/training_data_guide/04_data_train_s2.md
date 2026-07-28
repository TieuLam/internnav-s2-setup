# 04 — Cấu trúc data train **System 2** (`vln_ce`) — cái gì **bắt buộc**, cái gì **không**

> **File này để làm gì:** mô tả **chính xác** những gì phải có trên đĩa để `NavPixelGoalDataset` nạp
> được, và tách bạch ba mức: **(a) bắt buộc phải có**, **(b) bắt buộc tồn tại nhưng giá trị không
> quan trọng**, **(c) hoàn toàn không cần**.
>
> Mọi con số **đo thật** trên `InternNav/data/vln_ce/traj_data/r2r/17DRP5sb8fy` và đối chiếu code
> [internvla_n1_lerobot_dataset.py](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py).
> Bộ tài liệu: [03_code_train_s2](03_code_train_s2.md) · [05_data_train_s1](05_data_train_s1.md)

---

## 1. System 2 học "trò chơi" gì?

Đưa cho model **một tấm ảnh camera + một câu lệnh tiếng Anh**, nó phải trả lời một trong ba kiểu:

1. **"Đi tới điểm này"** → xuất `↓` (cúi nhìn sàn) rồi **toạ độ pixel `u v`** — *pixel goal*, câu trả
   lời quan trọng nhất.
2. **"Quay đã"** → xuất chuỗi `←`/`→` khi chưa thấy đích.
3. **"Xong rồi"** → xuất `STOP`.

Train = cho xem hàng chục nghìn tình huống kèm đáp án đúng của ba kiểu đó.

---

## 2. Cây thư mục một scene (căn phòng)

```
traj_data/<dataset>/<scene_id>/
├── meta/
│   ├── episodes.jsonl        ← ✅ BẮT BUỘC (loader chỉ đọc file này trong meta/)
│   ├── tasks.jsonl           ← ⚪ không bắt buộc (S2 loader không đọc)
│   ├── episodes_stats.jsonl  ← ⚪ không bắt buộc
│   └── info.json             ← ⚪ không bắt buộc
├── data/chunk-000/
│   └── episode_000000.parquet    ← ✅ BẮT BUỘC (bảng số từng frame)
└── videos/chunk-000/
    ├── observation.images.rgb.{H}cm_{pitch_1}deg/   episode_000000_0.jpg …   ← ✅ ảnh NHÌN THẲNG
    ├── observation.images.rgb.{H}cm_{pitch_2}deg/   episode_000000_0.jpg …   ← ✅ ảnh NHÌN CÚI
    └── observation.images.depth.{H}cm_{pitch_2}deg/ episode_000000_0.png …   ← ✅ ảnh ĐỘ SÂU
```

- **`{H}`** = chiều cao camera (cm), **`{pitch_1}`** = góc nhìn thẳng, **`{pitch_2}`** = góc cúi.
- **`setting = f"{H}cm_{pitch_2}deg"`** — chuỗi định danh dùng làm **hậu tố tên cột parquet**
  ([dòng 850](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L850)).
- **Mỗi scene là một dataset LeRobot độc lập**: `episode_index` đánh số lại từ 0 ở mỗi scene.
- `chunk = episode_index // 1000` — xem [07](07_phu_luc_lerobot_format.md) mục 1.

### 2.1. 💡 Chỉ cần **2 thư mục ảnh** nếu bạn dùng cấu hình 1 camera

Loader lấy đường dẫn ảnh cúi bằng phép **thay chuỗi**
([dòng 1018](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L1018)):

```python
lookdown = image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')
```

→ Nếu `pitch_1 == pitch_2` (cấu hình `60cm_15_15`, `60cm_30_30`) thì phép thay **không đổi gì** →
ảnh "thẳng" và ảnh "cúi" **là cùng một file**. Bạn chỉ cần **1 thư mục rgb + 1 thư mục depth**, tức
**một camera thật là đủ**.

| Cấu hình trong `data_dict` | pitch_1 | pitch_2 | Số thư mục rgb cần | Số camera thật |
|---|---|---|---|---|
| `60cm_15_15` | 15° | 15° | 1 | **1** |
| `60cm_30_30` | 30° | 30° | 1 | **1** |
| `125cm_0_30` | 0° | 30° | 2 | 2 (hoặc 1 camera cúi được) |
| `125cm_0_45` | 0° | 45° | 2 | 2 |

> ⚠️ **Bẫy tên đường dẫn.** Loader dựng đường dẫn depth bằng `.replace('rgb', 'depth')` trên **toàn
> bộ đường dẫn** ([dòng 1021](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L1021)).
> Nếu thư mục cha nào của bạn có chữ `rgb` (vd `/data/rgb_logs/...`), đường dẫn depth sẽ hỏng. **Đừng
> đặt tên thư mục cha chứa "rgb".**

---

## 3. Bảng số (`parquet`) — từng cột, kèm **dtype chính xác**

Đọc bằng `pandas.read_parquet` → mỗi **hàng = một frame**. Scene mẫu: episode 0 có **46 hàng × 21 cột**.

### 3.1. Bốn cột loader THỰC SỰ đọc

| Cột | Kiểu Arrow chính xác | Nghĩa | Bắt buộc? |
|---|---|---|---|
| `action` | `int32` | Nút bấm tại frame đó: `1=↑ tiến`, `2=← trái`, `3=→ phải`, `5=↓ cúi`, `0=STOP`, **`-1` = frame khởi đầu** | ✅ **Giá trị phải đúng** |
| `goal.{setting}` | `fixed_size_list<int32>[2]` | **Điểm đích trên ảnh `[u, v]`** (`u`=cột, `v`=hàng). `(-1,-1)` = frame này không có đích | ✅ **Nhãn chính — phải đúng** |
| `relative_goal_frame_id.{setting}` | `int32` | Còn **bao nhiêu frame** nữa tới đích; `-1` = không có đích | ✅ **Nhãn chính — phải đúng** |
| `pose.{setting}` | `list<list<float32>>` (4×4) | Ma trận **camera → world** tại frame đó | ✅ cột phải tồn tại · giá trị: xem mục 6 |

### 3.2. Năm cột quản lý chuẩn LeRobot

| Cột | Kiểu | Loader S2 có đọc không? |
|---|---|---|
| `timestamp` | `float32` | ❌ không |
| `frame_index` | `int64` | ❌ không |
| `episode_index` | `int64` | ❌ không |
| `index` | `int64` | ❌ không |
| `task_index` | `int64` | ❌ không |

→ **Không bắt buộc với loader S2**, nhưng **nên có** để dataset đúng chuẩn LeRobot v2.1 và để các
công cụ khác (LeRobot CLI, viewer) đọc được.

### 3.3. 🚨 Vì sao **dtype** quan trọng đến thế

Loader gọi `.tolist()` **trên từng ô**
([dòng 786-789](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L786)):

```python
ep_poses       = df[pose_key].apply(lambda x: x.tolist()).tolist()
ep_pixel_goals = [[df[rel_key][i].tolist(), df[goal_key][i].tolist()] for i in range(len(df))]
```

`.tolist()` là phương thức của **numpy**. Nếu bạn ghi parquet bằng kiểu Python thuần (`int`,
`list[int]`), pandas trả về `int`/`list` → **`AttributeError: 'int' object has no attribute
'tolist'`** → lỗi bị `try/except` ở [dòng 816] nuốt → **bỏ nguyên scene, dataset rỗng, không có
traceback**.

✅ Cách ghi đúng (dùng `pyarrow`, xem [tools/mcap2s2.py](tools/mcap2s2.py) hàm `write_parquet`):

```python
import pyarrow as pa
table = pa.table({
    "action":                          pa.array(actions, type=pa.int32()),
    f"pose.{s}":                       pa.array([p.tolist() for p in poses], type=pa.list_(pa.list_(pa.float32()))),
    f"goal.{s}":                       pa.array([g.tolist() for g in goals], type=pa.list_(pa.int32(), 2)),
    f"relative_goal_frame_id.{s}":     pa.array(rel_ids, type=pa.int32()),
    ...
})
```

### 3.4. Số liệu thật để đối chiếu (episode 0, setting `60cm_30deg`)

```
action : [-1, 3, 3, 3, 1, 3, 1, 1, 2, 1, 2, 1, ...]
goal   : [[500,93], [368,86], [241,87], [300,178], [298,212], [185,221], [158,272], [-1,-1], ...]
rel_id : [  25,      24,       23,       11,        10,        9,         8,         -1,     ...]
```

**Cách đọc `rel_id`:** các giá trị giảm dần `25,24,23` nghĩa là ba frame đầu **cùng nhắm tới frame
25**. Rồi `11,10,9,8` = bốn frame tiếp cùng nhắm frame 14. `-1` xen giữa = **frame đó không nhìn
thấy đích** (đang xoay, hoặc đích rơi ra ngoài khung).

---

## 4. Ảnh — định dạng và đơn vị (rất dễ sai)

| Loại | Đuôi | Kích thước đo thật | Kiểu số | Ghi chú |
|---|---|---|---|---|
| RGB (nhìn thẳng + nhìn cúi) | **`.jpg`** | 640×480, mode `RGB` | uint8 (0–255) | Model **tự resize về 384×384** → **đừng resize trước** |
| Depth (ở góc `pitch_2`) | **`.png`** | 640×480, mode `I;16` | **uint16, đơn vị MILIMÉT** | Loader **chia 1000** → mét, rồi clip 5 m; resize 224×224 |

Tên file: `episode_{episode_index:06d}_{frame_index}.{jpg|png}` — ví dụ `episode_000000_12.jpg`.
(Chú ý: số frame **không** đệm số 0.)

> 🚨 **Bẫy đơn vị depth.** S2 chia **1000**, S1 chia **10000**
> ([05](05_data_train_s1.md) mục 4). Sai hằng số → độ sâu lệch 10 lần mà **không có lỗi nào báo ra**.
>
> Đo thật trên `episode_000000_0.png`: `dtype=uint16`, min = 388, max = 6645 → tức 0.388 m … 6.6 m. ✓

---

## 5. `pose.{setting}` — quy ước hình học (phần dễ sai nhất)

### 5.1. Nó là ma trận gì?

Docstring trong code ghi *"T_world2camera"* nhưng **giá trị thật là camera → world**: cột thứ 4 chính
là **vị trí camera trong world**. Đo thật `pose.60cm_30deg` frame 0:

```
[[-0.000  -0.500   0.866   0.000]     cột 1 = trục "phải" của camera
 [-1.000   0.000  -0.000   0.000]     cột 2 = trục "xuống"
 [ 0.000  -0.866  -0.500   0.600]     cột 3 = TRỤC QUANG  = (cos30°, 0, −sin30°) → CÚI 30° ✓
 [ 0.000   0.000   0.000   1.000]]    cột 4 = vị trí (0, 0, 0.6) → CAO 60 cm ✓
```

→ Tên setting `60cm_30deg` được **chứng minh** ngay trong con số: `z = 0.6` m và phần xoay mã hoá 30°.

### 5.2. Công thức dựng lại (đã kiểm chứng khớp 100% với data gốc)

Cho pose robot `(x, y, yaw)` trên mặt sàn, chiều cao camera `h`, góc cúi `p`:

```python
Rz = rot_z(yaw)
z_cam = Rz @ (cos p, 0, -sin p)     # trục quang: nhìn ra trước, chúc xuống p độ
x_cam = Rz @ (0, -1, 0)             # "phải" của camera = -y của robot
y_cam = cross(z_cam, x_cam)         # "xuống" của camera (giữ hệ thuận tay phải OpenCV)
T[:3,0], T[:3,1], T[:3,2] = x_cam, y_cam, z_cam
T[:3,3] = (x, y, h)
```

(Code chạy được: [tools/mcap2s2.py](tools/mcap2s2.py) hàm `camera_pose_from_base`.)

### 5.3. Cách tự kiểm định quy ước (round-trip test)

Hàm `get_trajectory_relative_to_frame(extrinsics, camera_deg=pitch_2)`
([dòng 592](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L592)) là "trọng tài":
nó phải trả về quỹ đạo `(x_tiến, y_trái, yaw)` **bắt đầu bằng `(0,0,0)`** và tăng đúng theo chuyển
động thật. Chạy trên pose do [tools/mcap2s2.py](tools/mcap2s2.py) sinh ra:

```
[[0.   0.   0.   ]   ← gốc
 [0.25 0.   0.   ]   ← tiến 0.25 m
 [0.5  0.   0.   ]
 ...
 [2.5  0.  -0.262]]  ← xoay phải 15°
```

→ Khớp chính xác kết quả chạy trên data gốc. **Nếu tự sinh pose, hãy chạy test này trước khi train.**

---

## 6. ⭐ BẢNG BẮT BUỘC / KHÔNG BẮT BUỘC (phần cốt lõi của file này)

### 6.1. Mức FILE

| Thành phần | Train S2 (`pixel_goal_only=False`) | Train dual (`pixel_goal_only=True`) | Điều gì xảy ra nếu thiếu |
|---|---|---|---|
| `meta/episodes.jsonl` | ✅ **Bắt buộc** | ✅ | `FileNotFoundError` → scene bị bỏ im lặng |
| `data/chunk-XXX/episode_XXXXXX.parquet` | ✅ **Bắt buộc** | ✅ | như trên |
| `videos/.../rgb.{H}cm_{pitch_1}deg/*.jpg` | ✅ **Bắt buộc** | ✅ | crash ở `__getitem__` |
| `videos/.../rgb.{H}cm_{pitch_2}deg/*.jpg` | ✅ **Bắt buộc** (dù `pitch_1==pitch_2` thì trùng file) | ✅ | crash |
| `videos/.../depth.{H}cm_{pitch_2}deg/*.png` | ✅ **Bắt buộc phải TỒN TẠI** | ✅ | crash — `Image.open` chạy ở **mọi frame**, [dòng 1020] |
| `meta/tasks.jsonl` · `meta/info.json` · `meta/episodes_stats.jsonl` | ⚪ Không cần | ⚪ Không cần | loader S2 không đọc |
| 4 setting camera còn lại | ⚪ Không cần | ⚪ Không cần | chỉ cần setting đang khai trong `data_dict` |
| `pointcloud.ply` | ⚪ Không cần | ⚪ Không cần | đó là thứ của **System 1** |

### 6.2. Mức CỘT trong parquet

| Cột | Phải tồn tại? | Giá trị phải đúng? |
|---|---|---|
| `action` | ✅ | ✅ **Có** — dùng để phân loại mẫu *turn* và làm đáp án của mẫu *turn* |
| `goal.{setting}` | ✅ | ✅ **Có** — đây là nhãn model học xuất ra |
| `relative_goal_frame_id.{setting}` | ✅ | ✅ **Có** — quyết định mẫu là *turn* hay *pixel_goal*, và độ dài cửa sổ |
| `pose.{setting}` | ✅ **Bắt buộc tồn tại** | ⚠️ **Train S2: KHÔNG** (chỉ dùng làm lá cờ `is not None`) · **Train dual: CÓ** |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | ⚪ Không | — |

> 🔴 **Không được bỏ hẳn cột `pose.{setting}`** dù train S2. Điều kiện ở
> [dòng 785](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L785) đòi **cả ba** cột
> `pose/goal/relative_goal_frame_id` cùng có mặt; thiếu một cái → rơi vào nhánh `else` viết dở →
> `NameError` → **bỏ nguyên scene**.

### 6.3. Mức GIÁ TRỊ — "tồn tại nhưng không cần đúng"

Đây là phần tiết kiệm công sức nhiều nhất khi bạn **chỉ train S2**:

| Dữ liệu | Train S2 | Vì sao |
|---|---|---|
| **Giá trị `pose` 4×4** | Có thể là **ma trận đơn vị / pose thô từ odometry** | Chỉ được dùng làm cờ `pose is not None` ([1034], [1069]). Giá trị số chỉ vào nhãn ở khối 🔵 `pixel_goal_only` ([1110-1131]) — khối này **không chạy** khi train S2. |
| **Giá trị pixel trong ảnh depth** | Có thể là **ảnh uint16 rỗng/xấp xỉ** | Depth chỉ được nạp rồi bỏ vào `traj_depths`, và `traj_depths` chỉ vào batch khi `pixel_goal_only=True`. **Nhưng file vẫn phải tồn tại, đúng uint16.** |
| **Ảnh RGB `pitch_2` ở các frame không phải `start_frame_id`** | Có thể trùng ảnh khác | Chỉ ảnh lookdown **tại `start_frame_id`** đi vào input model ([1035-1037]). |
| **Câu lệnh** | ❗ **Bắt buộc thật** | Là input chính; câu lệnh rỗng ⇒ mẫu vô nghĩa (model không có gì để bám). |

> ⚖️ **Lời khuyên thực dụng:** dồn công sức làm chuẩn **RGB + `goal` + `relative_goal_frame_id` +
> câu lệnh**. `pose` chỉ cần đúng *định dạng*; `depth` chỉ cần *tồn tại và đúng kiểu*. Nhưng **nếu
> có ý định train dual về sau**, hãy làm `pose` và `depth` đúng ngay từ đầu — sinh lại data sau này
> tốn hơn nhiều.

---

## 7. Câu lệnh (instruction) — lấy từ đâu

`meta/episodes.jsonl`, mỗi dòng một episode (đo thật):

```json
{"episode_index": 0, "tasks": ["Exit the bedroom, enter the bathroom, wait at the toilet. "], "length": 46}
```

| Trường | Bắt buộc? | Ghi chú |
|---|---|---|
| `episode_index` | ✅ | dùng để ghép đường dẫn parquet và tên file ảnh |
| `tasks` | ✅ | **list**; loader lấy `tasks[0]`, tách `<INSTRUCTION_SEP>` → **mỗi câu lệnh thành một episode riêng** trong tập train (nhân bản dữ liệu) |
| `length` | ✅ | **phải bằng đúng số hàng parquet** — có `assert` ở [dòng 793] |

Câu lệnh là **phần bắt buộc con người phải viết** — máy không tự bịa ra được. Tiếng Anh tự nhiên,
mô tả lộ trình theo mốc nhìn thấy được, ví dụ:
> *"You are facing toward the entrance of the church, turn around and stop in between the steps…"*

---

## 8. Bảng "cấu hình camera nào dùng được"

`data_dict` chỉ khai 4 tổ hợp; **`pitch_2` luôn ∈ {15°, 30°, 45°}, không bao giờ 0°**:

| setting (`{H}cm_{pitch_2}deg`) | Thấy sàn? | Có `goal` thật? | Dùng train được? |
|---|---|---|---|
| `125cm_0deg` | ❌ | ❌ (đo thật: toàn `[-1,-1]`) | ❌ **Không** — và cũng không có entry nào trong `data_dict` dùng nó làm `pitch_2` |
| `125cm_30deg` | ✅ | ✅ | ✅ |
| `125cm_45deg` | ✅ | ✅ | ✅ |
| `60cm_15deg` | ✅ | ✅ | ✅ |
| `60cm_30deg` | ✅ | ✅ | ✅ |

> **Bài học khi tự thu data:** camera **phải cúi** đủ để thấy sàn phía trước. Camera nhìn thẳng đơ →
> waypoint trên sàn rơi ra ngoài khung → `goal` toàn `-1` → không có mẫu *pixel_goal* → data vô dụng.

---

## 9. Tóm tắt: để có MỘT ví dụ train S2, bạn cần

| Thành phần | Bắt buộc | Nguồn lấy |
|---|---|---|
| Ảnh RGB nhìn thẳng (`pitch_1`) | ✅ | camera |
| Ảnh RGB nhìn cúi (`pitch_2`) | ✅ (trùng ảnh trên nếu `pitch_1==pitch_2`) | camera |
| Ảnh Depth uint16 mm (`pitch_2`) | ✅ tồn tại | camera RGB-D **hoặc** DepthAnythingV2 **hoặc** ảnh giữ chỗ nếu chỉ train S2 |
| Câu lệnh tiếng Anh | ✅ | **con người viết** |
| `action` rời rạc mỗi frame | ✅ | suy từ chuyển động (Δvị trí, Δgóc) |
| `goal [u,v]` | ✅ | chiếu waypoint 3D về ảnh (cần `K` + pose), hoặc chấm tay |
| `relative_goal_frame_id` | ✅ | khoảng cách (số frame) tới waypoint |
| `pose` 4×4 | ✅ cột phải có | SLAM / ARKit / simulator — giá trị chỉ cần đúng nếu train dual |

---

## 10. Checklist tự kiểm trước khi train

- [ ] Tên cột đúng công thức `{height}cm_{pitch_2}deg` (dùng **pitch_2**, không phải pitch_1).
- [ ] dtype parquet đúng: `int32` / `list<list<float32>>` / `fixed_size_list<int32>[2]`.
- [ ] `episodes.jsonl` có `length` **bằng đúng** số hàng parquet.
- [ ] Ảnh RGB `.jpg`, depth `.png` **uint16 milimét**.
- [ ] Có ít nhất một frame với `relative_goal_frame_id ≥ 3` (nếu không: **0 mẫu pixel_goal**).
- [ ] Đường dẫn không chứa chữ `rgb` ở thư mục cha.
- [ ] Đã đăng ký entry trong `data_dict` và trỏ `--vln_dataset_use` đúng tên.
- [ ] Chạy round-trip `get_trajectory_relative_to_frame` (mục 5.3) nếu định train dual.

**Cách kiểm nhanh bằng chính loader thật** (không cần GPU, không cần cài `decord`/`torchcodec`):

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
print(len(L.NavPixelGoalDataset(tokenizer=None, data_args=a)))   # > 0 là format đúng
```

(Đã chạy thật trên dataset do [tools/mcap2s2.py](tools/mcap2s2.py) sinh ra: 27 mẫu với
`pixel_goal_only=False`, 14 mẫu với `True` — trùng khớp con số mà bước tự kiểm định của script báo.)

---

Tiếp theo: data cho System 1 → [05_data_train_s1](05_data_train_s1.md).
