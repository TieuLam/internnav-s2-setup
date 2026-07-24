# 03 — Data Contract: cấu tạo `vln_ce` · `vln_pe` · `vln_n1` và cách dùng chúng để chạy thử eval

> **File này để làm gì:** mô tả cấu trúc dữ liệu thật của 3 subset trong dataset
> `InternRobotics/InternData-N1` (HuggingFace), và hướng dẫn **lấy dữ liệu từ đó đút vào các function
> đã mô tả trong [02_code_structure](02_code_structure.md)** (`s2_step`, `s1_step_latent`,
> `agent.step`) khi muốn chạy thử eval open-loop.
>
> Mọi số liệu schema là **đo thật trên file tải về** (21–22/07/2026), không chép từ dataset card —
> vì card và `info.json` đã được chứng minh là sai ở nhiều chỗ (PL-D1, PL-D2 trong
> [05_appendix](05_appendix.md)). Chỗ nào chưa chạy thử sẽ đánh dấu ⬜ và nói rõ.

---

## 0. Điều quan trọng nhất: KHÔNG có "một schema của InternVLA-N1"

InternVLA-N1 là dual-system → **mỗi system ăn một loại data khác nhau**. Ba subset của
`InternData-N1` không phải 3 định dạng của cùng một thứ:

| Subset | Nuôi cái gì | Action | Đặc điểm nhận dạng | Bằng chứng |
|---|---|---|---|---|
| **`vln_ce`** (Continuous Environment) | **System 2** (VLM chấm pixel-goal) | int rời rạc `{1↑,2←,3→,5↓}` + pixel goal | cột `pose.{setting}` / `goal.{setting}`; **có sẵn RGB+depth PNG** | PL-D4, PL-D5 |
| **`vln_n1`** (tên model N1) | **System 1** (NavDP, quỹ đạo liên tục) | **ma trận 4×4 SE(3)** mỗi frame | cột `camera_intrinsic` / `camera_extrinsic`; nén `.tar.gz` | PL-D3, PL-D5 |
| **`vln_pe`** (Physical Embodiment) | **Baseline CMA/RDP** (đối chứng, không thuộc dual-system) | int rời rạc | cột `position/orientation/yaw/progress/step`; file rời | PL-D2, PL-D5 |

Cách chứng minh (PL-D5): mỗi loader trong `internnav/dataset/*_lerobot_dataset.py` đọc đúng bộ cột
của một subset — `navdp_lerobot_dataset.py` ↔ `vln_n1`, `internvla_n1_lerobot_dataset.py` ↔ `vln_ce`,
`cma_lerobot_dataset.py` ↔ `vln_pe`. Ba subset **không thay thế được cho nhau**: đưa nhầm subset là
loader tìm không thấy cột và crash (hoặc tệ hơn — chạy sai im lặng).

Cả ba đều bọc trong khung **LeRobotDataset v2.1**, mỗi scene = một dataset độc lập:

```
<scene>/
├── data/chunk-000/episode_XXXXXX.parquet     ← bảng số (cột KHÁC NHAU tuỳ subset)
├── videos/chunk-000/<key>/...                ← ảnh (mp4 / npy / png tuỳ subset)
└── meta/{info,episodes,tasks,episodes_stats}.jsonl
```

---

## 1. `vln_ce` — data của System 2 (dùng chính cho eval open-loop)

### 1.1. Vị trí & cách tải (file nhỏ nhất toàn repo chỉ 16.16 MB)

```
vln_ce/traj_data/<subset>/<scene>.tar.gz        subset ∈ {r2r, rxr, scalevln}
                                                (r2r 61 file · rxr 59 · scalevln 794)
```

```python
from huggingface_hub import hf_hub_download
import tarfile

tgz = hf_hub_download("InternRobotics/InternData-N1",
        filename="vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz",   # file nhỏ nhất, để thử
        repo_type="dataset", local_dir="/kaggle/temp/cedl")   # NHỚ repo_type (PL-A5)
tarfile.open(tgz).extractall("/kaggle/temp/cex")              # giải nén ở /kaggle/temp (PL-A2)
```

### 1.2. Parquet — 21 cột (đo thật trên `rxr/YmJkqBEsHnH`, PL-D4)

| Cột | dtype/shape | Nội dung |
|---|---|---|
| `action` | int32 (1) | rời rạc `{1:↑, 2:←, 3:→, 5:↓}`; **`-1` = frame bắt đầu**. Đo được: `[-1,2,2,2,2,2,2,2,2,2,1,1,2,1,2,1]` |
| `pose.{setting}` | float32 (4,4) | pose camera từng frame — ×5 setting |
| `goal.{setting}` | int32 (2) | **pixel goal `[u, v]`** (= [cột, hàng]); `(-1,-1)` = không có — ×5 setting |
| `relative_goal_frame_id.{setting}` | int32 (1) | số frame còn lại tới đích; `-1` = không có — ×5 setting |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | | ✅ đủ 5 trường LeRobot chuẩn |

→ 1 + 5×3 + 5 = 21 cột. `{setting}` là cấu hình camera, ghép theo công thức trong loader
(`internvla_n1_lerobot_dataset.py`): **`f'{height}cm_{pitch_2}deg'`**:

| setting | height | pitch | dùng được cho eval? |
|---|---|---|---|
| `125cm_0deg` | 125 cm | 0° | ❌ **TRÁNH** — camera nhìn thẳng, không thấy sàn → `goal` toàn `(-1,-1)` (đo thật, PL-D4); cũng bị loại khỏi `train_system2.sh` |
| `125cm_30deg` | 125 cm | 30° | ✅ |
| `125cm_45deg` | 125 cm | 45° | ✅ |
| `60cm_15deg` | 60 cm | 15° | ✅ |
| `60cm_30deg` | 60 cm | 30° | ✅ |

### 1.3. Ảnh — RGB + depth CÓ SẴN (điểm ăn tiền của `vln_ce`)

Lưu PNG **từng frame** (không phải mp4 dù `info.json` ghi vậy — lỗi nhỏ đã ghi nhận PL-D4):

```
videos/chunk-000/observation.images.rgb.{setting}/episode_XXXXXX_<frame>.png     640×480, uint8
videos/chunk-000/observation.images.depth.{setting}/episode_XXXXXX_<frame>.png   640×480, uint16 (mode I;16)
```

**Depth đơn vị milimét, clip tại 10000 (= 10 m)** — đo thật (PL-D4). Vì có sẵn cả RGB lẫn depth,
`vln_ce` đủ đầu vào cho **cả S2 lẫn S1** mà không cần chạy DepthAnything.

### 1.4. `meta/` — nơi lấy câu lệnh (instruction)

- `tasks.jsonl` — câu lệnh tiếng Anh tự nhiên, ví dụ đo được: *"You are facing toward the entrance
  of the church, turn around and stop in between the steps..."*
- `episodes.jsonl` — `{"episode_index", "tasks": [...], "length"}`; một episode có thể nhiều
  instruction, ngăn bằng `<INSTRUCTION_SEP>` (loader lấy `tasks[0].split("<INSTRUCTION_SEP>")`);
  `length` luôn bằng số dòng parquet (loader có `assert`).

---

## 2. `vln_n1` — data của System 1 (GT quỹ đạo liên tục)

### 2.1. Vị trí & cách tải (mỗi file ~250 MB — nặng, cân nhắc trước khi tải)

```
vln_n1/traj_data/<simulator>_<camera>/<scene_uuid>.tar.gz
  simulator ∈ {3dfront, gibson, hm3d, hssd, matterport3d, replica}
  camera    ∈ {d435i, zed}          ← mỗi scene render 2 lần với 2 camera THẬT
```

Toàn bộ 3 774 file đều là `.tar.gz` — **không có file `meta/` rời** để tải lẻ (PL-D1).
Mẫu đã đo: `matterport3d_d435i/pLe4wQe7qrG.tar.gz` (248.7 MB).

### 2.2. Parquet — đúng 4 cột (PL-D3)

| Cột | dtype/shape | Nội dung |
|---|---|---|
| `index` | int64 | 0,1,2,… |
| `observation.camera_intrinsic` | float32 (3,3) | hằng số qua mọi frame |
| `observation.camera_extrinsic` | float32 (4,4) | pose camera |
| `action` | float32 **(4,4) SE(3)** | thay đổi từng frame → **chính là quỹ đạo camera** (GT của S1) |

Intrinsic đo được ở biến thể `_d435i`: `fx=355.81, fy=351.69, cx=240, cy=135` → ảnh **480×270**,
FOV 68°×42° → khớp spec Intel RealSense D435i. Nghĩa là: muốn sinh data giống `vln_n1`, camera phải
mô phỏng đúng model thật, không phải camera tuỳ ý. ⬜ Biến thể `_zed` chưa đo.

⚠️ **`vln_n1` KHÔNG có** `timestamp/frame_index/episode_index/task_index` (trường bắt buộc của
LeRobot v2.1) → loader LeRobot gốc không đọc được, chỉ loader riêng `navdp_lerobot_dataset.py` của
InternNav đọc được (PL-D3). Video dạng `.mp4`. `info.json` của subset này thì **khớp** parquet.

### 2.3. Dùng để làm gì trong eval

`vln_ce` **không có** GT quỹ đạo liên tục — nên với S1, eval trên `vln_ce` chỉ đánh giá **định tính**
(quỹ đạo có hợp lý không). Muốn có GT quỹ đạo để so số thì phải lấy từ `vln_n1` (cột `action` SE(3)).
Đây cũng là data mà script train S1 trỏ vào: `scripts/train/base_train/configs/navdp.py` khai
`root_dir='data/datasets/InternData-N1/vln_n1/traj_data'` (đọc local 23/07 — PL-D5).

---

## 3. `vln_pe` — data của baseline CMA/RDP (đối chứng, KHÔNG phải data dual-system)

### 3.1. Vị trí — subset duy nhất là file rời (tải lẻ vài trăm KB được)

```
vln_pe/traj_data/r2r_aliengo/<scene_id>/{meta/, data/, videos/}    ← 122 scene, ~5 193 episode
```

`r2r_aliengo` = episode R2R render trên robot chó 4 chân **Unitree Aliengo** trong Isaac Sim —
nên pose camera và pose robot tách riêng (có offset lắp đặt).

### 3.2. Parquet — 14 cột (PL-D2)

`observation.camera_position/orientation/yaw` (vị trí + quaternion + góc quay của camera),
`observation.robot_position/orientation/yaw`, `observation.progress`, `observation.step`
(bước sim 0,50,100 — không phải frame index), `observation.action` (**int64 rời rạc** — đo được
3,3,1 = phải, phải, tiến), + đủ 5 trường LeRobot chuẩn.

### 3.3. 🚨 Cảnh báo bắt buộc khi dùng `vln_pe`: `meta/info.json` của nó MÔ TẢ SAI chính nó

`features` trong `info.json` khai `camera_intrinsic (3,3)` / `camera_extrinsic (4,4)` /
`action float32 (4,4)` — **không cột nào tồn tại trong parquet**. Nguyên nhân đã truy tận gốc:
đây là **template của `vln_n1` bị copy sang mà không sửa** (PL-D2). Kèm hai lỗi nữa:
`fps: 30` nhưng timestamp thật 6 Hz (tính giờ bằng `frame/fps` sai 5 lần — **luôn dùng cột
`timestamp`**), và `splits` khai 1 episode trong khi có 23.

→ Quy tắc làm việc với dataset này: **schema lấy từ `df.dtypes` của parquet thật, không tin
`info.json`** (nguyên tắc "khai báo mâu thuẫn dữ liệu thì dữ liệu thắng").

### 3.4. Dùng để làm gì

Chạy baseline CMA/RDP làm đối chứng SR/SPL, hoặc làm "lưới an toàn" khi dual-system chưa chạy được
(baseline nhẹ ~0.15–0.57 GB, không cần checkpoint 16.79 GB). Checkpoint tương ứng nằm ở repo HF
`InternRobotics/VLN-PE` — lưu ý 4/7 thư mục thiếu `config.json`, xem
[04_checkpoint_details](04_checkpoint_details.md) mục 3.

---

## 4. Cách dùng data để truyền vào các function trong [02_code_structure](02_code_structure.md)

> Mục tiêu: eval **open-loop** trên Kaggle — cho model xem ảnh render sẵn của `vln_ce`, không cần
> simulator. Căn cứ khả thi: agent chạy trên observation tĩnh, pose hardcode ma trận đơn vị (PL-C5).
>
> ⬜ **Trạng thái trung thực:** các sketch dưới đây dựng từ (a) schema đo thật ở mục 1, (b) chữ ký
> function + tiền xử lý đã xác minh từng dòng trong file 02 — nhưng **pipeline ghép end-to-end chưa
> được chạy thật**. Blocker hiện tại: lần chạy S2 thật gần nhất rơi nhánh action, chưa sinh latent
> (PL-E1) → bước 4.3 chỉ chạy được sau khi gỡ blocker đó.

### 4.1. Dựng "một observation" từ `vln_ce`

Chọn MỘT setting có góc cúi (vd `60cm_30deg`) và dùng thống nhất:

```python
import glob, json, pandas as pd
import numpy as np
from PIL import Image

SETTING = "60cm_30deg"
SCENE   = "/kaggle/temp/cex"          # thư mục đã giải nén ở mục 1.1

df = pd.read_parquet(sorted(glob.glob(f"{SCENE}/**/data/**/*.parquet", recursive=True))[0])

# Câu lệnh: meta/episodes.jsonl -> tasks[0] (tách <INSTRUCTION_SEP> nếu có)
episodes = [json.loads(l) for l in open(glob.glob(f"{SCENE}/**/meta/episodes.jsonl", recursive=True)[0])]
instruction = episodes[0]["tasks"][0].split("<INSTRUCTION_SEP>")[0]

def load_frame(ep_idx, frame_idx):
    rgb_dir   = glob.glob(f"{SCENE}/**/videos/**/observation.images.rgb.{SETTING}",   recursive=True)[0]
    depth_dir = glob.glob(f"{SCENE}/**/videos/**/observation.images.depth.{SETTING}", recursive=True)[0]
    rgb   = np.array(Image.open(f"{rgb_dir}/episode_{ep_idx:06d}_{frame_idx}.png"))       # (480,640,3) uint8
    depth = np.array(Image.open(f"{depth_dir}/episode_{ep_idx:06d}_{frame_idx}.png"))     # (480,640) uint16, mm
    return rgb, depth

# Ground-truth để chấm điểm sau này:
gt_action = df["action"].tolist()                     # {-1,1,2,3,5}
gt_goal   = df[f"goal.{SETTING}"].tolist()            # [u,v] hoặc (-1,-1)
gt_pose   = df[f"pose.{SETTING}"].tolist()            # 4x4
```

(⬜ pattern tên file PNG `episode_XXXXXX_<frame>.png` ghi theo cây thư mục đo 22/07 — xác nhận lại
bằng `ls` khi giải nén, đừng tin mù.)

### 4.2. Truyền vào System 2 — `policy.s2_step(...)`

Đối chiếu chữ ký ở file 02 mục 3: `s2_step(rgb, depth, pose, instruction, intrinsic, look_down)`.

| Tham số | Lấy từ đâu trong `vln_ce` | Lưu ý |
|---|---|---|
| `rgb` | PNG RGB (mục 4.1), **giữ nguyên np.ndarray 640×480** | policy TỰ resize về 384×384 bên trong (`internvla_n1_policy.py:113–115`) — không resize trước 2 lần |
| `depth` | PNG depth | S2 không đưa depth vào prompt; truyền để giữ đúng chữ ký |
| `pose` | `np.eye(4)` | agent gốc cũng truyền ma trận đơn vị (PL-C5) |
| `instruction` | `meta/episodes.jsonl` / `tasks.jsonl` | 1 câu tiếng Anh |
| `intrinsic` | tự dựng như agent: `get_intrinsic_matrix(width, height, hfov)` (`internvla_n1_agent.py:119–131`) | ⬜ hfov đúng của `vln_ce` chưa đo — agent tính từ config `width/height/hfov` |
| `look_down` | `False` (frame thường) | frame sau action `5 (↓)` mới là `True` |

Kết quả nhận về `S2Output`: nếu `output_pixel`/`output_latent` khác `None` → so `output_pixel` với
`gt_goal` (**⚠️ nhớ hai điều: policy trả `[row, col]` trong khi data lưu `[u, v]` = [cột, hàng] —
phải đảo khi so; và pixel của policy nằm trong hệ ảnh 384×384 sau resize, phải scale về 640×480**
— chi tiết `../io_system2.md` mục 2.e, 3.d). Nếu `output_action` khác `None` → so với `gt_action`.

**Cách thay thế không cần tự ghép:** đút thẳng vào agent (nó tự điều phối S2/S1/thread):

```python
obs = [{"rgb": rgb, "depth": depth_for_agent, "instruction": instruction}]
result = agent.step(obs)      # xem file 02 mục 6.3 — yêu cầu dựng AgentCfg đúng trước
```

### 4.3. Truyền vào System 1 — `policy.s1_step_latent(rgbs, depths, latent)`

Chỉ chạy được khi 4.2 trả về `output_latent` (xem điều kiện sống còn — file 02 mục 5). Bám đúng
tiền xử lý của agent (`internvla_n1_agent.py:304–336`, file 02 mục 4.2):

```python
import torch

def prep_for_s1(rgb_np, depth_mm_np, device):
    # RGB: resize 224, chuan hoa [0,1]
    rgb224 = np.array(Image.fromarray(rgb_np).resize((224, 224))) / 255.0
    # Depth vln_ce la uint16 MILIMET -> quy ve MET roi clip 5 m.
    # (Agent goc nhan *10 vi depth Habitat da chuan hoa [0,1] -> "should be 0-10m";
    #  voi mm thi phep tuong duong la /1000.)                       # ⬜ chua chay thu — verify!
    d224 = np.array(Image.fromarray(depth_mm_np).resize((224, 224))).astype(np.float32) / 1000.0
    d224[d224 > 5.0] = 5.0
    return rgb224, d224

# Cap 2 frame: (frame luc S2 cham goal, frame hien tai) — theo dung agent
pix_rgb, pix_d = prep_for_s1(rgb_goal_frame, depth_goal_frame, "cuda")
cur_rgb, cur_d = prep_for_s1(rgb_now, depth_now, "cuda")

rgbs   = torch.stack([torch.from_numpy(pix_rgb), torch.from_numpy(cur_rgb)]).unsqueeze(0)              # [1,2,224,224,3]
depths = torch.stack([torch.from_numpy(pix_d),  torch.from_numpy(cur_d)]).unsqueeze(0).unsqueeze(-1)   # [1,2,224,224,1]

s1_out = policy.s1_step_latent(rgbs.to(device), depths.to(device), s2_out.output_latent)
print(s1_out.idx)      # toi da 4 action dau, ma thuoc {1,2,3} — 1=tien 0.25m, 2=trai 15°, 3=phai 15°
```

> **Giải mã output:** cấu trúc đầy đủ của `S1Output` (field nào có giá trị, field nào luôn None),
> bảng mã action kèm độ lớn vật lý, và đường biến đổi latent → 32 quỹ đạo diffusion → trung bình →
> action rời rạc: xem [02_code_structure](02_code_structure.md) **mục 4.3**.

> 🚨 **Điểm dễ sai nhất — đơn vị depth.** Agent gốc viết cho depth Habitat (chuẩn hoá [0,1], nhân 10
> thành mét). Depth `vln_ce` là **uint16 milimét** (PL-D4) → phép quy đổi phải là `/1000`. Dòng này
> ⬜ **chưa được chạy thử** — khi chạy thật, in `depths.min()/max()` và khẳng định nằm trong [0, 5]
> mét trước khi tin quỹ đạo. Ngoài ra depth train của NavDP là D435i 480×270 (PL-D3) còn `vln_ce`
> là 640×480 — khác phân bố, chấp nhận được cho đánh giá định tính, ghi rõ trong báo cáo.

### 4.4. Chấm điểm được gì trên `vln_ce` (và không được gì)

| Chỉ số | Có GT không? | Cách chấm |
|---|---|---|
| Action accuracy của S2 | ✅ cột `action` | so `output_action[0]` với GT từng frame + ma trận nhầm lẫn |
| Pixel-goal L2 của S2 | ✅ cột `goal.{setting}` (frame có goal ≠ `(-1,-1)`) | đảo [row,col]→[u,v], scale 384→640×480, tính L2 |
| Quỹ đạo S1 | ❌ **không có GT liên tục trong `vln_ce`** | chỉ định tính: overlay waypoint lên ảnh, đếm tỉ lệ frame sinh được quỹ đạo hợp lệ (không NaN/rỗng). GT thật nằm ở `vln_n1` (mục 2.3) |
| SR/SPL benchmark | ❌ | cần closed-loop trong simulator — ngoài phạm vi Kaggle |

Kết quả chấm theo bảng trên được vòng eval lưu thành `results.json` — **schema + cách đọc từng
field (kèm bản mẫu thật đã chạy 23/07): file [06](06_eval_plan_w_navdp_kaggle.md) mục G1.b**.

### 4.5. Nếu muốn dùng `vln_n1` / `vln_pe` cho eval

- **`vln_n1` → NavDP standalone:** loader `navdp_lerobot_dataset.py` đọc trực tiếp
  (`camera_intrinsic` 3×3, `camera_extrinsic` 4×4, `action` (N,4,4)); NavDP có đường load `.ckpt`
  riêng qua `navdp_pretrained` (`navdp.py:116–125` — file 02 mục 1.2) không cần checkpoint 16.79 GB.
  ⬜ Chưa chạy thử đường này.
- **`vln_pe` → baseline CMA/RDP:** dùng agent `cma_agent.py`/`rdp_agent.py` + checkpoint
  `VLN-PE/r2r/fine_tuned/{cma,rdp}` (2 thư mục CÓ đủ `config.json` — PL trong file 04 mục 3).
  Nhớ cảnh báo `info.json` sai ở mục 3.3.

---

## 5. Checklist bẫy dữ liệu (tổng hợp — tick trước khi tin kết quả)

- [ ] Không chọn setting `125cm_0deg` (goal toàn `-1` — PL-D4).
- [ ] So pixel: đảo `[row,col]` (policy) ↔ `[u,v]` (data) + scale 384→640×480.
- [ ] Depth: mm → m (`/1000`), clip 5 m; in min/max để kiểm tra (mục 4.3).
- [ ] Thời gian: dùng cột `timestamp`, không suy từ `fps` (fps=30 là giá trị template sai — PL-D2).
- [ ] Schema: tin `df.dtypes` của parquet, không tin `info.json` (đặc biệt `vln_pe` — PL-D2).
- [ ] `vln_n1` thiếu 4 trường LeRobot chuẩn → đừng đọc bằng LeRobot gốc (PL-D3).
- [ ] Action `-1` trong `vln_ce` là frame start — loại khỏi thống kê accuracy.
- [ ] Mỗi scene là một LeRobotDataset độc lập — gộp nhiều scene phải tự quản lý index.
