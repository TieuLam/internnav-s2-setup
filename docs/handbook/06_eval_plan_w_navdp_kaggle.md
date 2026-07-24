# 06 — Kế hoạch: Eval dual-system trên Kaggle với checkpoint `InternVLA-N1-w-NavDP`

> **Loại tài liệu:** runbook — kế hoạch triển khai từng bước, có code snippet chạy được cho từng phase.
> **Ngày viết:** 23/07/2026.
> **Mục tiêu:** chạy **full dual-system (S2 VLM + S1 NavDP) open-loop** trên Kaggle T4×2, dùng
> checkpoint **[`InternRobotics/InternVLA-N1-w-NavDP`](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP)**
> ([04_checkpoint_details](04_checkpoint_details.md) mục 2.1) trên dữ liệu `vln_ce`, xuất metric S2
> (action accuracy / pixel-goal L2) + quỹ đạo S1 (định tính).
> **Nguồn sự thật:** code local InternNav v0.3.1 (`InternNav/code`, commit `7a5c624`) + số đo trong
> [05_appendix](05_appendix.md) (PL-xx) + `requirements/internvla_n1.txt` của chính repo.
> **Nguyên tắc:** mỗi phase có **GATE (cổng kiểm chứng)** — chưa qua gate thì không đi tiếp;
> chỗ chưa chạy thử đánh dấu ⬜.

---

## 0. Tóm tắt quyết định — vì sao đường này, được gì, mất gì

| Câu hỏi | Trả lời |
|---|---|
| Vì sao `-w-NavDP` mà không phải bản đã đóng gói (`tieulam/internvla-n1-ckpt`)? | Config của nó **có sẵn `system1: "navdp_async"`** → S1 được dựng và nạp đúng, **không cần patch config** (PL-B6, PL-C2). Bản đóng gói thiếu field này + là bản wo-dagger điểm thấp hơn (SR 58.2 vs 64.1 — file 04 mục 4). |
| Giá phải trả | (1) Tải mới + đóng gói **16.78 GB** một lần (Phase A). (2) **Không có `tokenizer.json`** → phải pin `transformers==4.51.0` — đúng bản repo pin trong `requirements/internvla_n1.txt` (xem §1.2). |
| Chạy kiểu gì | **Gọi thẳng policy** (`InternVLAN1Net.s2_step` / `s1_step_latent`) trên observation tĩnh từ `vln_ce` — không dùng agent thread, không cần simulator (căn cứ: agent gốc cũng chạy pose = ma trận đơn vị — PL-C5). Luồng function: [02_code_structure](02_code_structure.md) mục 3–5. |
| Sản phẩm cuối | Notebook chạy end-to-end + `results.json` (metric S2) + ảnh visualize quỹ đạo S1 + bản ghi VRAM/thời gian. |
| Ngoài phạm vi | Closed-loop SR/SPL (cần Habitat/Isaac + server ≥24GB); fine-tune; biến thể DualVLN. |

**Sơ đồ tổng thể:**

```
Phase A (CPU, 1 lần)                Phase B (CPU, 1 lần — bỏ qua nếu đã có)
tải 16.78GB w-NavDP                  tải vài scene vln_ce (~100MB)
→ Kaggle Dataset:                    → Kaggle Dataset:
  internvla-n1-w-navdp-ckpt            vln-ce-eval-sample
        │                                   │
        └────────────┬──────────────────────┘ mount (read-only, không ăn quota 20GB)
                     ▼
   Phase C→G (GPU T4×2): cài env → load & VERIFY S1 → smoke S2 →
   smoke S1 → eval loop → metrics + visualize → /kaggle/working/results/
```

---

## 1. Những gì đã biết trước (đọc 5 phút trước khi gõ lệnh)

### 1.1. Đã xác minh — dùng làm nền

| # | Sự thật | Nguồn |
|---|---|---|
| 1 | `-w-NavDP`: 16 file, 4 shard ≈16.78GB, `system1: "navdp_async"`, `n_query: 4`, KHÔNG có `tokenizer.json`/`trainer_state.json` | PL-B6, PL-B4 |
| 2 | Điểm load duy nhất S1+S2: `InternVLAN1Net.__init__` → `from_pretrained` (hardcode flash-attn + pin 1 GPU → **phải patch 2 chỗ**) | PL-C1, PL-C3; file 02 mục 1 |
| 3 | Nhánh dựng S1 đọc `config.system1` — `"navdp_async"` khớp `'navdp' in ...` + `'async' in ...` → `self.navdp` được dựng | PL-C2 (`internvla_n1_arch.py:124–143`) |
| 4 | T4×2 là cấu hình Kaggle duy nhất đủ VRAM; weights S2 đo thật 16.58GB shard đều 2 GPU | PL-B2, PL-B3 |
| 5 | `vln_ce` có RGB (640×480 uint8) + depth (640×480 uint16 **milimét**) + GT `action`/`goal.{setting}`; tránh setting `125cm_0deg` | PL-D4 |
| 6 | Repo pin chính thức cho model này: `transformers==4.51.0`, `accelerate==1.4.0`, `diffusers==0.33.1`, `diffusion_policy` (git), `flash_attn==2.7.4.post1` (**ta bỏ qua flash_attn** — T4) | `requirements/internvla_n1.txt` (đọc local 23/07) |
| 7 | Config mẫu **chính chủ** cho NavDP-async: `scripts/eval/configs/h1_internvla_n1_async_cfg.py` — nguồn số cho `model_settings` ở Phase D (`num_frames: 32`, `num_future_steps: 4`, `continuous_traj: True`, `infer_mode: 'partial_async'`…) | đọc local 23/07 |
| 8 | `s2_step(rgb, depth, pose, instruction, intrinsic, look_down)`: tham số `intrinsic` **không được dùng trong thân hàm** (chỉ có ở chữ ký — grep xác minh 23/07) → giá trị truyền vào không ảnh hưởng S2 | grep `internvla_n1_policy.py` |
| 9 | `navdp.py` import `diffusion_policy` (submodule `third_party/diffusion-policy`) và `diffusers` → thiếu là **ImportError ngay khi import policy** | đọc local 23/07 |
| 10 | `internnav/__init__.py` nhẹ (không kéo habitat/isaac) → có thể `sys.path.insert` thay vì `pip install -e .` — đúng cách `scripts/eval/eval.py` của repo làm (`sys.path.append('./third_party/diffusion-policy')`) | đọc local 23/07 |

### 1.2. Rủi ro mở — gate nào bắt lỗi nào

| # | Rủi ro ⬜ | Gate bắt |
|---|---|---|
| R1 | ~~Slow→fast tokenizer convert lỗi trên 4.51~~ → ✅ **ĐÓNG 23/07:** convert OK, `is_fast=True` (PL-E2) | GATE 1 — pass |
| R2 | ~~`ModelCfg` (pydantic 2) đòi field không có default~~ → ✅ **ĐÓNG 23/07:** dict D1 được nhận (PL-E2) | GATE 1 — pass |
| R3 | `device_map="auto"` đặt `navdp` lệch GPU với latent → ✅ đo 23/07: `navdp` nằm GPU1 **cùng** layer cuối LLM → khả năng cao không xảy ra; giữ `.to(navdp_dev)` làm dây an toàn (PL-E2) | GATE 3 (Phase F) |
| R4 | ~~S2 rơi nhánh action, không có latent~~ → ✅ **ĐÓNG 23/07:** nguyên nhân thật là cơ chế **look-down 2 nhịp** — model trả `↓` (action 5) xin cúi camera, phải gọi `s2_step` lần hai với `look_down=True` mới ra pixel+latent (cell E2). PL-E1 được giải thích trọn vẹn. | GATE 2 — pass |
| R5 | Quy đổi depth mm→m cho S1 (agent gốc viết cho depth Habitat chuẩn hoá [0,1] ×10) | GATE 3 — in min/max depth |
| R6 | Submodule `diffusion-policy` rỗng sau clone (bản clone local của nhóm đang rỗng) | Phase C — verify + fallback pip git |
| R7 | OOM khi generate (KV-cache + navdp) | Phase D/F — đo `max_memory_allocated`, fallback §6 |

---

## Phase A — Đóng gói checkpoint `-w-NavDP` thành Kaggle Dataset (notebook CPU, ~30 phút, 1 lần)

> Vì sao phải đóng gói: mỗi session GPU tải lại 16.78GB là đốt thời gian + phụ thuộc mạng; Kaggle
> Dataset mount read-only **không ăn** quota 20GB của `/kaggle/working`. Đây đúng quy trình đã làm
> với bản cũ (PL-B1 → `tieulam/internvla-n1-ckpt`).

**A1. Notebook mới** — Accelerator **None (CPU)**, Internet **ON**.

**A2. Baseline + môi trường tải** (bẫy cache ghi đúp — PL-A2, PL-A3):

```python
import os
os.makedirs("/kaggle/temp/hf_cache", exist_ok=True)      # /kaggle/temp khong ton tai san (PL-A1)
os.environ["HF_HOME"] = "/kaggle/temp/hf_cache"          # cache KHONG duoc roi vao /kaggle/working
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"              # bien cu HF_HUB_ENABLE_HF_TRANSFER da chet (PL-A3)
```

**A3. Đếm file trước khi tải** (quy tắc PL-A4) + tải + verify:

```python
from huggingface_hub import HfApi, snapshot_download
import os, json

REPO = "InternRobotics/InternVLA-N1-w-NavDP"
files = HfApi().list_repo_files(REPO)
print(len(files), "file")                     # ky vong 16 (PL-B4)
assert "model.safetensors.index.json" in files
assert "tokenizer.json" not in files          # dung nhu khao sat — neu CO thi cang tot, ghi lai

DST = "/kaggle/working/ckpt/internvla-n1-w-navdp"
snapshot_download(REPO, local_dir=DST)        # model repo → KHONG can repo_type

# --- VERIFY GOI ---
cfg = json.load(open(f"{DST}/config.json"))
assert cfg["model_type"] == "internvla_n1",        cfg["model_type"]
assert cfg["system1"]    == "navdp_async",         cfg.get("system1")     # ← linh hon cua ke hoach (PL-B6)
print("n_query =", cfg["n_query"])                                        # ky vong 4

total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(DST) for f in fs)
print(round(total / 1e9, 2), "GB")            # ky vong ~16.78
assert total > 16.5e9, "TAI THIEU FILE — dung lai, dung Save Version"
```

**A4.** Kiểm tra `du -sh /kaggle/working` ≈ 16.8G (không phải ~33G — tức cache không ghi đúp, PL-A2).
Dọn rác nếu có → **Save Version** → tạo Dataset, ví dụ tên `internvla-n1-w-navdp-ckpt`.

**GATE A:** dataset mount thử vào một notebook trống, thấy đủ 16 file + `config.json` đọc được.

---

## Phase B — Đóng gói data `vln_ce` mẫu (notebook CPU, ~15 phút — bỏ qua nếu nhóm đã có)

```python
import os, fnmatch, tarfile
from huggingface_hub import HfApi, hf_hub_download

os.makedirs("/kaggle/temp/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/kaggle/temp/hf_cache"

DATA_REPO = "InternRobotics/InternData-N1"
files = HfApi().list_repo_files(DATA_REPO, repo_type="dataset")

# chon ~3-5 scene r2r + rxr; scene rxr nho nhat de smoke-test
WANT = ["vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz"]          # 16.16 MB (PL-D4)
WANT += fnmatch.filter(files, "vln_ce/traj_data/r2r/*.tar.gz")[:3]
for w in WANT:
    assert w in files, f"khong ton tai: {w}"                 # dem truoc khi tai (PL-A4)

OUT = "/kaggle/working/vln_ce_sample"
for w in WANT:
    tgz = hf_hub_download(DATA_REPO, filename=w, repo_type="dataset",   # THIEU repo_type = 404 (PL-A5)
                          local_dir="/kaggle/temp/dl")
    tarfile.open(tgz).extractall(f"{OUT}/{os.path.basename(w).replace('.tar.gz','')}")
```

**B2. Kiểm tra cấu trúc bằng code — Kaggle không có trình duyệt thư mục tử tế, đây là cách chuẩn.**

> Ghi chú UI: panel bên phải notebook editor (**Data → Output**) có hiện cây `/kaggle/working`
> nhưng load chậm, không duyệt sâu được, và **không hiển thị `/kaggle/temp`**. Đừng dựa vào nó —
> dùng cell dưới đây, vừa "nhìn" được cây vừa assert luôn GATE B.

```python
import os, glob, json
import numpy as np
import pandas as pd
from PIL import Image

OUT = "/kaggle/working/vln_ce_sample"

# --- 1. Nhin nhanh cay thu muc (thay cho file explorer) ---
!find {OUT} -maxdepth 4 -type d | head -30

# --- 2. Dem thanh phan tung scene ---
rgb_files, dep_files = [], []
for scene in sorted(glob.glob(f"{OUT}/*")):
    name  = os.path.basename(scene)
    pq    = glob.glob(f"{scene}/**/*.parquet", recursive=True)
    metas = glob.glob(f"{scene}/**/meta/*.jsonl", recursive=True)
    pngs  = glob.glob(f"{scene}/**/*.png", recursive=True)
    rgbs  = [p for p in pngs if "images.rgb."   in p]
    deps  = [p for p in pngs if "images.depth." in p]
    print(f"{name}:  parquet={len(pq)}  meta={len(metas)}  rgb_png={len(rgbs)}  depth_png={len(deps)}")
    assert pq and metas and rgbs and deps, f"SCENE {name} THIEU thanh phan — dung lai!"
    rgb_files += rgbs; dep_files += deps

# --- 3. GATE B: kiem sau tren 1 scene ---
df = pd.read_parquet(glob.glob(f"{OUT}/*/**/*.parquet", recursive=True)[0])
print(len(df.columns), "cot parquet")                       # ky vong 21 (PL-D4)
assert "action" in df.columns and any(c.startswith("goal.") for c in df.columns)

rgb = np.array(Image.open(rgb_files[0]))
print("RGB:",   rgb.shape, rgb.dtype)                       # ky vong (480, 640, 3) uint8
assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8

dep = np.array(Image.open(dep_files[0]))
print("DEPTH:", dep.shape, dep.dtype, "max:", dep.max())    # ky vong (480, 640) uint16, max<=10000
assert dep.dtype == np.uint16 and dep.max() <= 10000

eps = [json.loads(l) for l in open(glob.glob(f"{OUT}/*/**/meta/episodes.jsonl", recursive=True)[0])]
print("episodes:", len(eps), "| length[0]:", eps[0]["length"])

# --- 4. Chot PATTERN TEN FILE PNG (giai quyet luon dau ⬜ o Phase E) ---
print("\n".join(sorted(os.listdir(os.path.dirname(rgb_files[0])))[:5]))
```

**GATE B:** cell trên chạy hết không assert lỗi (parquet 21 cột; RGB 640×480×3 uint8; depth
640×480 uint16 max ≤ 10000; meta đọc được) — và bạn đã **ghi lại pattern tên file PNG** in ở bước 4
để dùng cho `load_frame` ở Phase E.

**B3. Save Version → tạo Dataset (thao tác UI, theo đúng thứ tự):**

1. Đảm bảo dữ liệu cuối nằm trong **`/kaggle/working`** (KHÔNG phải `/kaggle/temp` — temp bị vứt
   khi commit; file tar.gz tải về đã để ở temp nên không cần dọn).
2. Góc trên phải → **Save Version** → chọn **Save & Run All (Commit)** → **Save**.
   ⚠️ Kaggle sẽ **chạy lại toàn bộ notebook từ đầu trong một session mới** — những gì session
   tương tác hiện tại đã làm tay KHÔNG được mang theo. Vì vậy mọi bước (tải, giải nén, verify)
   phải nằm trong code cell chạy tuần tự được.
3. Chờ version chạy xong (icon chuông/thông báo, hoặc mở tab số version cạnh nút Save).
4. Mở trang notebook ở chế độ **viewer** (không phải editor) → tab/phần **Output** ở cuối trang →
   thấy cây `vln_ce_sample/...`.
5. Trong phần Output đó bấm **New Dataset** → đặt tên `vln-ce-eval-sample` → **Create**.
6. Ở notebook eval (Phase C): **Add Input** → tìm `vln-ce-eval-sample` → bấm **+**. Xác định
   đường dẫn mount thật bằng code (đừng đoán):

   ```python
   !ls /kaggle/input/
   !find /kaggle/input/vln-ce-eval-sample -maxdepth 3 -type d | head
   # → gan gia tri DATA o Phase E theo ket qua nay,
   #   thuong la: /kaggle/input/vln-ce-eval-sample/vln_ce_sample
   ```

**B-lite (phương án tắt, hợp lệ):** data mẫu chỉ ~100–200 MB nên có thể **bỏ hẳn Phase B**, tải +
giải nén trực tiếp trong notebook GPU mỗi session (thêm ~2–3 phút/lần, Internet vốn đã ON). Chỉ
checkpoint 16.78 GB mới bắt buộc đóng gói thành Dataset. Đánh đổi: mỗi session phụ thuộc HF thêm
một lần tải.

---

## Phase C — Notebook GPU: dựng môi trường (T4×2, ~20 phút mỗi session)

**C1. Notebook mới** — Accelerator **GPU T4 ×2** (không P100 — PL-B2), Internet ON, mount 2 dataset
của Phase A + B.

**C2. Cài pin đúng theo `requirements/internvla_n1.txt` của repo — TRỪ flash-attn:**

```python
# Cell 1 — pip (chay dau tien, truoc moi import nang)
!pip install -q "transformers==4.51.0" "accelerate==1.4.0" "diffusers==0.33.1" ftfy
# KHONG cai flash_attn==2.7.4.post1 (T4 sm_75 khong chay duoc — se patch code sang sdpa o C4)
# KHONG can pip install -e . : duong policy-truc-tiep chi can sys.path (giong scripts/eval/eval.py)
```

> 🚨 Kaggle preinstall `transformers 5.0.0` → lệnh trên **hạ cấp** xuống 4.51.0. Bắt buộc
> **Restart kernel** (Run → Restart & clear cell outputs) sau cell này, rồi chạy cell verify:

```python
import transformers, torch
print(transformers.__version__)   # PHAI la 4.51.0
print(torch.__version__, torch.cuda.device_count())   # cuda ban, 2 GPU
```

**C3. Clone repo + verify submodule (R6):**

```python
%cd /kaggle/working
!git clone --recursive https://github.com/InternRobotics/InternNav.git

import os
DP = "/kaggle/working/InternNav/third_party/diffusion-policy/diffusion_policy"
if not os.path.isdir(DP):                    # submodule rong → fallback pip theo pin cua repo
    !pip install -q "diffusion_policy @ git+https://github.com/real-stanford/diffusion_policy.git@5ba07ac6661db573af695b419a7947ecb704690f"
else:
    print("submodule OK")
```

**C4. Patch 2 dòng load (căn cứ PL-C1, PL-C3 — file 02 mục 1.1):**

```python
import pathlib
p = pathlib.Path("/kaggle/working/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py")
src = p.read_text(encoding="utf-8")
assert 'attn_implementation="flash_attention_2"' in src and 'device_map={"": self.model_config.device}' in src, \
    "Code repo da doi — doc lai truoc khi patch mu"
src = src.replace('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')       # T4 khong co flash-attn
src = src.replace('device_map={"": self.model_config.device}', 'device_map="auto"')              # khong pin 1 GPU (PL-C3)
p.write_text(src, encoding="utf-8")
print("PATCHED")
```

**GATE C:** import không lỗi:

```python
import sys
sys.path.insert(0, "/kaggle/working/InternNav")
sys.path.insert(0, "/kaggle/working/InternNav/third_party/diffusion-policy")
from internnav.model import get_policy, get_config          # keo theo diffusers + diffusion_policy (R6)
print("import OK")
```

**C5. Tải backbone DepthAnything V2 ViT-S — bắt buộc, thiếu là Phase D crash (✅ đã gặp thật 23/07)**

> Khi dựng kiến trúc NavDP, constructor của `DAT_RGBD_Patch_Backbone` load ngay một file
> **hardcode đường dẫn tương đối** `checkpoints/depth_anything_v2_vits.pth`
> (`internnav/model/encoder/navdp_backbone.py:109` + `:124`, được `navdp.py:53–55` gọi mà không
> truyền path khác — PL-C7). Thiếu file → `FileNotFoundError` ngay trong `PolicyCls(...)` ở D3.
> **Weights file này không ảnh hưởng kết quả** — sau khi dựng, `from_pretrained` ghi đè toàn bộ
> `navdp.rgbd_encoder.rgb_model.*` bằng weights trong checkpoint 16.78GB (nhóm tensor này có mặt
> trong checkpoint — PL-B3). Nó chỉ cần tồn tại để constructor chạy qua.

```python
import os
from huggingface_hub import hf_hub_download

os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
hf_hub_download(
    "depth-anything/Depth-Anything-V2-Small",     # repo chinh chu, public, apache-2.0 (xac minh 23/07)
    "depth_anything_v2_vits.pth",
    local_dir="/kaggle/working/checkpoints",
)
p = "/kaggle/working/checkpoints/depth_anything_v2_vits.pth"
print(round(os.path.getsize(p) / 1e6, 1), "MB")   # ~99 MB
assert os.path.getsize(p) > 5e7

os.chdir("/kaggle/working")                        # duong dan hardcode la TUONG DOI theo CWD
print("CWD:", os.getcwd())
```

> Ghi chú: file họ hàng `depth_anything_v2_metric_hypersim_vits.pth`
> (`internvla_n1_arch.py:36`) **không cần** cho đường này — chỉ nhánh `nextdit_async` (DualVLN) dùng.

---

## Phase D — Load full dual-system + GATE 1 (điểm quyết định số 1)

**D1. Dựng `model_settings`** — số liệu chép từ config chính chủ NavDP-async
`scripts/eval/configs/h1_internvla_n1_async_cfg.py` (mục 1.1 #7), thêm 2 field pydantic bắt buộc (R2):

```python
CKPT = "/kaggle/input/internvla-n1-w-navdp-ckpt/ckpt/internvla-n1-w-navdp"   # sua theo duong dan mount that

model_settings = {
    # -- 2 field ModelCfg khong co default (pydantic v2) — PHAI co mat (R2) --
    "policy_name": "InternVLAN1_Policy",          # ten dung trong get_policy (PL-C6)
    "state_encoder": None,
    # -- chep tu h1_internvla_n1_async_cfg.py (config chinh chu duong navdp_async) --
    "model_path": CKPT,
    "device": "cuda:0",                            # sau patch C4 chi con vai tro phu
    "camera_intrinsic": [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
    "width": 640, "height": 480, "hfov": 79,
    "resize_w": 384, "resize_h": 384,
    "max_new_tokens": 1024,
    "num_frames": 32, "num_history": 8, "num_future_steps": 4,
    "predict_step_nums": 32, "continuous_traj": True,
    "infer_mode": "partial_async",
    "vis_debug": False, "vis_debug_path": "./vis_debug",
}
```

**D2. Kiểm tra config checkpoint TRƯỚC khi load** (30 giây, tránh mất 10 phút load rồi mới biết sai):

```python
import json
cfg = json.load(open(f"{CKPT}/config.json"))
assert cfg.get("system1") == "navdp_async", f"config sai/thieu system1: {cfg.get('system1')}"
```

**D3. Load (~5–10 phút):**

```python
import torch
PolicyCls = get_policy("InternVLAN1_Policy")
PolicyCfg = get_config("InternVLAN1_Policy")
policy = PolicyCls(config=PolicyCfg(model_cfg={"model": model_settings}))
policy.eval()
```

> ℹ️ **Warning kỳ vọng khi load (✅ gặp thật 23/07):** hàng loạt dòng
> `UserWarning: for pretrained.cls_token: copying from a non-meta parameter in the checkpoint to a
> meta parameter in the current model, which is a no-op...` — **vô hại, bỏ qua.**
> Nguyên nhân: `device_map="auto"` dựng model trên **meta device** (khung rỗng, chưa cấp phát bộ
> nhớ); constructor NavDP lại tự `load_state_dict` file DepthAnything (C5) ngay lúc khung còn rỗng
> (`navdp_backbone.py:124` — PL-C7) → copy vào tensor meta = no-op, mỗi tham số một dòng warning.
> Weights DA đó đằng nào cũng bị checkpoint 16.78GB ghi đè (PL-B3/PL-C7). **KHÔNG** làm theo gợi ý
> `assign=True` của warning. Điều kiện để thật sự yên tâm: phép kiểm (1b) meta-check ở GATE 1 dưới đây.

**D4. GATE 1 — 5 phép kiểm bắt buộc:** *(✅ **ĐÃ PASS toàn bộ 23/07/2026** — số đo thật ở PL-E2:
navdp 98.8M · meta sót 0 · latent_queries (1,4,3584) · tokenizer fast OK · GPU 7.57+9.21 GB)*

```python
m = policy.model

# (1) S1 da duoc nap that su (PL-C2)
n_navdp = sum(p.numel() for n, p in m.named_parameters() if "navdp" in n)
print(f"navdp params: {n_navdp/1e6:.1f}M");  assert n_navdp > 0, "S1 KHONG duoc nap!"

# (1b) khong con tham so meta sot lai (canh bao no-op luc dung la vo hai CHI KHI dieu nay dung)
meta_left = [n for n, p in m.named_parameters() if p.is_meta]
print("meta con sot:", len(meta_left), meta_left[:5]);  assert not meta_left, "Co tham so chua duoc nap!"

# (1c) navdp weights la so thuc (checkpoint da ghi de len phan khoi tao meta/no-op)
t = next(p for n, p in m.named_parameters() if "navdp" in n and "rgb_model" in n)
print("navdp sample:", t.device, t.dtype, float(t.abs().mean()))   # mean > 0 la co weights that

# (2) latent_queries dung shape n_query=4 (PL-B6)
lq = m.get_model().latent_queries
print("latent_queries:", tuple(lq.shape));   assert lq.shape[1] == cfg["n_query"] == 4

# (3) tokenizer: convert slow→fast thanh cong (R1)
print("tokenizer fast:", policy.tokenizer.is_fast)
ids = policy.tokenizer("test forward ↑", return_tensors="pt").input_ids
print("encode OK:", ids.shape)

# (4) model chia deu 2 GPU (PL-B3)
print(getattr(m, "hf_device_map", "KHONG co device map — kiem tra patch C4!"))

# (5) VRAM con du cho generate
for i in range(torch.cuda.device_count()):
    print(f"GPU{i}: {torch.cuda.max_memory_allocated(i)/1e9:.2f} GB")   # tong ky vong ~16.6-17
```

**Xử lý khi fail:**
- (1) fail → in `cfg.get("system1")`; nếu đúng `navdp_async` mà vẫn 0 → soi log load tìm
  `MISSING/UNEXPECTED` chứa `navdp` → đối chiếu PL-C2, báo lại nhóm (đây là phát hiện mới).
- (3) fail (R1) → fallback: chép `tokenizer.json` từ bản mirror wo-dagger
  (`/kaggle/input/internvla-n1-ckpt/.../tokenizer.json`) sang thư mục CKPT **sau khi so sánh
  `added_tokens.json` hai bản giống hệt nhau** (cùng họ Qwen2.5-VL — nhưng phải kiểm, đừng tin):
  ```python
  import filecmp, shutil
  OLD = "/kaggle/input/internvla-n1-ckpt/..."      # mirror wo-dagger
  assert filecmp.cmp(f"{OLD}/added_tokens.json", f"{CKPT}/added_tokens.json", shallow=False)
  # CKPT mount read-only → copy sang /kaggle/working roi tro model_path vao ban copy
  ```
- (5) sát trần → giảm `num_history` (8→4) trước khi nghĩ tới 4-bit (§6).

---

## Phase E — Smoke test System 2 trên 1 episode + GATE 2 (điểm quyết định số 2)

**E1. Nạp 1 episode `vln_ce`** (schema: [03_data_contract](03_data_contract.md) mục 4.1):

```python
import glob, json
import numpy as np
import pandas as pd
from PIL import Image

DATA = "/kaggle/input/vln-ce-eval-sample"       # sua theo mount that
SETTING = "60cm_30deg"                           # TRANH 125cm_0deg (PL-D4)

scene   = sorted(glob.glob(f"{DATA}/*"))[0]
parquet = sorted(glob.glob(f"{scene}/**/data/**/*.parquet", recursive=True))[0]
df      = pd.read_parquet(parquet)

eps  = [json.loads(l) for l in open(glob.glob(f"{scene}/**/meta/episodes.jsonl", recursive=True)[0])]
instruction = eps[0]["tasks"][0].split("<INSTRUCTION_SEP>")[0]
print("instruction:", instruction)

rgb_dir   = glob.glob(f"{scene}/**/observation.images.rgb.{SETTING}",   recursive=True)[0]
depth_dir = glob.glob(f"{scene}/**/observation.images.depth.{SETTING}", recursive=True)[0]

def load_frame(ep, fr):
    # ⬜ pattern ten file: xac nhan bang os.listdir truoc khi tin (03 muc 4.1)
    rgb   = np.array(Image.open(f"{rgb_dir}/episode_{ep:06d}_{fr}.png"))
    depth = np.array(Image.open(f"{depth_dir}/episode_{ep:06d}_{fr}.png"))   # uint16, milimet
    return rgb, depth

INTR = np.array(model_settings["camera_intrinsic"])   # s2_step KHONG dung intrinsic trong than ham (muc 1.1 #8)
POSE = np.eye(4)                                       # agent goc cung truyen ma tran don vi (PL-C5)
```

**E2. Chạy S2 hai nhịp mỗi frame — BẮT BUỘC xử lý cơ chế "look down"** *(✅ bản này đã chạy được
23/07; bản một-nhịp cũ cho toàn `action [5]` không latent — xem cơ chế ngay dưới)*:

> 🔑 **Cơ chế look-down (phát hiện 23/07, gặp thật khi chạy):** S2 thường KHÔNG nhả pixel-goal ngay.
> Nó trả `llm_output = "↓"` → `parse_actions` dịch thành **action 5 = LOOK DOWN** — model *xin cúi
> camera xuống* trước khi chọn điểm đến. Agent gốc xử lý riêng action 5
> (`internvla_n1_agent.py:287–292`): đặt `look_down=True` rồi **gọi lại `s2_step` lần hai**; ở lượt
> hai `s2_step` nối tiếp hội thoại thay vì xoá (`internvla_n1_policy.py:139–146`) và lúc đó model
> mới nhả toạ độ pixel → nhánh `generate_latents` chạy (`:186–194`) → **có latent**.
> Với setting `60cm_30deg`, config train chính chủ là `r2r_60cm_30_30` (file 04 mục 5.1) — cặp
> `30_30` = góc nhìn thường và góc cúi **cùng 30°** → pass 2 dùng lại chính ảnh frame đó là trung
> thực với cách model được train.

```python
policy.reset()                                   # xoa history giua cac episode — BAT BUOC
ep_len = int(eps[0]["length"])
records = []
for fr in range(min(ep_len, 6)):                 # smoke: 6 frame dau
    rgb, depth = load_frame(0, fr)

    # PASS 1 — nhin thuong
    out = policy.s2_step(rgb, depth, POSE, instruction, INTR, look_down=False)
    tag = "normal  "

    # PASS 2 — model xin cui camera (llm '↓' → action 5): goi lai voi look_down=True
    # (mo phong agent goc :287–292; chi retry 1 lan, khong lap vo han)
    if out.output_action and out.output_action[0] == 5:
        out = policy.s2_step(rgb, depth, POSE, instruction, INTR, look_down=True)
        tag = "lookdown"

    records.append(out)
    print(fr, tag,
          "| pixel:",  out.output_pixel,
          "| action:", out.output_action,
          "| latent:", None if out.output_latent is None else tuple(out.output_latent.shape),
          "| llm:",    repr(policy.llm_output))
```

**GATE 2 — đọc kết quả:** *(✅ **ĐÃ PASS 23/07/2026** — với cell 2 nhịp ở trên, S2 trả
`output_pixel` + `output_latent`; nghi vấn PL-E1 chính thức đóng: hôm 22/07 không ra pixel vì
thiếu đúng cú look-down hai nhịp + full policy, không phải model hỏng)*

| Quan sát | Nghĩa là | Đi tiếp thế nào |
|---|---|---|
| Có frame ra `output_pixel` + `output_latent` (thường ở pass 2) | ✅ nhánh pixel-goal hoạt động — **kết quả thực tế 23/07** | → Phase F |
| Pass 1 ra `↓` / action `[5]` | **KHÔNG phải lỗi** — model xin cúi camera, cơ chế 2 nhịp ở trên | Cell E2 đã tự xử lý (pass 2) |
| Pass 2 **vẫn** ra `↓` | Model chưa chịu chốt goal trên ảnh này | Debug E3 ↓ |
| Ra action điều hướng thật (`←→↑`) | S2 chọn hành động rời rạc cho frame đó — hợp lệ nếu GT cũng là action | Đối chiếu GT (E3 bước 2) |
| `output_action == []` hoặc text lạ | parser rơi lỗ hổng đã ghi (`../io_system2.md` 3.d) | In `policy.llm_output` thô từng frame, ghi lại làm bằng chứng |

**E3. Debug khi pass 2 vẫn không ra pixel** — thử theo thứ tự rẻ → đắt, ghi kết quả từng bước:
1. In `os.listdir(os.path.dirname(rgb_dir))` xem scene có setting góc cúi sâu hơn không
   (vd `*_45deg`) → dùng ảnh setting đó làm ảnh pass 2 (mô phỏng cú cúi thật hơn).
2. Đối chiếu: frame nào GT `goal.{SETTING}` ≠ `(-1,-1)` mà model vẫn không ra pixel → đó mới là
   bất thường thật; frame GT không có goal thì model ra action là **đúng hành vi**.
3. Chạy **nhiều frame hơn / episode khác / scene r2r**, thử setting khác (`125cm_30deg`) +
   instruction khác.
4. Vẫn 100% action trên các frame có GT goal → dừng, ghi nhận trung thực làm kết quả báo cáo.

---

## Phase F — Smoke test System 1 + GATE 3

Chạy khi GATE 2 cho ít nhất 1 frame có `output_latent`. Tiền xử lý bám agent gốc
(`internvla_n1_agent.py:304–336` — file 02 mục 4.2), **đổi đúng một chỗ: depth mm → m** (R5):

```python
import torch

# thiet bi cua navdp (device_map="auto" co the dat navdp o GPU khac — R3)
navdp_dev = next(p.device for n, p in policy.model.named_parameters() if "navdp" in n)
print("navdp device:", navdp_dev)

def prep_s1(rgb_np, depth_mm):
    r = np.array(Image.fromarray(rgb_np).resize((224, 224))) / 255.0
    d = np.array(Image.fromarray(depth_mm).resize((224, 224))).astype(np.float32) / 1000.0   # mm -> MET (R5 ⬜)
    d = np.clip(d, 0.0, 5.0)                     # sys1_depth_threshold = 5.0 (agent dong 59)
    return r, d

# cap 2 frame theo agent: (frame S2 cham goal, frame hien tai)
GOAL_FR = 3                                       # frame co latent o Phase E — sua theo thuc te
rgb_g, dep_g = load_frame(0, GOAL_FR)
rgb_c, dep_c = load_frame(0, GOAL_FR)             # smoke: dung cung frame; loop that se la frame hien tai

pr, pd_ = prep_s1(rgb_g, dep_g)
cr, cd_ = prep_s1(rgb_c, dep_c)
rgbs   = torch.stack([torch.from_numpy(pr), torch.from_numpy(cr)]).unsqueeze(0).to(navdp_dev)              # [1,2,224,224,3]
depths = torch.stack([torch.from_numpy(pd_), torch.from_numpy(cd_)]).unsqueeze(0).unsqueeze(-1).to(navdp_dev)  # [1,2,224,224,1]
print("depth range (m):", depths.min().item(), depths.max().item())    # GATE: phai nam trong [0, 5]

latent = records[GOAL_FR].output_latent
if latent.device != navdp_dev:
    latent = latent.to(navdp_dev)                 # fix R3 neu lech GPU

# --- s1_step_full: nhu s1_step_latent cua repo NHUNG dien ca trajectory ---
# Repo tinh quy dao trung binh roi VUT (file 02 muc 4.3e). Ta chay generate_traj MOT lan,
# suy ra ca idx lan waypoints tu cung dp_actions.
# .clone() la bat buoc: traj_to_actions un-normalize IN-PLACE (chia 4 vao tensor) —
# moi tensor chi duoc di qua no dung 1 lan.
from internnav.model.utils.vln_utils import traj_to_actions, S1Output

def s1_step_full(rgbs, depths, latent):
    with torch.no_grad():
        dp_actions = policy.model.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths)
    waypoints = traj_to_actions(dp_actions.clone(), use_discrate_action=False)  # (T+1, 2) met, xuat phat (0,0)
    actions   = traj_to_actions(dp_actions,         use_discrate_action=True)   # giong nhanh continuous_traj cua repo
    actions   = [x for x in actions if x != 0][:4]      # loc stop + cat 4 — y het internvla_n1_policy.py:212-214
    return S1Output(idx=actions, trajectory=waypoints)

s1_out = s1_step_full(rgbs, depths, latent)
print("S1 actions:",    s1_out.idx)               # ky vong list 1-4 phan tu thuoc {1,2,3}
print("S1 trajectory:", s1_out.trajectory.shape,  # ky vong (T+1, 2)
      "| NaN:", bool(np.isnan(s1_out.trajectory).any()))
# (1=tien 0.25m, 2=trai 15°, 3=phai 15°; ma 0=stop bi loc san; ma 5 KHONG bao gio co tu S1 —
#  no la chuyen rieng cua S2/look-down. Giai ma day du: file 02 muc 4.3)
```

**GATE 3:**
- `s1_out.idx` là list hợp lệ (không rỗng, không NaN) → pipeline dual-system **thông**.
- `s1_out.trajectory` shape (T+1, 2), không NaN, waypoint đầu = (0,0) — quỹ đạo mét trong hệ robot.
- Lỗi dtype (bf16 vs float32) → thử `rgbs = rgbs.to(torch.bfloat16)` (và depths tương tự) — ghi lại
  bản nào chạy. ⬜ chưa xác minh trước dtype nào đúng.
- Lỗi device → đã có fix `.to(navdp_dev)` ở trên; nếu vẫn lỗi trong `generate_traj`, in device từng
  input và báo lại (ứng viên bug report).

**F2 (tuỳ chọn — lấy cả 32 ứng viên thô để vẽ "quạt" quỹ đạo):** quỹ đạo **trung bình** đã nằm sẵn
trong `s1_out.trajectory` (hàm `s1_step_full` ở trên) — F2 chỉ cần khi muốn visualize độ phân tán
của 32 mẫu diffusion (định tính độ "tự tin" của S1):

```python
with torch.no_grad():
    trajs = policy.model.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths)
print(trajs.shape)         # (32, 32, 3) — 32 ung vien × 32 buoc (dx,dy,dyaw); predict_size=32 ✅ do 23/07

trajs[:, :, :2] /= 4.0                                   # un-normalize xy (nhu vln_utils.py:129)
fan = np.cumsum(trajs[:, :, :2].float().cpu().numpy(), axis=1)   # (32, T, 2) — 32 duong xy de ve
# ve: 32 duong mo + s1_out.trajectory dam — thay duoc do phan tan quanh quy dao trung binh
```

---

## Phase G — Eval loop + metric + xuất kết quả

**G1. Vòng lặp chuẩn** (mô phỏng nhịp `partial_async`: S2 mỗi frame để lấy metric per-frame;
S1 chỉ chạy ở frame có latent):

```python
import time, os
RESULTS = {"setting": SETTING, "ckpt": "InternVLA-N1-w-NavDP", "frames": []}
os.makedirs("/kaggle/working/results", exist_ok=True)

for ep_i, ep in enumerate(eps):
    policy.reset()
    ep_len = int(ep["length"])
    instr  = ep["tasks"][0].split("<INSTRUCTION_SEP>")[0]
    gt_act  = df[df.episode_index == ep_i]["action"].tolist()
    gt_goal = df[df.episode_index == ep_i][f"goal.{SETTING}"].tolist()

    for fr in range(ep_len):
        rgb, depth = load_frame(ep_i, fr)
        t0 = time.time()
        out = policy.s2_step(rgb, depth, POSE, instr, INTR, False)
        looked_down = False
        if out.output_action and out.output_action[0] == 5:   # co che look-down 2 nhip (nhu E2)
            out = policy.s2_step(rgb, depth, POSE, instr, INTR, True)
            looked_down = True
        rec = {"ep": ep_i, "fr": fr, "t_s2": round(time.time() - t0, 2),
               "looked_down": looked_down,
               "gt_action": int(gt_act[fr]),
               "gt_goal": [int(x) for x in np.ravel(gt_goal[fr])],
               "pred_action": None if out.output_action is None else [int(a) for a in out.output_action],
               "pred_pixel_rowcol": None if out.output_pixel is None else [int(x) for x in out.output_pixel],
               "llm_output": policy.llm_output}
        if out.output_latent is not None:            # S1 chi chay khi co latent
            # ... prep nhu Phase F (frame goal = frame nay, frame hien tai = frame nay) ...
            s1 = s1_step_full(rgbs, depths, out.output_latent.to(navdp_dev))   # ham dinh nghia o Phase F
            rec["s1_actions"] = [int(a) for a in s1.idx]
            rec["s1_traj"]    = np.round(s1.trajectory, 3).tolist()            # (T+1, 2) met — cho G3 ve
        RESULTS["frames"].append(rec)
        if fr % 8 == 0:
            torch.cuda.empty_cache()                 # quan ly VRAM tren loop dai

json.dump(RESULTS, open("/kaggle/working/results/results.json", "w"), indent=1)
```

**G1.b — Cách đọc `results.json`** (schema đầu ra — bản mẫu thật: `docs/handbook/results.json`):

Cấp cao nhất có 3 key: `setting` (camera setting của run — quyết định cột GT `goal.{setting}` nào
được so), `ckpt` (checkpoint đã dùng), `frames` (list record — **mỗi phần tử = 1 frame đã qua S2**,
và qua cả S1 nếu frame đó có latent).

Từng field trong một record của `frames`:

| Field | Kiểu | Ý nghĩa & cách đọc |
|---|---|---|
| `ep` | int | Chỉ số episode trong scene (khớp thứ tự `meta/episodes.jsonl` và cột `episode_index` parquet — file 03 mục 1) |
| `fr` | int | Chỉ số frame trong episode, đếm từ 0 |
| `t_s2` | float (giây) | Thời gian S2 xử lý frame — **gồm cả 2 nhịp** nếu `looked_down` (đo thật: mean 35.5s — PL-E3) |
| `looked_down` | bool | `true` = nhịp 1 model trả `↓` (xin cúi camera) và record này là kết quả **nhịp 2** (cơ chế Phase E) |
| `gt_action` | int | GT từ cột `action` parquet; **`-1` = frame đầu không có GT → loại khỏi accuracy** |
| `gt_goal` | [int, int] | GT từ cột `goal.{setting}`, thứ tự **[u, v]** (u ngang 0–639, v dọc 0–479); **`[-1,-1]` = frame không có goal** → loại khỏi L2 |
| `pred_action` | list \| null | Chuỗi action khi S2 rơi nhánh action — mã {0,1,2,3}: 0=STOP, 1=tiến, 2=trái, 3=phải; `null` khi frame ra pixel. Metric chỉ so phần tử `[0]` |
| `pred_pixel_rowcol` | [int, int] \| null | Pixel goal, **lưu đảo [row, col] = [v, u]** — muốn so với `gt_goal` phải đảo lại (`u`=phần tử thứ 2, `v`=phần tử thứ 1); cùng không gian 640×480 với GT, **không scale** (PL-E3) |
| `llm_output` | str | Text thô model sinh ở **nhịp cuối** của frame (nếu `looked_down` → đây là output nhịp 2); `"270 173"` đọc là `"u v"` |
| `s1_actions` | list | **Chỉ có khi frame ra latent.** ≤4 mã ∈ {1,2,3}: 1=tiến 0.25 m, 2=trái 15°, 3=phải 15° (bảng mã: file 02 mục 4.3) |
| `s1_traj` | list (33, 2) | **Chỉ có khi frame ra latent.** 33 waypoint `[x, y]` **mét, hệ robot**: gốc (0,0) = vị trí hiện tại, **+x = hướng nhìn, +y = bên trái**; 33 = predict_size 32 + điểm gốc (PL-E3) |

**Đọc thử record đầu tiên của bản mẫu** (ep0/fr0): `looked_down: true` → model phải cúi mới chốt
goal; `llm_output "270 173"` → goal tại (u,v)=(270,173); `gt_goal [500,93]` → L2 ≈ 244 px (outlier
— frame 0 chưa có history, đúng thiết kế); `s1_traj` có y trôi dần về −0.34 (lệch **phải**) — khớp
`s1_actions [1,1,3,1]` (tiến, tiến, rẽ phải, tiến); `gt_action: -1` → frame này không tính accuracy.

Quy tắc suy metric từ record (đúng logic G2 bên dưới): **accuracy** đếm trên frame có
`gt_action ≠ -1` VÀ `pred_action ≠ null`; **L2** đếm trên frame có `pred_pixel_rowcol ≠ null` VÀ
`gt_goal ≠ [-1,-1]` — hai tập frame này **không giao nhau** (mỗi frame chỉ ra 1 trong 2 nhánh).

**G2. Metric S2** (đúng giới hạn đã ghi ở [03](03_data_contract.md) mục 4.4) — *✅ chạy thật 23/07
trên 3 episode/128 frame (PL-E3): pixel L2 **mean 41.8 / median 20.0 px** (n=98, công thức đã sửa
dưới đây); action acc 37.04% (n=27) — nhưng 10/17 lỗi là "STOP sớm" dồn ở cuối episode, chỉ 7 lỗi
điều hướng thật*:

```python
frames = RESULTS["frames"]

# (a) Action accuracy — loai frame GT = -1 (frame start)
pairs = [(f["gt_action"], f["pred_action"][0]) for f in frames
         if f["gt_action"] != -1 and f["pred_action"]]
acc = sum(g == p for g, p in pairs) / max(len(pairs), 1)
print(f"action acc: {acc:.2%}  ({len(pairs)} frame)")

# (b) Pixel-goal L2 — model nha toa do "u v" TRUC TIEP trong khong gian 640x480 goc, KHONG scale!
# Chung minh bang 98 frame that 23/07 (PL-E3): 20 frame co toa do >384 (max 563) — khong the la
# khong gian 384. Ban dau G2 nham scale 384->640 → mean bi thoi tu 41.8 len 206 px.
l2s = []
for f in frames:
    if f["pred_pixel_rowcol"] and f["gt_goal"] != [-1, -1]:
        row, col = f["pred_pixel_rowcol"]                # [row,col] = [v,u] — chi bi DAO thu tu khi luu
        u, v = col, row                                  # dao lai la xong — cung he 640x480 voi GT
        gu, gv = f["gt_goal"]
        l2s.append(((u - gu) ** 2 + (v - gv) ** 2) ** 0.5)
print(f"pixel L2 (px): mean={np.mean(l2s):.1f}  median={np.median(l2s):.1f}  n={len(l2s)}"
      if l2s else "khong co frame pixel nao")
```

**G3. Visualize** — vẽ `pred_pixel` (chấm xanh) lên RGB gốc; vẽ bird-eye quỹ đạo S1 từ
`rec["s1_traj"]` đã lưu trong `results.json` (xy mét, xuất phát (0,0) — trục x là hướng nhìn robot);
nếu chạy thêm F2 thì overlay "quạt" 32 ứng viên quanh quỹ đạo trung bình
→ lưu `/kaggle/working/results/vis/*.png` (≥3 ảnh cho báo cáo).

**G4. Ghi bản ghi phần cứng** vào `summary.json`: VRAM đỉnh mỗi GPU, giây/frame S2 và S1, version
(`transformers`, commit repo) — số này dùng cho mail xin server.

---

## §4. Definition of Done

- [x] Kaggle Dataset `internvla-n1-w-navdp-ckpt` tồn tại, verify đủ 16 file + `system1: navdp_async` (GATE A).
- [x] Notebook GPU qua **GATE 1**: navdp 98.8M, meta sót 0, latent_queries (1,4,3584), tokenizer fast OK, chia 2 GPU 7.57+9.21 GB (✅ 23/07 — PL-E2).
- [x] **GATE 2** có kết luận rõ: ✅ 23/07 — pixel-goal + latent bật được qua cơ chế look-down 2 nhịp (cell E2); PL-E1 đóng.
- [x] **GATE 3**: ✅ 23/07 — S1 chạy trên 101 frame latent, `s1_actions` ∈ {1,2,3}, `s1_traj`
      (33, 2) không NaN, `predict_size=32` (PL-E3). ⬜ riêng bản in depth-range chưa lưu lại.
- [ ] `results.json` ✅ (128 frame, 3 ep — bản sao tại `docs/handbook/results.json`); ⬜ còn
      `summary.json` (G4) + ≥3 ảnh visualize (G3).
- [ ] Ghi chú giới hạn trong notebook: open-loop ≠ benchmark SR/SPL; S1 không có GT trong `vln_ce`; 1 setting camera.

## §5. Ngân sách thời gian & quota (ước lượng, chưa đo ⬜)

| Phase | Session | Thời gian | Ghi chú |
|---|---|---|---|
| A | CPU | ~30 ph | tải 16.78GB (bản cũ đo được 1m51s tốc độ tải — PL trong `SETUP_NOTES.md`; + Save Version nén lâu) |
| B | CPU | ~15 ph | bỏ qua nếu dataset đã có |
| C+D | GPU T4×2 | ~30–40 ph | pip + restart + clone + load ~10 ph |
| E+F | GPU T4×2 | ~30 ph | smoke + debug |
| G | GPU T4×2 | 1–3 h | tuỳ số episode; đo giây/frame ở E để tính |

## §6. Bảng rủi ro & phương án lùi

| # | Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|---|
| 1 | Hạ cấp transformers kéo xung đột torch | pip báo conflict / import lỗi sau restart | thử `transformers==4.51.3`; ghi lại; tuyệt đối không quay về 5.0.0 cho bản không có fast tokenizer |
| 2 | Tokenizer convert fail (R1) | lỗi ở `AutoTokenizer(..., use_fast=True)` | fallback chép `tokenizer.json` từ mirror wo-dagger sau khi so `added_tokens.json` (Phase D4) |
| 3 | S1 không nạp dù config đúng | GATE 1 (1) fail | soi log MISSING/UNEXPECTED; đối chiếu PL-C2; báo nhóm — phát hiện mới |
| 4 | OOM khi generate | kernel chết im (swap=0 — PL-A1) | giảm `num_history` 8→4; `empty_cache()` mỗi 8 frame; phương án cuối: 4-bit (đổi chất lượng) hoặc server ≥24GB |
| 5 | ✅ **đã gặp & hoá giải:** S2 không bật pixel-goal (R4) | GATE 2 toàn `action [5]` (llm ra `↓`) | Cơ chế look-down 2 nhịp — gọi lại `s2_step` với `look_down=True` (cell E2); nếu pass 2 vẫn `↓` → quy trình E3 |
| 6 | Device/dtype mismatch S1 (R3) | RuntimeError trong `generate_traj` | `.to(navdp_dev)` cho latent/inputs; thử bf16; ghi lại tổ hợp chạy được |
| 7 | Submodule diffusion-policy rỗng (R6) | ImportError `diffusion_policy` | pip git pin (C3) |
| 8 | Pattern tên PNG khác dự đoán | FileNotFoundError trong `load_frame` | `os.listdir` in 5 tên đầu, sửa f-string — đã đánh dấu ⬜ ở E1 |
| 9 | Tràn 20GB working ở Phase A | Errno 28 | HF_HOME đã trỏ `/kaggle/temp` (A2); không giải nén gì thêm vào working |
| 10 | ✅ **đã gặp:** thiếu `checkpoints/depth_anything_v2_vits.pth` (hardcode trong `navdp_backbone.py:109`) | `FileNotFoundError` ngay khi dựng policy ở D3 | Bước C5: tải từ `depth-anything/Depth-Anything-V2-Small` + `os.chdir("/kaggle/working")`; weights bị checkpoint ghi đè nên không ảnh hưởng kết quả (PL-C7) |

## §7. Sau khi xong — cập nhật tài liệu

1. Điền kết quả GATE 1–3 + số đo (VRAM, s/frame, acc, L2) vào [05_appendix](05_appendix.md)
   (thêm mục PL-E2 "chạy thật w-NavDP") — các mục ⬜ trong kế hoạch này chuyển thành ✅/❌.
2. Cập nhật [04_checkpoint_details](04_checkpoint_details.md) mục 2.1 (bỏ dòng "chưa tải/chạy bản này")
   và mục 6 câu hỏi mở #2.
3. Nếu GATE 2 bật được pixel-goal → đóng luôn nghi vấn PL-E1 (nghi phạm số 3 "cần full policy" được
   xác nhận/bác bỏ).
