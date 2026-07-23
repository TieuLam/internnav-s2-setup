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
| R1 | Slow→fast tokenizer convert lỗi trên 4.51 (không có `tokenizer.json`) | GATE 1 (Phase D) |
| R2 | `ModelCfg` (pydantic 2) đòi field không có default (`policy_name`, `state_encoder`) → ValidationError | GATE 1 — dict ở Phase D đã điền sẵn cả hai |
| R3 | `device_map="auto"` đặt `navdp` lệch GPU với latent → lỗi device mismatch khi `generate_traj` | GATE 3 (Phase F) — có sẵn cách fix `.to()` |
| R4 | S2 rơi nhánh action (`←←←←`), không có latent — như lần đo PL-E1 (nhưng lần đó là bản wo-dagger + class HF thuần; lần này là bản DAgger + full policy → chính là thứ cần kiểm) | GATE 2 (Phase E) |
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

**D4. GATE 1 — 5 phép kiểm bắt buộc:**

```python
m = policy.model

# (1) S1 da duoc nap that su (PL-C2)
n_navdp = sum(p.numel() for n, p in m.named_parameters() if "navdp" in n)
print(f"navdp params: {n_navdp/1e6:.1f}M");  assert n_navdp > 0, "S1 KHONG duoc nap!"

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

**E2. Chạy S2 tuần tự vài frame đầu episode** (frame 0 chưa có history — đúng thiết kế, file 02 mục 3.2):

```python
policy.reset()                                   # xoa history giua cac episode — BAT BUOC
ep_len = int(eps[0]["length"])
records = []
for fr in range(min(ep_len, 6)):                 # smoke: 6 frame dau
    rgb, depth = load_frame(0, fr)
    out = policy.s2_step(rgb, depth, POSE, instruction, INTR, look_down=False)
    records.append(out)
    print(fr,
          "| pixel:",  out.output_pixel,
          "| action:", out.output_action,
          "| latent:", None if out.output_latent is None else tuple(out.output_latent.shape))
```

**GATE 2 — đọc kết quả:**

| Quan sát | Nghĩa là | Đi tiếp thế nào |
|---|---|---|
| Có frame ra `output_pixel` + `output_latent` | ✅ nhánh pixel-goal hoạt động — **khác biệt then chốt so với PL-E1** (lần đó bản wo-dagger + class HF thuần) | → Phase F |
| Toàn bộ ra `output_action` (mũi tên) | S2 vẫn né pixel-goal | Debug E3 ↓ |
| `output_action == []` hoặc text lạ | parser rơi lỗ hổng đã ghi (`../io_system2.md` 3.d) | In `policy.llm_output` thô từng frame, ghi lại làm bằng chứng |

**E3. Debug khi vẫn action-only (R4)** — thử theo thứ tự rẻ → đắt, ghi kết quả từng bước:
1. Chạy **nhiều frame hơn / episode khác / scene r2r** (một scene rxr nhỏ có thể toàn action hợp lệ
   — GT của nó cũng có nhiều frame action, xem cột `action` trong parquet).
2. Đối chiếu: frame nào GT `goal.{SETTING}` ≠ `(-1,-1)` mà model vẫn không ra pixel → đó mới là
   bất thường thật; frame GT không có goal thì model ra action là **đúng hành vi**.
3. Thử setting khác (`125cm_30deg`) + instruction khác.
4. Vẫn 100% action trên các frame có GT goal → dừng, cập nhật PL-E1 với dữ kiện mới
   (bản DAgger + full policy vẫn không bật pixel) — đây tự nó là một kết quả đáng báo cáo.

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

s1_out = policy.s1_step_latent(rgbs, depths, latent)
print("S1 actions:", s1_out.idx)                  # ky vong list 1-4 phan tu thuoc {1,2,3,5}
```

**GATE 3:**
- `s1_out.idx` là list hợp lệ (không rỗng, không NaN) → pipeline dual-system **thông**.
- Lỗi dtype (bf16 vs float32) → thử `rgbs = rgbs.to(torch.bfloat16)` (và depths tương tự) — ghi lại
  bản nào chạy. ⬜ chưa xác minh trước dtype nào đúng.
- Lỗi device → đã có fix `.to(navdp_dev)` ở trên; nếu vẫn lỗi trong `generate_traj`, in device từng
  input và báo lại (ứng viên bug report).

**F2 (tuỳ chọn — lấy quỹ đạo thô để visualize):** `s1_step_latent` chỉ trả action rời rạc đã
lượng tử hoá (file 02 mục 4.1); muốn vẽ waypoint thì gọi thẳng:

```python
with torch.no_grad():
    trajs = policy.model.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths)
print(type(trajs), getattr(trajs, "shape", None))   # ⬜ shape chua do — ghi lai lan dau chay
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
        rec = {"ep": ep_i, "fr": fr, "t_s2": round(time.time() - t0, 2),
               "gt_action": int(gt_act[fr]),
               "gt_goal": [int(x) for x in np.ravel(gt_goal[fr])],
               "pred_action": None if out.output_action is None else [int(a) for a in out.output_action],
               "pred_pixel_rowcol": None if out.output_pixel is None else [int(x) for x in out.output_pixel],
               "llm_output": policy.llm_output}
        if out.output_latent is not None:            # S1 chi chay khi co latent
            # ... prep nhu Phase F (frame goal = frame nay, frame hien tai = frame nay) ...
            s1 = policy.s1_step_latent(rgbs, depths, out.output_latent.to(navdp_dev))
            rec["s1_actions"] = [int(a) for a in s1.idx]
        RESULTS["frames"].append(rec)
        if fr % 8 == 0:
            torch.cuda.empty_cache()                 # quan ly VRAM tren loop dai

json.dump(RESULTS, open("/kaggle/working/results/results.json", "w"), indent=1)
```

**G2. Metric S2** (đúng giới hạn đã ghi ở [03](03_data_contract.md) mục 4.4):

```python
frames = RESULTS["frames"]

# (a) Action accuracy — loai frame GT = -1 (frame start)
pairs = [(f["gt_action"], f["pred_action"][0]) for f in frames
         if f["gt_action"] != -1 and f["pred_action"]]
acc = sum(g == p for g, p in pairs) / max(len(pairs), 1)
print(f"action acc: {acc:.2%}  ({len(pairs)} frame)")

# (b) Pixel-goal L2 — DAO [row,col]→[u,v] + scale 384→640x480 (file 02 muc 3.2; io_system2 3.d)
l2s = []
for f in frames:
    if f["pred_pixel_rowcol"] and f["gt_goal"] != [-1, -1]:
        row, col = f["pred_pixel_rowcol"]
        u, v = col * 640 / 384, row * 480 / 384          # ve he anh goc
        gu, gv = f["gt_goal"]
        l2s.append(((u - gu) ** 2 + (v - gv) ** 2) ** 0.5)
print(f"pixel L2 (px): mean={np.mean(l2s):.1f}  n={len(l2s)}" if l2s else "khong co frame pixel nao")
```

**G3. Visualize** — vẽ `pred_pixel` (chấm xanh) lên RGB gốc; nếu có F2 thì overlay quỹ đạo/bird-eye
→ lưu `/kaggle/working/results/vis/*.png` (≥3 ảnh cho báo cáo).

**G4. Ghi bản ghi phần cứng** vào `summary.json`: VRAM đỉnh mỗi GPU, giây/frame S2 và S1, version
(`transformers`, commit repo) — số này dùng cho mail xin server.

---

## §4. Definition of Done

- [ ] Kaggle Dataset `internvla-n1-w-navdp-ckpt` tồn tại, verify đủ 16 file + `system1: navdp_async` (GATE A).
- [ ] Notebook GPU qua **GATE 1**: `navdp params > 0`, `latent_queries (1,4,3584)`, tokenizer hoạt động, model chia 2 GPU.
- [ ] **GATE 2** có kết luận rõ (bật được pixel-goal hay không — kể cả "không" cũng là kết quả, cập nhật PL-E1).
- [ ] Nếu có latent: **GATE 3** — S1 xuất action hợp lệ; depth range in ra nằm trong [0,5] m.
- [ ] `results.json` + `summary.json` + ≥3 ảnh visualize trong `/kaggle/working/results/`.
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
| 5 | S2 không bật pixel-goal (R4) | GATE 2 toàn action | quy trình E3; kết luận trung thực + cập nhật PL-E1 |
| 6 | Device/dtype mismatch S1 (R3) | RuntimeError trong `generate_traj` | `.to(navdp_dev)` cho latent/inputs; thử bf16; ghi lại tổ hợp chạy được |
| 7 | Submodule diffusion-policy rỗng (R6) | ImportError `diffusion_policy` | pip git pin (C3) |
| 8 | Pattern tên PNG khác dự đoán | FileNotFoundError trong `load_frame` | `os.listdir` in 5 tên đầu, sửa f-string — đã đánh dấu ⬜ ở E1 |
| 9 | Tràn 20GB working ở Phase A | Errno 28 | HF_HOME đã trỏ `/kaggle/temp` (A2); không giải nén gì thêm vào working |

## §7. Sau khi xong — cập nhật tài liệu

1. Điền kết quả GATE 1–3 + số đo (VRAM, s/frame, acc, L2) vào [05_appendix](05_appendix.md)
   (thêm mục PL-E2 "chạy thật w-NavDP") — các mục ⬜ trong kế hoạch này chuyển thành ✅/❌.
2. Cập nhật [04_checkpoint_details](04_checkpoint_details.md) mục 2.1 (bỏ dòng "chưa tải/chạy bản này")
   và mục 6 câu hỏi mở #2.
3. Nếu GATE 2 bật được pixel-goal → đóng luôn nghi vấn PL-E1 (nghi phạm số 3 "cần full policy" được
   xác nhận/bác bỏ).
