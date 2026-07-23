# 01 — System Requirements: Cấu hình yêu cầu & cách setup để chạy InternVLA-N1 trên Kaggle

> **File này để làm gì:** liệt kê phần cứng/phần mềm bắt buộc và **trình tự chuẩn bị** trước khi chạy
> inference InternVLA-N1 (dual-system) trên Kaggle. Viết cho người chưa từng dùng Kaggle/HuggingFace.
> Mọi con số có gắn mã **PL-xx** đều là số **đo thật**, tra được trong [05_appendix.md](05_appendix.md).
>
> Bộ tài liệu: [02_code_structure](02_code_structure.md) · [03_data_contract](03_data_contract.md) ·
> [04_checkpoint_details](04_checkpoint_details.md) · [05_appendix](05_appendix.md)

---

## 0. Từ điển thuật ngữ tối thiểu

| Thuật ngữ | Nghĩa trong ngữ cảnh này |
|---|---|
| **Checkpoint** | File trọng số model đã huấn luyện (`.safetensors`). Model lớn bị chia thành nhiều file (shard). Checkpoint của ta nặng **16.79 GB**. |
| **VRAM** | Bộ nhớ của GPU. Hết VRAM (OOM — out of memory) là process chết. |
| **bf16** | Kiểu số 16-bit → mỗi tham số chiếm 2 byte → model 8.3 tỷ tham số ≈ 16.6 GB. |
| **Kaggle Notebook / Dataset / Accelerator** | Notebook = môi trường chạy code có GPU miễn phí. Dataset = dữ liệu mount sẵn vào `/kaggle/input/<tên>`. Accelerator = loại GPU chọn trong Settings. |
| **`device_map="auto"`** | Bảo thư viện `transformers` tự chia model ra nhiều GPU cho vừa bộ nhớ. |
| **HF (HuggingFace)** | Kho model + dataset. Tải bằng thư viện `huggingface_hub`. |
| **flash-attention** | Kỹ thuật tăng tốc attention, **cần GPU đời mới**. T4 của Kaggle là đời cũ (sm_75) → phải né. |
| **Open-loop** | Cho model xem ảnh render sẵn từ dataset, không có simulator phản hồi. Đây là thứ chạy được trên Kaggle. Closed-loop (model lái robot trong simulator) cần server riêng, **không** chạy Kaggle. |

---

## 1. Yêu cầu phần cứng (Kaggle)

### 1.1. GPU: bắt buộc **T4 × 2** — không phải lựa chọn, là ép buộc số học

Weights System 2 đo thật là **16.58 GB** ở bf16 (PL-B3), vì base model là Qwen2.5-VL-**7B**
(xác nhận hai nguồn độc lập — PL-B2):

| Cấu hình Kaggle | VRAM | Kết luận |
|---|---|---|
| 1× T4 | 15 GB | ❌ Weights một mình đã vượt |
| 1× P100 | 16 GB | ❌ Sát nút, không còn chỗ cho KV-cache/vision token |
| **2× T4** | **2×15 GB** | ✅ Duy nhất chạy nổi bf16, **bắt buộc** `device_map="auto"` để chia model ra 2 GPU |

Số đo khi chạy thật trên T4×2 (PL-B3): GPU0 = 7.57 GB, GPU1 = 9.01 GB, tổng 16.58 GB, dư ~13 GB
cho KV-cache. Ước tính tổng nhu cầu thực tế ~19–21 GB khi generate.

> ⚠️ Code gốc của repo **pin toàn bộ model vào MỘT GPU** (`device_map={"": device}` — PL-C3)
> → nếu chạy nguyên bản trên Kaggle sẽ OOM. Cách override ở mục 3, bước 6.

### 1.2. Đĩa: trần cứng 20 GB — ngân sách phải tính trước

Đo thật (PL-A1): `/kaggle/working` = 20 GB; checkpoint chiếm 16.79 GB → **chỉ dư ~3.2 GB** cho mọi
thứ còn lại (data, kết quả, file tạm).

Quy tắc sống còn:
- File tải/giải nén tạm → để ở `/kaggle/temp` (không tính vào 20 GB), **nhớ tự tạo thư mục** (PL-A1).
- Cache HuggingFace → trỏ ra ngoài `/kaggle/working` bằng `HF_HOME`, nếu không dung lượng tải bị
  **nhân đôi** (cache 1 bản + đích 1 bản) và tràn đĩa giữa chừng (PL-A2).
- Dọn `/kaggle/temp` trước khi bấm **Save Version** (file trong working sẽ bị đóng gói thành output).

### 1.3. RAM & swap

RAM 31 GB — đủ. **Swap = 0** (PL-A1): hết RAM là kernel **chết ngay không báo lỗi** — triệu chứng
là "kernel died" khó hiểu. Đừng chẩn đoán nhầm thành lỗi mạng/lỗi code.

---

## 2. Yêu cầu phần mềm

| Thành phần | Trạng thái trên Kaggle | Lưu ý |
|---|---|---|
| `transformers` | **5.0.0 preinstall** | Load S2 bằng class HF thuần chạy bình thường (PL-A6). Dùng `dtype=` thay `torch_dtype=` khi gọi trực tiếp. |
| `torch` | bản CUDA ở session GPU | Session CPU là bản `+cpu` — **đo lại baseline khi đổi loại session** (PL-A1). |
| Repo **InternNav** | phải tự cài: `pip install -e .` | **Bắt buộc** nếu muốn chạy đủ dual-system (S1+S2). Class HF thuần chỉ load được S2, toàn bộ System 1 bị bỏ qua im lặng (~120 tensor `UNEXPECTED` — PL-B3). |
| `flash-attn` | ❌ khó/không chạy trên T4 (sm_75) | Code repo hardcode `flash_attention_2` → phải ép `attn_implementation="sdpa"` khi load (mục 3 bước 6). |
| Internet | Bật trong Settings notebook | Cần để tải data + cài repo. |

⚠️ Rủi ro chưa kiểm chứng (PL-A6): `pip install -e .` của repo có thể **hạ cấp `transformers`**
xuống 4.x → đọc `pyproject.toml`/`requirements` của repo TRƯỚC khi cài; nếu version đổi thì
**restart kernel** rồi chạy lại cell baseline.

---

## 3. Trình tự setup trước khi chạy (checklist làm theo thứ tự)

### Bước 1 — Tạo notebook đúng cấu hình

- Accelerator: **GPU T4 ×2** (không phải P100 — mục 1.1).
- Internet: **ON**.
- Mount 2 Kaggle Dataset (Add Input):
  1. `tieulam/internvla-n1-ckpt` — checkpoint 16.79 GB đã đóng gói sẵn từ 21/07
     (nguồn gốc và lưu ý về bản này: [04_checkpoint_details](04_checkpoint_details.md) mục 2.3).
  2. Dataset chứa data `vln_ce` (tự đóng gói ở bước 4, hoặc dùng lại nếu đã có).

### Bước 2 — Cell baseline (chạy đầu tiên, mọi session)

```python
import os, torch, transformers
print(torch.__version__, transformers.__version__)
os.makedirs("/kaggle/temp/hf_cache", exist_ok=True)      # /kaggle/temp KHONG ton tai san (PL-A1)
os.environ["HF_HOME"] = "/kaggle/temp/hf_cache"          # tranh cache ghi dup (PL-A2)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"              # tang toc tai; bien cu da deprecated (PL-A3)
```

Set biến môi trường **trước khi** import `huggingface_hub`.

### Bước 3 — Quy tắc tải file từ HF (tránh 2 bẫy đã dính)

```python
import fnmatch
from huggingface_hub import HfApi, hf_hub_download

files = HfApi().list_repo_files("InternRobotics/InternData-N1", repo_type="dataset")
m = fnmatch.filter(files, "vln_ce/traj_data/r2r/*")
assert m, "PATTERN KHONG KHOP - dung lai, dung download"   # bay '0 file van bao thanh cong' (PL-A4)

tgz = hf_hub_download("InternRobotics/InternData-N1",
                      filename=m[0], repo_type="dataset",     # thieu repo_type => 404 (PL-A5)
                      local_dir="/kaggle/temp/cedl")
```

- Dataset **bắt buộc** `repo_type="dataset"` — quên là lỗi 404 trông như lỗi quyền (PL-A5).
- Ưu tiên `hf_hub_download` (ném exception khi sai) thay vì `!hf download --include`
  (im lặng khi khớp 0 file — PL-A4).

### Bước 4 — Chuẩn bị data `vln_ce` (làm ở notebook CPU riêng cho rẻ)

Giải nén vào `/kaggle/temp`, chọn setting camera **có góc cúi** (`60cm_30deg` / `125cm_30deg`…,
**tránh `125cm_0deg`** vì goal toàn `(-1,-1)` — PL-D4), dựng index frame, rồi Save Version thành
Kaggle Dataset. Chi tiết schema và cách đọc từng cột: [03_data_contract](03_data_contract.md).

### Bước 5 — Cài repo InternNav (session GPU)

```bash
!git clone https://github.com/InternRobotics/InternNav.git --recursive
# DOC pyproject.toml / requirements TRUOC: co pin transformers<5 khong? (PL-A6)
!pip install -e ./InternNav --no-build-isolation
```

Nếu lệnh cài đổi version `transformers` → **restart kernel** rồi chạy lại Bước 2.

### Bước 6 — Load model với 2 override bắt buộc

```python
import torch
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

MODEL = "/kaggle/input/internvla-n1-ckpt/..."   # duong dan mount

model = InternVLAN1ForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",     # OVERRIDE 1: KHONG flash_attention_2 (T4 sm_75) — muc 2
    device_map="auto",              # OVERRIDE 2: KHONG pin 1 GPU nhu code goc (PL-C3)
)
print(model.hf_device_map)          # ky vong: chia deu 2 GPU nhu PL-B3
```

Vì sao phải override và code gốc nằm ở đâu: [02_code_structure](02_code_structure.md) mục 1.

### Bước 7 — Verify System 1 đã thật sự được nạp (bắt buộc, đừng tin log)

Checkpoint đã đóng gói có `config.json` **thiếu field `system1`** → nếu không patch, S1 bị vứt
im lặng **dù đã dùng đúng class repo** (cơ chế: PL-C2; bằng chứng config: PL-B6).

```python
# TRUOC khi load: doc config.json trong checkpoint mount
import json
cfg = json.load(open(f"{MODEL}/config.json"))
print(cfg.get("system1"))           # None => phai patch: them "system1": "navdp_async"

# SAU khi load: dem tham so cua navdp
n_navdp = sum(p.numel() for n, p in model.named_parameters() if 'navdp' in n)
assert n_navdp > 0, "S1 (navdp) CHUA duoc nap — kiem tra config.system1"
```

Phương án thay thế không cần patch: dùng checkpoint `InternVLA-N1-w-NavDP` (config đã chuẩn sẵn
`system1: "navdp_async"`) — đánh đổi là phải tải mới 16.78 GB và bản đó **không có fast tokenizer**.
So sánh đầy đủ: [04_checkpoint_details](04_checkpoint_details.md) mục 2.1–2.3.

### Bước 8 — Đo VRAM sau khi load

```python
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.max_memory_allocated(i) / 1e9, "GB")
```

Kỳ vọng ~16.6 GB tổng (PL-B3). Nếu sát trần → chuẩn bị phương án 4-bit hoặc chuyển server ≥24 GB.

---

## 4. Bảng lỗi thường gặp & cách xử lý nhanh

| Triệu chứng | Nguyên nhân | Xử lý | Bằng chứng |
|---|---|---|---|
| `ValueError: Unrecognized configuration class Qwen2_5_VLConfig` | Dùng `AutoModelForCausalLM` (chỉ map model text thuần) | Dùng `Qwen2_5_VLForConditionalGeneration` (S2 thuần) hoặc `InternVLAN1ForCausalLM` (dual) | PL-A6 |
| 404 khi tải dataset | Thiếu `repo_type="dataset"` | Thêm tham số | PL-A5 |
| `Fetching 0 files` nhưng vẫn `✓ Downloaded` | Pattern `--include` không khớp / placeholder chưa thay | Đếm `fnmatch.filter` trước; dùng `hf_hub_download` | PL-A4 |
| `OSError: [Errno 28] No space left on device` | Cache HF ghi đúp vào `/kaggle/working` | `HF_HOME=/kaggle/temp/hf_cache`; giải nén ở `/kaggle/temp` | PL-A2 |
| Kernel chết im lặng | OOM RAM (swap=0) hoặc OOM VRAM | Đổi T4×2 + `device_map="auto"`; theo dõi `max_memory_allocated` | PL-A1, PL-B3 |
| Lỗi build/import `flash_attn` | T4 không hỗ trợ tốt flash-attn 2 | Ép `attn_implementation="sdpa"` | mục 2 |
| Model chạy nhưng S1 không có output | `config.system1` thiếu → `self.navdp` không được dựng | Patch config + verify đếm params | PL-C2, PL-B6 |
| ~120 dòng `UNEXPECTED` khi load | Load bằng class HF thuần → không có chỗ chứa `navdp.*` | Cài repo, dùng `InternVLAN1ForCausalLM` | PL-B3 |

---

## 5. Giới hạn cần biết trước khi bắt đầu

- Kaggle chỉ chạy được **open-loop** (ảnh render sẵn từ `vln_ce`). Điểm benchmark thật (SR/SPL)
  đòi closed-loop trong Habitat/Isaac → cần server ≥24 GB VRAM, ngoài phạm vi Kaggle.
- Session GPU giới hạn ~9h/phiên và có quota tuần — tách bước chuẩn bị data sang notebook CPU.
- Blocker hiện tại của pipeline: lần chạy S2 thật mới nhất trả về chuỗi action `←←←←` thay vì
  pixel-goal → **chưa có latent để nuôi System 1** (PL-E1). Xem nghi phạm và việc-cần-làm-tiếp trong
  PL-E1 trước khi kỳ vọng dual-system chạy trọn vẹn.
