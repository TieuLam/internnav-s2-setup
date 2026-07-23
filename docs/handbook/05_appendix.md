# 05 — Phụ lục: Bằng chứng đo được trong quá trình chạy thử

> **File này để làm gì:** đây là "kho bằng chứng" của bộ tài liệu. Mọi kết luận trong các file
> [01_system_requirements](01_system_requirements.md), [02_code_structure](02_code_structure.md),
> [03_data_contract](03_data_contract.md), [04_checkpoint_details](04_checkpoint_details.md)
> đều trỏ về một mục **PL-xx** ở đây. Khi đọc các file kia mà thắc mắc *"lấy đâu ra kết luận này?"*,
> tra mã PL tương ứng trong file này.
>
> **Quy ước:**
> - ✅ = đã chạy thử / đo thật, có output kèm theo.
> - 📖 = đọc trực tiếp từ source code repo InternNav hoặc từ HuggingFace API (không bịa, có đường dẫn).
> - ⬜ = **chưa xác minh** — chỉ là suy luận/giả thuyết, KHÔNG được dùng làm căn cứ quyết định.
> - Ngày đo ghi theo từng mục. Số dòng code lấy theo bản clone local `InternNav/code`
>   (InternNav **v0.3.1**, commit `7a5c624`, remote `https://github.com/InternRobotics/InternNav.git`).
>
> **Nguồn gốc:** tổng hợp từ `SETUP_NOTES.md` (nhật ký đo 21–22/07/2026), các file
> `docs/{io_system2, data_contract, checkpoint_variants, vln_subsets_architecture, eval_plan_kaggle_s2}.md`,
> và phiên xác minh bổ sung ngày **23/07/2026** (đọc code local + HuggingFace API).

---

## Nhóm A — Môi trường Kaggle

### PL-A1 ✅ Baseline session Kaggle (đo 21/07/2026)

Lệnh chạy trên session CPU:

```bash
!nvidia-smi || echo "CPU session - dung"
!df -h /kaggle/working /kaggle/temp
!free -g
!python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Output thô:

```
/bin/bash: line 1: nvidia-smi: command not found
df: /kaggle/temp: No such file or directory
/dev/loop1       20G   72K   20G   1% /kaggle/working
Mem:  31 total / 27 free       Swap: 0 0 0
2.10.0+cpu 5.0.0
```

| Hạng mục | Giá trị đo được | Hệ quả |
|---|---|---|
| `/kaggle/working` | **20 GB** trần cứng | Checkpoint 16.79 GB → chỉ dư ~3.2 GB |
| `/kaggle/temp` | **không tồn tại sẵn** | Phải `os.makedirs` trước khi dùng (PL-A2) |
| RAM | 31 GB | Đủ cho việc load model |
| Swap | **0** | Hết RAM = process chết ngay, không cảnh báo |
| `torch` | 2.10.0+cpu (session CPU) | Session GPU sẽ là bản CUDA — phải đo lại baseline |
| `transformers` | **5.0.0** preinstall | Xem PL-A6 |

> Nguồn: `SETUP_NOTES.md` mục 1, 2.5, 2.6.

### PL-A2 ✅ Cache HuggingFace ghi đúp — fix bằng `HF_HOME` (đo 21/07/2026)

`hf download --local-dir X` ghi **một bản vào cache** rồi mới copy sang `X` → tải 16.79 GB sẽ chiếm
~33 GB nếu cache nằm trong `/kaggle/working` (trần 20 GB) → tràn đĩa giữa chừng
(`OSError: [Errno 28] No space left on device`).

Fix đã kiểm chứng:

```python
import os
os.makedirs("/kaggle/temp/hf_cache", exist_ok=True)   # /kaggle/temp phai tu tao (PL-A1)
os.environ["HF_HOME"] = "/kaggle/temp/hf_cache"       # set TRUOC khi import huggingface_hub
```

**Kiểm chứng kết quả:** sau khi tải 16.79 GB, `du -sh /kaggle/working` = ~16G (không phải ~32G)
→ cache không còn ăn vào vùng 20 GB. Nguồn: `SETUP_NOTES.md` 2.1.

### PL-A3 ✅ `HF_HUB_ENABLE_HF_TRANSFER` đã deprecated (đo 21/07/2026)

Cảnh báo thật gặp phải:

```
FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as
'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead.
```

→ Dùng `os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"` (set trước khi import). Dấu hiệu hoạt động:
log có dòng `Reconstructing` / `Reconstruction complete`. Nguồn: `SETUP_NOTES.md` 3.12.

### PL-A4 ✅ Bẫy `hf download --include` khớp 0 file vẫn báo "✓ Downloaded" (dính 2 lần, 21/07/2026)

```
Fetching 0 files: 0it [00:00, ?it/s]
✓ Downloaded  path: /kaggle/working/data
```

- Lần 1: `--include "preview/*"` — thư mục `preview/` **không tồn tại** trong `InternData-N1`.
- Lần 2: `--include "<placeholder>"` — dòng `!` là shell, **không nội suy biến Python**.

**Quy tắc rút ra (bắt buộc):** đếm số file khớp pattern TRƯỚC khi tải, và ưu tiên API Python
(`hf_hub_download` **ném exception** khi sai tên; `hf download --include` thì im lặng):

```python
import fnmatch
from huggingface_hub import HfApi
files = HfApi().list_repo_files("InternRobotics/InternData-N1", repo_type="dataset")
m = fnmatch.filter(files, "<pattern>")
assert m, "PATTERN KHONG KHOP - dung download"
```

> Nguồn: `SETUP_NOTES.md` 3.10.

### PL-A5 ✅ Bảng lỗi HuggingFace 401/403/404 (đúc kết 21/07/2026)

| Mã | Nguyên nhân thật | Xử lý |
|---|---|---|
| **401** Unauthorized | Vấn đề *token*: chưa set / sai / thu hồi | Kiểm tra `HF_TOKEN`, `.strip()` whitespace |
| **403** GatedRepoError | Token đúng nhưng *tài khoản* chưa được duyệt | Bấm agree trên trang repo |
| **404** Not Found | Thường là **thiếu `repo_type="dataset"`** | Dataset bắt buộc khai `repo_type` |

> Nguồn: `SETUP_NOTES.md` mục 5.

### PL-A6 ✅ `transformers` 5.0.0 trên Kaggle — cái gì chạy, cái gì không (đo 21/07/2026)

- ✅ **Load System 2 bằng class HF thuần chạy bình thường** trên 5.0.0 — `Qwen2_5_VLForConditionalGeneration`
  có sẵn native, tokenizer + vision processor không lỗi.
- ❌ `AutoModelForCausalLM` **không load được** Qwen2.5-VL (lỗi thật:
  `ValueError: Unrecognized configuration class Qwen2_5_VLConfig...`) — vì Auto class này chỉ map model
  text thuần. Phải dùng `Qwen2_5_VLForConditionalGeneration` (hoặc `AutoModelForImageTextToText`).
- ⚠️ `torch_dtype=` bị deprecated ở 5.x → dùng `dtype=` (khi gọi trực tiếp API transformers 5.x).
- ✅ **Không cần** `trust_remote_code=True` (model được hỗ trợ native — bật chỉ thêm rủi ro).
- ⬜ `pip install -e .` repo InternNav có thể **hạ cấp** transformers xuống 4.x và kéo xung đột `torch`
  — rủi ro chưa kiểm chứng, phải đọc `pyproject.toml` trước khi cài và **restart kernel** sau khi cài.

> Nguồn: `SETUP_NOTES.md` 2.2, 2.3.

---

## Nhóm B — Checkpoint

### PL-B1 ✅ Survey dung lượng repo HF (đo 21/07/2026, qua HF API — không tải weights)

```
=== InternRobotics/InternVLA-N1-System2 | TONG 16.59 GB | 18 file ===
    4.968 / 4.991 / 4.933 / 1.692 GB  (4 shard safetensors)
=== InternRobotics/InternVLA-N1 | TONG 16.79 GB | 18 file ===      ← bản đã đóng gói
    4.966 / 4.991 / 4.933 / 1.888 GB
=== InternRobotics/VLN-PE | TONG 1.86 GB | 25 file ===
    rdp 0.565 / cma 0.148 / seq2seq 0.133 ... (fine_tuned + zero_shot)
```

**Điểm mấu chốt:** shard 1–3 của `InternVLA-N1` và `-System2` gần **trùng byte** → chung backbone
VLM (System 2). Chênh **+0.196 GB ở shard 4** = phần System 1 (NavDP) gộp thêm — được xác nhận
bằng tên tensor thật ở PL-B3. Nguồn: `SETUP_NOTES.md` mục 3, 3.2.

### PL-B2 ✅ Base model là Qwen2.5-VL-**7B**, không phải 3B (đo 21/07/2026)

Hai nguồn bằng chứng độc lập:

1. **Số học dung lượng:** weights bf16 = 2 byte/tham số → 16.59 GB ÷ 2 ≈ 8.3 tỷ tham số.
2. **`config.json` tải riêng (vài KB):** `hidden_size: 3584`, `num_hidden_layers: 28` — đúng chữ ký
   Qwen2.5-VL-7B (bản 3B là 2048/36).

Hệ quả: **T4 15GB đơn hoặc P100 16GB không chạy nổi bf16** — riêng weights đã 16.6 GB.
Nguồn: `SETUP_NOTES.md` 3.1, 3.9.

### PL-B3 ✅ Smoke test load S2 trên Kaggle T4×2 (chạy 21/07/2026, notebook `internnav-s2-smoke`)

| Hạng mục | Giá trị đo được |
|---|---|
| Params nạp được | **8.289 B** (đúng 7B-class) |
| GPU0 | 7.57 GB (`model.visual` + layer 0–10) |
| GPU1 | 9.01 GB (layer 11–27 + `norm` + `lm_head`) |
| **Tổng VRAM weights** | **16.58 GB** |
| `device_map="auto"` | shard đúng qua 2 GPU, không dồn `cuda:0` |

**Phát hiện quan trọng nhất:** log in ~120 dòng `UNEXPECTED`, tất cả cùng tiền tố:

```
model.language_model.navdp.rgbd_encoder.rgb_model.*      ← encoder RGB (12 block)
model.language_model.navdp.rgbd_encoder.depth_model.*    ← encoder Depth (12 block)
model.language_model.navdp.rgbd_encoder.former_net.*
model.language_model.navdp.decoder.layers.{0..15}.*      ← diffusion decoder 16 layer
model.language_model.navdp.action_head.* / critic_head.*
model.language_model.navdp.goal_compressor.*
model.language_model.latent_queries                      ← cầu nối S2 → S1
```

`UNEXPECTED` = tensor **có trong checkpoint nhưng kiến trúc đang dựng không có chỗ chứa** → bị bỏ qua
im lặng. Tức: load bằng class HF thuần thì **chỉ có System 2**; toàn bộ System 1 (NavDP) bị vứt.
Muốn đủ dual-system phải dùng model class của repo (xem PL-C1, PL-C2).
Nguồn: `SETUP_NOTES.md` 2.4.

### PL-B4 ✅ Khác biệt tokenizer giữa các repo (đo 21/07/2026 + 23/07/2026)

| Repo | `tokenizer.json` (fast) | Ghi chú |
|---|---|---|
| `InternVLA-N1` (nay là `-wo-dagger`, xem PL-B6) | ✅ có | nạp thẳng, an toàn với transformers 5.x |
| `InternVLA-N1-System2` | ❌ không (chỉ `vocab.json`+`merges.txt`) | slow tokenizer, 5.x rủi ro |
| `InternVLA-N1-System2-wo-dagger` | ✅ có (kiểm 23/07 qua HF API) | |
| `InternVLA-N1-DualVLN` | ❌ không | nên pin `transformers==4.51.0` nếu dùng |
| `InternVLA-N1-w-NavDP` | ❌ **không** (kiểm 23/07, liệt kê đủ 16 file — không có `tokenizer.json`, không có `trainer_state.json`) | |

> Nguồn: `SETUP_NOTES.md` 3.6, 3.15; HF API `/api/models/.../tree/main` fetch 23/07/2026.

### PL-B5 ✅📖 Survey `InternVLA-N1-DualVLN` (đo 22/07/2026 qua HF API + đọc weight keys)

- Tổng safetensors **16.77 GB** (không phải ~8GB như comment trong notebook demo).
- `config.json`: `model_type: internvla_n1`, `architectures: InternVLAN1ForCausalLM`,
  **`system1: "nextdit_async"`**.
- Weight keys (đọc `model.safetensors.index.json`): S1 là **NextDiT** (`traj_dit` 12 lớp DiT,
  `rgb_model`, `rgb_resampler`, `memory_encoder`) — **không có nhánh depth riêng, không có critic** —
  khác hẳn NavDP (decoder 16 lớp + `rgbd_encoder` RGB-D + `critic_head`).
- Shard 1–3 gần trùng byte với bản NavDP → **chung System 2**.

> Nguồn: `docs/checkpoint_variants.md` mục 1, 3, 4; `SETUP_NOTES.md` 3.15.

### PL-B6 📖 (MỚI 23/07/2026) Trạng thái hiện tại trên HuggingFace: repo `InternVLA-N1` đã ĐỔI TÊN

Fetch HF API ngày 23/07/2026 (`/api/models?author=InternRobotics` + từng repo):

1. **`https://huggingface.co/api/models/InternRobotics/InternVLA-N1` giờ redirect về
   `InternRobotics/InternVLA-N1-wo-dagger`** — tức repo mà nhóm đã survey/đóng gói ngày 21/07
   dưới tên `InternVLA-N1` nay mang tên mới `-wo-dagger` (wo = without DAgger). Danh sách file khớp
   với bản đã đóng gói: có `tokenizer.json`, có `trainer_state.json`, 4 shard.
2. Xuất hiện repo **`InternRobotics/InternVLA-N1-w-NavDP`** (tạo 10/12/2025, sửa lần cuối
   10/12/2025) — đúng là bản mà README model zoo v0.3.1 link cho dòng
   *"InternVLA-N1 (Dual System) w/ NavDP\*"*.
3. Bảng `config.json` so sánh (fetch raw 23/07/2026):

| Trường | `-wo-dagger` (= bản đã đóng gói) | `-w-NavDP` | `-DualVLN` (đo 22/07) |
|---|---|---|---|
| `model_type` | `qwen2_5_vl` | **`internvla_n1`** | `internvla_n1` |
| `architectures` | `InternVLAN1ForCausalLM` | `InternVLAN1ForCausalLM` | `InternVLAN1ForCausalLM` |
| `system1` | ❌ **KHÔNG CÓ** | ✅ **`"navdp_async"`** | ✅ `"nextdit_async"` |
| `n_query` | 16 | 4 | (chưa ghi) |
| `hidden_size` / layers | 3584 / 28 | 3584 / 28 | 3584 / 28 |

   → **Giải quyết dứt điểm câu hỏi mở** trong `checkpoint_variants.md` mục 6.6: config bản đã đóng gói
   **thật sự thiếu field `system1`** → load bằng class repo thì S1 **vẫn bị vứt im lặng**
   (cơ chế ở PL-C2). Trong khi `-w-NavDP` có `system1: "navdp_async"` → dựng S1 đúng.
4. Kích thước `-w-NavDP`: 4 shard = 4.965 + 4.991 + 4.933 + 1.888 GB ≈ **16.78 GB**, 8.39B params bf16.
5. `InternVLA-N1-Preview`: tồn tại (tạo 21/07/2025), `model_type: internvla_n1`, có `tokenizer.json`,
   ~16.78 GB. ⬜ Giá trị `system1` trong config **chưa đọc được** (endpoint raw trả 401 khi fetch).
6. `InternVLA-N1-System2-wo-dagger`: tồn tại (tạo 25/07/2025), `model_type: qwen2_5_vl`,
   có `tokenizer.json`, ~16.6 GB (8.29B params).

> Nguồn: HuggingFace API, fetch ngày 23/07/2026. Danh sách đầy đủ checkpoint và vai trò từng bản:
> [04_checkpoint_details.md](04_checkpoint_details.md).

---

## Nhóm C — Đọc source code (xác minh trên clone local v0.3.1, ngày 23/07/2026)

> Các mục dưới đây ban đầu đọc trên GitHub (22/07); ngày 23/07 đã **xác minh lại từng dòng trên bản
> clone local** `InternNav/code` (v0.3.1, commit `7a5c624`) — số dòng ghi theo bản local này.

### PL-C1 📖 Chuỗi load checkpoint S1+S2 — một lệnh `from_pretrained` duy nhất

**File 1 — điểm load:** `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`, dòng 33–38
(class `InternVLAN1Net.__init__`):

```python
self.model = InternVLAN1ForCausalLM.from_pretrained(
    self.model_config.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": self.model_config.device},
)
```

**File 2 — dựng chỗ chứa S1 trước khi đổ weights:** `internvla_n1_arch.py`, dòng 121–145
(`InternVLAN1MetaModel.__init__`) — xem nguyên văn ở PL-C2.

**File 3 — loader NavDP standalone (tuỳ chọn):** `navdp.py`, dòng 116–125
(`NavDP_Policy_DPT_CriticSum_DAT.load_model`):

```python
def load_model(self):
    ...
    if self.navdp_pretrained is None:
        rank0_print("No pretrained weights provided, initializing randomly.")
        return
    try:
        pretrained_dict = torch.load(self.navdp_pretrained)
```

→ **Kết luận:** S1 không có file weights riêng trong checkpoint dual-system — `navdp.*` nằm trong 4
shard chung và chỉ "rơi vào đúng chỗ" nếu kiến trúc đã dựng `self.navdp` trước đó. Khớp và giải thích
trọn hiện tượng ~120 `UNEXPECTED` ở PL-B3.

### PL-C2 📖 Cổng `hasattr(config, "system1")` — lý do S1 có thể bị vứt im lặng DÙ dùng đúng class repo

Nguyên văn `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`, dòng 121–145:

```python
class InternVLAN1MetaModel:
    def __init__(self, config):
        super(InternVLAN1MetaModel, self).__init__(config)
        if hasattr(config, "system1"):                                    # <-- CỔNG (dòng 124)
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))

            if 'nextdit' in config.system1:
                self.traj_dit, self.noise_scheduler = build_traj_dit(config)
                ...
            elif 'navdp' in config.system1:
                if 'async' in config.system1:
                    self.navdp = build_navdp(config, memory_size=2)       # <-- dòng 143
            else:
                raise NotImplementedError
```

Và `build_navdp` (cùng file, dòng 10–15):

```python
def build_navdp(navdp_cfg, memory_size):
    from .navdp import NavDP_Policy_DPT_CriticSum_DAT
    navdp = NavDP_Policy_DPT_CriticSum_DAT(memory_size=memory_size, navdp_version=0.1)
    navdp.load_model()
    return navdp
```

**Bảng rẽ nhánh theo giá trị `config.system1`:**

| `config.system1` | Kết quả dựng kiến trúc |
|---|---|
| *(không có field)* | **Không dựng gì** — cả `latent_queries` lẫn `navdp` đều không tồn tại → toàn bộ weights S1 trong checkpoint bị `UNEXPECTED` im lặng |
| `"navdp_async"` | ✅ dựng `latent_queries` + `self.navdp` (NavDP) |
| `"navdp"` (không có `async`) | ⚠️ dựng `latent_queries` nhưng **KHÔNG dựng `navdp`** (điều kiện lồng `'async' in ...` không thoả — đọc nguyên văn ở trên) |
| `"nextdit_async"` | ✅ dựng `latent_queries` + NextDiT + DepthAnything encoder |

Đối chiếu PL-B6: bản đóng gói (`-wo-dagger`) **không có field `system1`** → rơi vào hàng 1.
**Cách xử lý khi muốn dùng bản đã đóng gói:** patch `config.json` thêm `"system1": "navdp_async"`
trước khi `from_pretrained`. ⬜ Việc patch này **chưa chạy thử** — sau khi load bắt buộc verify:

```python
n_navdp = sum(p.numel() for n, p in model.named_parameters() if 'navdp' in n)
assert n_navdp > 0, "S1 (navdp) chua duoc nap!"
```

### PL-C3 📖 Lời gọi load gốc pin toàn bộ model vào MỘT GPU

Dòng 37 của `internvla_n1_policy.py` (trích ở PL-C1): `device_map={"": self.model_config.device}` —
cú pháp này nghĩa là "đặt **toàn bộ** model lên một device". Với checkpoint 16.79 GB và T4 15 GB
→ OOM chắc chắn (số đo PL-B3: cần 16.58 GB chỉ riêng weights). Trên Kaggle T4×2 phải override
`device_map="auto"` (cách smoke test PL-B3 đã chạy thành công). Agent realworld
(`internvla_n1_agent_realworld.py` dòng 31–35) cũng pin một device y hệt.

### PL-C4 📖 Cơ chế bất đồng bộ trong agent — thread S2 + main-thread S1

`internnav/agent/internvla_n1_agent.py` (v0.3.1):

- Dòng 36: `self.mode = getattr(self._model_settings, 'infer_mode', 'sync')` — hai mode `sync` /
  `partial_async`.
- Dòng 37: `self.sys2_max_forward_step = getattr(..., 'sys2_max_forward_step', 8)`.
- Dòng 68–76: tạo thread + lock + `self._start_s2_thread()`.
- Dòng 133–208: `_start_s2_thread` — vòng lặp daemon: chờ cờ `should_infer` → gọi
  `self.policy.s2_step(...)` (dòng 159–166) → ghi kết quả vào `self.s2_output` dưới lock.
- Dòng 210–241: `should_infer_s2(mode)` — docstring nguyên văn trong code:
  *"sync": Sys1 and Sys2 execute in a sequential inference chain; "partial_async": Sys2 performs
  a single inference, while Sys1 performs multiple inference cycles.*
- Dòng 269: comment nguyên văn `# S1 inference is done in the main thread`.
- Dòng 334 / 336: hai đường gọi `self.policy.s1_step_latent(...)` (partial_async / sync).

Chi tiết luồng đầy đủ + giải thích: [02_code_structure.md](02_code_structure.md) mục 6.

### PL-C5 📖 Agent chạy trên observation tĩnh — không cần simulator

`internvla_n1_agent.py` dòng 243–250 (`step(obs)`):

```python
obs = obs[0]
rgb = obs['rgb']
depth = obs['depth']
instruction = obs['instruction']
pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # ma trận đơn vị
```

Pose **hardcode ma trận đơn vị** — agent không đòi pose thật từ simulator. Đầu vào chỉ cần
`rgb` + `depth` + `instruction` → có thể feed từ dataset tĩnh (open-loop trên Kaggle).
Đây là căn cứ cho kết luận "chạy full dual-system open-loop không cần sim" trong
`eval_plan_kaggle_s2.md` §3.1.

### PL-C6 📖 `get_policy` — tên policy đúng là `'InternVLAN1_Policy'`

`internnav/model/__init__.py` dòng 18–24:

```python
elif policy_name == 'InternVLAN1_Policy':
    from .basemodel.internvla_n1.internvla_n1_policy import (
        InternVLAN1ModelConfig,
        InternVLAN1Net,
    )
    return InternVLAN1Net
```

→ Trả lời câu hỏi mở trong `eval_plan_kaggle_s2.md` §8 ("tên policy đúng cho NavDP là gì").
Cùng file có `get_config('InternVLAN1_Policy')` → `InternVLAN1ModelConfig` (dòng 50–56).

---

## Nhóm D — Dữ liệu `InternRobotics/InternData-N1`

### PL-D1 ✅ Cấu trúc gốc dataset: đúng 3 thư mục, 2 trong số đó là `.tar.gz` (đo 21/07/2026)

- Quét `list_repo_files` → **20 829 file**, gốc chỉ có `vln_ce/`, `vln_n1/`, `vln_pe/`
  (không có `preview/` như tài liệu cũ nói — PL-A4).
- `vln_pe/traj_data/` = **file rời** chuẩn LeRobot (122 scene, mỗi scene có `meta/` riêng, 5 193 parquet).
- `vln_n1/traj_data/` = **3 774 file `.tar.gz`** (6 simulator × 2 camera). LeRobot nằm *bên trong* archive.
- `vln_ce/traj_data/` = `.tar.gz` theo scene (r2r 61 · rxr 59 · scalevln 794).

> Nguồn: `SETUP_NOTES.md` 3.10, 3.11, 3.14.

### PL-D2 ✅ Đo `vln_pe/traj_data/r2r_aliengo/17DRP5sb8fy` (21/07/2026) — `info.json` mô tả SAI chính nó

Parquet thật có 14 cột: `observation.camera_position/orientation/yaw`,
`observation.robot_position/orientation/yaw`, `observation.progress`, `observation.step`,
`observation.action` (**int64 rời rạc**, thấy giá trị 3, 3, 1), + đủ 5 trường LeRobot chuẩn
(`timestamp/frame_index/episode_index/index/task_index`).

**Ba mâu thuẫn khai báo ↔ thực tế:**

1. `info.json.features` khai `camera_intrinsic (3,3)` / `camera_extrinsic (4,4)` / `action float32 (4,4)`
   — **parquet không có cột nào trong đó**. Nguyên nhân đã truy ra: features là **template của `vln_n1`
   copy sang mà không sửa** (đối chiếu được vì `vln_n1` khai đúng 3 feature đó và parquet của nó có đúng
   3 cột đó — PL-D3).
2. `fps: 30` nhưng `timestamp` bước đều 0.166667 s → **6 Hz thật** (tính giờ bằng `frame/fps` sẽ sai 5×).
3. `splits: {"train": "0:1"}` nhưng `total_episodes: 23` → trường `splits` vô dụng.

> Nguồn: `SETUP_NOTES.md` 3.13; `docs/data_contract.md` mục 3.

### PL-D3 ✅ Đo `vln_n1/traj_data/matterport3d_d435i/pLe4wQe7qrG.tar.gz` (248.7 MB, 21/07/2026)

- 13 episode / 1 811 frame. Parquet đúng **4 cột**: `index`,
  `observation.camera_intrinsic` (3×3, hằng số), `observation.camera_extrinsic` (4×4),
  `action` (**4×4 SE(3) liên tục** — quỹ đạo camera). `info.json` **khớp** parquet.
- **Intrinsic đo được:** `fx=355.81, fy=351.69, cx=240, cy=135` → ảnh **480×270**, FOV 68°×42°
  → khớp spec Intel RealSense **D435i (69°×42°)**. Hậu tố `_d435i`/`_zed` là camera thật.
  ⬜ chưa đo biến thể `_zed`.
- ⚠️ **Thiếu** `timestamp`, `frame_index`, `episode_index`, `task_index` (trường bắt buộc LeRobot v2.1)
  → loader LeRobot gốc không đọc được, chỉ loader riêng của InternNav đọc được.
- Video dạng `.mp4`.

> Nguồn: `SETUP_NOTES.md` 3.14, 3.14.b; `docs/data_contract.md` mục 2.

### PL-D4 ✅ Đo `vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz` (16.16 MB — file nhỏ nhất repo, 22/07/2026)

- 3 episode / 16 frame (scene nhỏ). Parquet **21 cột** =
  `action` (int32, `{1:↑, 2:←, 3:→, 5:↓}`, `-1` = frame start; đo được
  `[-1,2,2,2,2,2,2,2,2,2,1,1,2,1,2,1]`)
  + 5 setting × {`pose.{s}` (4×4), `goal.{s}` (pixel `[u,v]`, `(-1,-1)` = không có; đo được
  `[230,372] [280,356] [275,392]`), `relative_goal_frame_id.{s}`}
  + đủ 5 trường LeRobot chuẩn.
- **5 setting camera:** `125cm_0deg`, `125cm_30deg`, `125cm_45deg`, `60cm_15deg`, `60cm_30deg`
  (`{setting} = f'{height}cm_{pitch_2}deg'` — ghép ở `internvla_n1_lerobot_dataset.py`).
  🚨 Đo được **`goal` toàn `(-1,-1)` ở setting `125cm_0deg`** (camera không cúi → không thấy sàn
  để chấm pixel) — setting này cũng **không có** trong `train_system2.sh`.
- **RGB + depth CÓ SẴN** dạng PNG từng frame trong `videos/`:
  RGB **640×480 uint8**; depth **640×480 uint16 (mode `I;16`), đơn vị milimét, clip 10000 (=10 m)**.
- `meta/episodes.jsonl`: `tasks[0].split("<INSTRUCTION_SEP>")` → nhiều instruction/episode;
  `length` = số dòng parquet (loader có `assert`). `meta/tasks.jsonl`: câu lệnh tiếng Anh tự nhiên.
- `info.json` **khớp** parquet (lỗi PL-D2 chỉ cục bộ ở `vln_pe`). Lỗi nhỏ: `video_path` ghi `.mp4`
  nhưng file thật `.png`; `total_videos: 0`.
- ⬜ Trên scene rxr nhỏ này `relative_goal_frame_id` toàn `-1` dù `goal` có giá trị ở vài frame —
  chưa kết luận được, cần kiểm thêm 1 scene r2r lớn hơn.

> Nguồn: `docs/data_contract.md` mục 4.b; `SETUP_NOTES.md` (bổ sung 22/07).

### PL-D5 📖 Bằng chứng "subset nào nuôi system nào" — từ loader + train script (xác minh local 23/07/2026)

1. **`vln_n1` → System 1 (NavDP):** loader `internnav/dataset/navdp_lerobot_dataset.py` đọc đúng
   3 cột đặc trưng (`camera_intrinsic` reshape(3,3), `camera_extrinsic` reshape(4,4), `action`
   reshape(-1,4,4)); train config `scripts/train/base_train/configs/navdp.py` khai
   `root_dir='data/datasets/InternData-N1/vln_n1/traj_data'` (đọc local 23/07, dòng ~54).
2. **`vln_ce` → System 2 (VLM):** loader `internnav/dataset/internvla_n1_lerobot_dataset.py` ghép
   `setting = f'{height}cm_{pitch_2}deg'` và đọc `pose.{setting}` / `goal.{setting}` /
   `relative_goal_frame_id.{setting}`; registry cùng file trỏ `data_path: traj_data/{r2r,rxr,scalevln}`;
   train script `scripts/train/qwenvl_train/train_system2.sh` (đọc local 23/07) khai
   `llm=Qwen/Qwen2.5-VL-7B-Instruct` và
   `vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_...`
   (scalevln bị comment), chạy SLURM `-N 8 --gres=gpu:8` = 64 GPU,
   `run_name=InternVLA-N1-System2`.
3. **`vln_pe` → baseline CMA/RDP:** loader `cma_lerobot_dataset.py` đọc bộ cột
   `position/orientation/yaw/progress/step` — đúng từ vựng của parquet `vln_pe` (PL-D2).

> Nguồn: `docs/vln_subsets_architecture.md` mục 2 (số dòng GitHub 22/07) + xác minh local 23/07/2026.

---

## Nhóm E — Chạy thử System 2 thật

### PL-E1 ✅ Chạy `generate()` S2 thuần trên T4×2 (22/07/2026) — model trả `←←←←`, CHƯA bật pixel-goal

| Lần | Prompt | Ảnh | Output thô | Nhánh parser |
|---|---|---|---|---|
| 1 | tự dựng (chỉ instruction) | 1 RGB `vln_ce` (setting có pitch), **chưa resize 384** | `←←←←` | action |
| 2 | prompt thật của repo (xin coordinates) | như trên | `←←←←` | action (không có số → `pixel_goal` rỗng) |

**Ý nghĩa:** dù prompt yêu cầu tường minh "output the next waypoint's coordinates", model vẫn trả
chuỗi mũi tên → rơi nhánh action → **không sinh `output_latent`** → không có gì feed System 1.
Đây là blocker trực tiếp cho pipeline dual-system (S1 chỉ chạy khi S2 ra latent — xem
[02_code_structure.md](02_code_structure.md) mục 5).

**Nghi phạm, xếp theo khả năng (chưa loại trừ được cái nào):**
1. Chưa `resize((384, 384))` trước khi đưa vào processor (số config thật: `resize_w/h=384`).
2. Thiếu câu dẫn `"you can see <image>."` (conjunction[0]) nối cuối prompt.
3. Chế độ pixel cần **full policy** (`generate_latents`, token 151667) thay vì class HF thuần.

⬜ Việc tiếp theo đã ghi trong `io_system2.md` mục 5: chạy lại đúng snippet chuẩn (có resize +
conjunction) trước khi kết luận.

> Nguồn: `docs/io_system2.md` mục 4.a.

---

## Chỉ mục tra ngược (file nào dùng bằng chứng nào)

| Bằng chứng | Được dùng ở |
|---|---|
| PL-A1…A6 | [01_system_requirements](01_system_requirements.md) |
| PL-B1, B2, B3 | [01](01_system_requirements.md) (VRAM/disk), [04](04_checkpoint_details.md) (kích thước, 7B) |
| PL-B4, B5, B6 | [04_checkpoint_details](04_checkpoint_details.md) |
| PL-C1, C2, C3 | [02_code_structure](02_code_structure.md) mục 1–2, [01](01_system_requirements.md) (override), [04](04_checkpoint_details.md) (chọn checkpoint) |
| PL-C4, C5, C6 | [02_code_structure](02_code_structure.md) mục 3–6 |
| PL-D1…D5 | [03_data_contract](03_data_contract.md) |
| PL-E1 | [02](02_code_structure.md) mục 5, [03](03_data_contract.md) (giới hạn eval) |
