# SETUP_NOTES — Người B (Cloud & Eval)

Nhật ký môi trường + lỗi gặp phải & cách fix. Ghi ngay lúc gặp, không để cuối ngày.
Các mục đánh dấu 🔧 **PR candidate** là chỗ docs repo InternNav thiếu/sai → ứng viên cho PR.

---

## 1. Baseline môi trường Kaggle (session CPU) — 21/07/2026

Lệnh chạy:

```bash
!nvidia-smi || echo "CPU session - dung"
!df -h /kaggle/working /kaggle/temp
!free -g
!python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Output thô:

```
/bin/bash: line 1: nvidia-smi: command not found
CPU session - dung
df: /kaggle/temp: No such file or directory
Filesystem      Size  Used Avail Use% Mounted on
/dev/loop1       20G   72K   20G   1% /kaggle/working
               total        used        free      shared  buff/cache   available
Mem:              31           0          27           0           3          30
Swap:              0           0           0
2.10.0+cpu 5.0.0
```

Tổng hợp:

| Hạng mục | Giá trị | Ghi chú |
|---|---|---|
| Accelerator | None (CPU) | Đúng chủ đích — session hôm nay chỉ để tải & đóng gói, không đốt quota GPU |
| `/kaggle/working` | **20GB**, đang trống 20GB | Trần cứng cho toàn bộ output |
| `/kaggle/temp` | **KHÔNG TỒN TẠI** | Xem mục 2.1 |
| RAM | 31GB (còn trống 27GB) | Thoải mái so với laptop 12GB |
| Swap | **0** | Không có lưới an toàn — OOM là process chết ngay, xem 2.5 |
| torch | 2.10.0+**cpu** | Bản CPU-only, đúng với session None — xem 2.6 |
| transformers | **5.0.0** | Rủi ro tương thích, xem 2.2 và 2.3 |

**Kết luận cho ngân sách disk:** 20GB là trần. Checkpoint `InternVLA-N1` đo được **16.79GB**
(không phải ~15GB như ước lượng ban đầu) → chỉ còn **~3.2GB** dư. Bất kỳ thứ gì ghi trùng lặp
(cache HF, file `.bin` trùng `.safetensors`) đều làm tràn. Chi tiết ngân sách ở 3.3.

---

## 2. Lỗi & phát hiện

### 2.1. 🔧 `/kaggle/temp` không tồn tại — kế hoạch đặt `HF_HOME` vào đó bị hỏng

**Triệu chứng:** `df: /kaggle/temp: No such file or directory`

**Vì sao đây là vấn đề:** kế hoạch Ngày 1 dựa vào việc trỏ `HF_HOME=/kaggle/temp/hf_cache`
để cache của HuggingFace **không** ăn vào trần 20GB của `/kaggle/working`. Lý do: `hf download
--local-dir X` vẫn ghi một bản vào cache rồi mới đưa sang `X` → 15GB thành 30GB → tràn ở
khoảng 85% với `OSError: [Errno 28] No space left on device`.

Nếu `/kaggle/temp` không có sẵn thì phải tìm chỗ khác cho cache, **trước khi bắt đầu tải**.

**Cần xác minh (chạy ngay, trước khi tải bất cứ thứ gì):**

```bash
!df -h / /tmp /kaggle
!ls -la /kaggle
```

**Cách fix theo thứ tự ưu tiên:**

1. Tự tạo thư mục — nhiều khả năng `/kaggle` ghi được, chỉ là `temp` không được tạo sẵn:
   ```python
   import os
   os.makedirs("/kaggle/temp/hf_cache", exist_ok=True)
   os.environ["HF_HOME"] = "/kaggle/temp/hf_cache"
   ```
2. Không tạo được → dùng `/tmp` (nằm trên root fs của container, không tính vào output):
   ```python
   os.environ["HF_HOME"] = "/tmp/hf_cache"
   ```
   Điều kiện: `df -h /` phải cho thấy còn ≥ 20GB trống. **Kiểm tra trước, đừng giả định.**
3. Root fs cũng chật → bỏ hẳn cơ chế cache, tải thẳng, không nhân đôi:
   ```bash
   !hf download <repo> --local-dir /kaggle/working/ckpt/s2 --cache-dir /tmp/hfcache
   ```
   Hoặc dùng `snapshot_download(..., local_dir=..., cache_dir=...)` trong Python.

**Kiểm chứng đã fix xong (bắt buộc, đừng tin là nó chạy):**
```bash
!du -sh /kaggle/working $HF_HOME
```
Nếu `/kaggle/working` phình đúng bằng ~2× kích thước file tải → cache vẫn nằm sai chỗ, chưa fix được.

✅ **Kết quả 21/07:** sau khi tải 16.79GB, `/kaggle/working` dùng đúng ~16G (không phải ~32G)
→ fix có tác dụng. (Bản gốc mục này ghi "tải thử `preview/` 59.5MB để kiểm chứng" — `preview/`
**không tồn tại**, xem 3.10.)

🔧 **PR candidate:** docs của InternNav hướng dẫn chạy trên Kaggle/Colab nhưng không nhắc gì tới
việc `HF_HOME` mặc định nằm trong vùng bị giới hạn dung lượng, trong khi checkpoint của repo này
là 15–17GB — sát trần 20GB của Kaggle. Đây là lỗi mà bất kỳ ai làm theo docs đều gặp.

---

### 2.2. ⚠️ `transformers 5.0.0` — rủi ro tương thích cao, chưa gặp lỗi nhưng phải phòng

**Phát hiện:** Kaggle preinstall `transformers==5.0.0`.

**Vì sao đáng lo:** kế hoạch dự đoán rủi ro là *transformers quá cũ* (`Qwen2_5_VLForConditionalGeneration`
không tồn tại vì bản < 4.49). Thực tế **ngược lại** — bản này quá mới. Repo InternNav phát hành trong
thời kỳ `transformers` 4.x, và 5.0 là major version → có breaking change. Rủi ro cụ thể:

- Code `trust_remote_code=True` của repo gọi API 4.x đã bị bỏ/đổi tên ở 5.x
- `AutoProcessor` / vision processor của Qwen2.5-VL đổi signature
- `requirements` của repo có thể pin `transformers<5` → `pip install -e .` sẽ **hạ cấp** bản này,
  kéo theo xung đột với `torch` preinstall

> ✅ **KẾT QUẢ 21/07 — 5.0.0 load S2 BÌNH THƯỜNG.** Không cần pin về 4.x. Rủi ro dự đoán ở trên
> **không xảy ra**: `Qwen2_5_VLForConditionalGeneration` có sẵn native, tokenizer nạp được, vision
> processor không lỗi. Lỗi thật gặp phải là **chọn nhầm Auto class** (xem 2.3), không liên quan version.
>
> Chỉ còn **một** breaking change 5.x thực sự chạm vào: `torch_dtype=` → `dtype=`.
>
> ⬜ **Vẫn phải cảnh giác ở Ngày 3–4** khi `pip install -e .` repo InternNav: nếu repo pin
> `transformers<5` thì lệnh cài sẽ **hạ cấp** bản này và kéo theo xung đột với `torch`. Đó là rủi ro
> chưa được kiểm chứng.

**Chưa xử lý gì lúc này.** Đúng nguyên tắc: chưa lỗi thì chưa sửa. Nhưng khi smoke-test load model
(Khối 4) mà gặp `AttributeError` / `ImportError` / `TypeError` về signature → **nghi ngờ chỗ này đầu tiên**,
đừng đi đào chỗ khác.

**Phương án nếu lỗi:** kiểm tra repo yêu cầu bản nào rồi pin đúng bản đó
```bash
!pip install -q "transformers==4.51.3" accelerate qwen-vl-utils
```
(đọc `requirements/*.txt` hoặc `pyproject.toml` của InternNav để lấy con số thật, không đoán).
Cài xong **phải restart kernel** thì Python mới nạp bản mới.

🔧 **PR candidate tiềm năng:** nếu repo không pin trần trên cho `transformers`, đây là chỗ nên đề xuất
`transformers>=4.49,<5`.

---

### 2.3. ❌ `AutoModelForCausalLM` không load được Qwen2.5-VL — sai Auto class

**Triệu chứng:**
```
ValueError: Unrecognized configuration class Qwen2_5_VLConfig for this kind of AutoModel:
AutoModelForCausalLM. Model type should be one of GPT2Config, ... Qwen2Config, Qwen3Config, ...
```
Kèm cảnh báo `[transformers] torch_dtype is deprecated! Use dtype instead!`

**Nguyên nhân:** `AutoModelForCausalLM` chỉ map các kiến trúc **text thuần**. Qwen2.5-VL là
vision-language → không nằm trong bảng mapping đó. Danh sách lỗi in ra có `Qwen2Config`/`Qwen3Config`
nhưng **không có** `Qwen2_5_VLConfig` — đọc kỹ danh sách là thấy ngay.

**Cách fix:**
```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
print(m.hf_device_map)
```
Hoặc `AutoModelForImageTextToText` nếu muốn giữ tính tổng quát (5.x đổi tên từ `AutoModelForVision2Seq`).

**Ba điểm đi kèm:**
1. `torch_dtype=` **đã deprecated** ở `transformers` 5.x → dùng `dtype=`
2. **Bỏ `trust_remote_code=True`** — Qwen2.5-VL được hỗ trợ *native*, bằng chứng là traceback in ra
   `transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig`, tức class có sẵn
   trong thư viện. Bật cờ này chỉ thêm rủi ro thực thi code lạ, không giải quyết gì.
3. Dự đoán ở kế hoạch ("`Qwen2_5_VLForConditionalGeneration` không tồn tại → `transformers` quá cũ")
   **không đúng với môi trường này**. Bản 5.0.0 có đủ class; lỗi thật là chọn nhầm Auto class.
   Xem thêm 2.2.

🔧 **PR candidate:** nếu docs/README của InternNav có ví dụ load bằng `AutoModelForCausalLM` hoặc
`torch_dtype=`, đó là chỗ cần sửa cho `transformers` 5.x.

---

### 2.4. ✅ Smoke test S2 trên T4 x2 — ĐẠT, và lộ ra chỗ System 1 nằm trong checkpoint

**Kết quả đo (21/07, notebook `internnav-s2-smoke`, T4 x2):**

| Hạng mục | Giá trị |
|---|---|
| Params nạp được | **8.289 B** → đúng Qwen2.5-VL-7B, khớp 3.9 |
| GPU0 | 7.57 GB (`model.visual` + layer 0–10) |
| GPU1 | 9.01 GB (layer 11–27 + `norm` + `lm_head`) |
| **Tổng** | **16.58 GB** — khớp dự tính 16.6GB weights bf16 |
| Dư trên T4 x2 (30GB) | ~13.4 GB cho KV-cache + vision token |

`device_map="auto"` shard **đúng qua 2 GPU**, không dồn vào `cuda:0` → tránh được nguyên nhân OOM số 1.
Không có dòng `MISSING` nào → phần S2 nạp đủ, checkpoint lành.

**→ Con số cho mail xin server: đây là số ĐO ĐƯỢC, không phải ước lượng.** 16.58GB weights, cần
≥24GB để chạy thoải mái ở bf16 với KV-cache.

#### Toàn bộ System 1 bị bỏ qua khi load bằng class thuần của `transformers`

Log in ra ~120 dòng `UNEXPECTED`, tất cả cùng tiền tố:

```
model.language_model.navdp.rgbd_encoder.rgb_model.*      ← encoder RGB, 12 block
model.language_model.navdp.rgbd_encoder.depth_model.*    ← encoder Depth, 12 block
model.language_model.navdp.rgbd_encoder.former_net.*     ← 2 layer
model.language_model.navdp.decoder.layers.{0...15}.*     ← diffusion decoder, 16 layer
model.language_model.navdp.action_head.* / critic_head.*
model.language_model.navdp.goal_compressor.*
model.language_model.latent_queries                      ← cau noi S2 -> S1
```

`UNEXPECTED` = tensor **có trong checkpoint nhưng không có trong kiến trúc đang dựng** → bỏ qua im lặng.
`Qwen2_5_VLForConditionalGeneration` chỉ biết phần Qwen2.5-VL, nên `navdp` (= **NavDP = System 1**,
diffusion policy) rơi hết ra ngoài. **Không phải lỗi** với mục tiêu Khối 4 (smoke-test S2).

**✅ Chứng minh suy luận ở 3.2:** từng suy "chênh 0.196GB ở shard 4 gần như chắc chắn là System 1".
Giờ có tên tensor cụ thể, và số học khớp: nạp được 16.58GB (S2) / tổng checkpoint 16.79GB.

**⚠️ Hệ quả cho Ngày 4 — ghi ngay, đừng để hôm đó mới biết:** muốn chạy **dual-system** thì
**không dùng `transformers` thuần được**. Phải cài repo InternNav và dùng model class của nó
(`internnav/model/...`). Load bằng class HF chỉ ra được System 2.

**Chi tiết kiến trúc đáng ghi:** `navdp` nằm **lồng trong** `model.language_model.navdp`, không phải
nhánh song song ở cấp `model` → InternNav gắn System 1 vào bên trong module ngôn ngữ.

**Điểm sync với Người A:** `model.language_model.latent_queries` chính là hiện thân của *latent plan*
mà kế hoạch tuần (dòng 112) giao Người A tìm hiểu trong `internvla_n1_agent.py`. Mang ra buổi sync —
mình có bằng chứng từ checkpoint, Người A có code.

---

### 2.5. Swap = 0 — không có lưới an toàn khi hết RAM

**Phát hiện:** `Swap: 0 0 0`.

**Ý nghĩa:** khác với laptop (kế hoạch cấu hình WSL2 có `swap=16GB`), trên Kaggle khi chạm trần RAM
thì process **bị kill ngay**, không có giai đoạn chậm dần để kịp phản ứng. Triệu chứng điển hình là
kernel chết im lặng hoặc `^C` không rõ lý do — dễ bị chẩn đoán nhầm thành lỗi mạng hay lỗi code.

31GB RAM là dư dả cho việc hôm nay. Chỉ cần lưu ý ở bước load model bằng `torch_dtype=torch.bfloat16`
+ `device_map="auto"` — có giai đoạn weights đi qua RAM trước khi lên VRAM.

---

### 2.6. `torch` là bản CPU-only

`2.10.0+cpu` — đúng như mong đợi với Accelerator = None, không phải lỗi. Ghi lại để khỏi hoảng.
Khi đổi sang session GPU, Kaggle cấp image khác có `torch` bản CUDA; **`transformers` khi đó cũng có
thể là version khác** → phải chạy lại cell baseline ở mục 1 cho session GPU, đừng giả định giống nhau.

---

## 3. Survey dung lượng thật (bước A.2) — 21/07/2026

Output thô:

```
=== InternRobotics/InternVLA-N1-System2 | TONG 16.59 GB | 18 file ===
    4.991 GB  model-00002-of-00004.safetensors
    4.968 GB  model-00001-of-00004.safetensors
    4.933 GB  model-00003-of-00004.safetensors
    1.692 GB  model-00004-of-00004.safetensors

=== InternRobotics/InternVLA-N1 | TONG 16.79 GB | 18 file ===
    4.991 GB  model-00002-of-00004.safetensors
    4.966 GB  model-00001-of-00004.safetensors
    4.933 GB  model-00003-of-00004.safetensors
    1.888 GB  model-00004-of-00004.safetensors

=== InternRobotics/VLN-PE | TONG 1.86 GB | 25 file ===
    0.565 GB  r2r/fine_tuned/rdp/pytorch_model.bin
    0.369 GB  r2r/fine_tuned/rdp/optimizer.pt
    0.148 GB  r2r/fine_tuned/cma/pytorch_model.bin
    0.148 GB  r2r/zero_shot/cma/pytorch_model.bin
    0.148 GB  r2r/fine_tuned/cma_plus/pytorch_model.bin
    0.133 GB  r2r/fine_tuned/seq2seq/pytorch_model.bin
    0.133 GB  r2r/fine_tuned/seq2seq_plus/pytorch_model.bin
    0.133 GB  r2r/zero_shot/seq2seq/pytorch_model.bin
```

### 3.1. 🚨 Base model là **7B, KHÔNG phải 3B** — kế hoạch tuần sai giả định (ĐÃ XÁC NHẬN, xem 3.9)

**Suy luận từ dung lượng** (chưa cần tải một byte weights nào):

| Giả thuyết | Kích thước bf16 dự kiến | Thực tế đo được |
|---|---|---|
| Qwen2.5-VL-**3B** | ~7.5 GB | ✗ lệch hơn 2× |
| Qwen2.5-VL-**7B** (8.29B params) | **~16.6 GB** | ✓ **16.59 GB — khớp** |

Weights bf16 = 2 byte/tham số → 16.59 GB ÷ 2 ≈ **8.3 tỷ tham số**. Đây đúng là Qwen2.5-VL-7B.

**Hệ quả — phải xử lý ngay, đừng để đến Ngày 3:**

1. **Kế hoạch tuần ghi "Qwen2.5-VL 3B, ~6-7GB weights" → con số này sai.** Mọi tính toán VRAM dựa
   trên nó đều phải làm lại.
2. **Chạy trên T4 15GB là không khả thi ở bf16.** Riêng weights đã 16.6GB > 15GB VRAM, chưa tính
   KV-cache và vision token. Kaggle T4 x2 (2×15GB) thì cần shard qua 2 GPU (`device_map="auto"`),
   hoặc P100 16GB cũng vẫn không đủ chỗ trống. Bảng rủi ro của kế hoạch ghi "S2 OOM trên T4 15GB"
   như một *khả năng* — thực tế nó là **điều chắc chắn**.
3. **Yêu cầu server phải sửa:** đã xin "≥16GB VRAM" trong mail Khối 0.1 — con số đó **không còn đủ**.
   Phải xin **≥24GB** (A10/A30/L4/3090/4090) để chạy thoải mái ở bf16, và nếu W4 fine-tune thì
   cần hơn nữa. **Gửi mail bổ sung cho anh Huy ngay hôm nay**, đừng chờ.

**Cần xác nhận bằng bước 1.5** (đọc `config.json`, vài KB, 5 giây) — tìm `hidden_size: 3584` và
`num_hidden_layers: 28` là chốt 7B. Suy luận từ dung lượng đã rất chắc, nhưng báo mentor thì
phải có bằng chứng trực tiếp. → **Đã chạy, đã xác nhận: mục 3.9.**

### 3.2. 💡 `InternVLA-N1` chỉ lớn hơn `-System2` **0.2 GB** → nên tải bản đầy đủ

So sánh từng shard:

| Shard | `-System2` | `InternVLA-N1` | Chênh |
|---|---|---|---|
| 1 | 4.968 | 4.966 | ~0 |
| 2 | 4.991 | 4.991 | 0 |
| 3 | 4.933 | 4.933 | 0 |
| 4 | 1.692 | **1.888** | **+0.196** |
| **Tổng** | **16.59** | **16.79** | **+0.20 GB** |

Shard 1–3 giống hệt nhau → hai repo dùng chung phần backbone VLM. Chênh lệch 0.2GB nằm trọn ở
shard 4, gần như chắc chắn là **phần System 1** (diffusion policy head) được gộp vào bản dual-system.

**Kết luận thực tế:** tải `InternVLA-N1` thay vì `-System2` chỉ tốn thêm **0.2GB (+1.2%)** nhưng có
**cả hai system**. Trong khi Ngày 4 của kế hoạch là ghép dual-system → nếu hôm nay tải `-System2`
thì Ngày 4 phải tải lại 16.79GB từ đầu, mất thêm một session.

→ **Quyết định: đổi sang tải `InternRobotics/InternVLA-N1`.** Kế hoạch gốc chọn `-System2` với lý do
"nhẹ hơn, tiết kiệm 2GB" — số liệu thật cho thấy lý do đó không tồn tại.

### 3.3. ⚠️ Ngân sách disk giờ rất chật — phải tách notebook

Trần `/kaggle/working` = **20 GB**. Nếu nhét hết vào một chỗ:

| Món | GB |
|---|---|
| `InternVLA-N1` | 16.79 |
| `VLN-PE` (sau `--exclude`) | 1.42 |
| Mẫu LeRobot `vln_pe` (`meta/` + 1 parquet) | ~0.01 |
| **Tổng** | **18.22** |

> ⚠️ **Cập nhật 21/07:** dòng `InternData-N1/preview` cũ đã bỏ (không tồn tại, 3.10).
> Thêm một khoản **không** tính vào bảng này: 1 tar.gz của `vln_n1` = **249MB**, và giải nén ra
> còn phồng thêm. **Bắt buộc để ở `/kaggle/temp`, không được vào `/kaggle/working`** — nếu không
> sẽ tràn và mất luôn checkpoint 16.79GB. Dọn `/kaggle/temp/n1x` + `/kaggle/temp/n1dl` trước khi
> Save Version.

Còn dư **1.78GB** — quá mỏng. Chỉ cần cache HF rơi nhầm chỗ, hoặc file tạm lúc đóng gói, là tràn.
Kế hoạch cũ tính "15GB → dư 5GB" là dựa trên con số ước lượng sai.

**Xử lý: tách làm 2 notebook / 2 dataset riêng.**

- Notebook A — chỉ `InternVLA-N1` (16.79GB, dư 3.2GB để thao tác) ✅ **đã làm**, dataset
  `tieulam/internvla-n1-ckpt`
- Notebook B — `VLN-PE` + mẫu LeRobot (~1.43GB, rất nhẹ, chạy nhanh)

Tách ra còn có lợi khác: notebook B xong trong ~5 phút → có ngay artifact để làm việc, không phải
chờ 16.79GB.

**Tiết kiệm thêm:** `VLN-PE` có `optimizer.pt` (0.369GB) — đây là optimizer state chỉ dùng khi
*train tiếp*, inference không cần. Loại ra:
```bash
!hf download InternRobotics/VLN-PE --local-dir /kaggle/working/ckpt/vln_pe \
    --exclude "*optimizer.pt"
```

### 3.5. Còn phải kiểm tra: 18 file nhưng chỉ thấy 4

Survey chỉ in file > 50MB. 14 file còn lại là config/tokenizer — nhỏ nhưng **thiếu một cái là hỏng**.
Trước khi đóng gói phải xác nhận có đủ, đặc biệt:

- `config.json`
- `model.safetensors.index.json` — thiếu thì không load được shard
- `preprocessor_config.json` — **thiếu là hỏng toàn bộ phần vision**, và lỗi chỉ lộ ra lúc load, rất khó truy
- `tokenizer.json` / `tokenizer_config.json`

Xem đầy đủ bằng cách hạ ngưỡng: `survey("InternRobotics/InternVLA-N1", min_gb=0)`. → đã chạy, kết quả ở 3.6.

### 3.6. Kết quả `min_gb=0` — đủ file thiết yếu, nhưng có một khác biệt then chốt

**Cả hai repo đều có đủ** file bắt buộc: `config.json`, `model.safetensors.index.json`,
`preprocessor_config.json` (vision — quan trọng nhất), `chat_template.json`, `generation_config.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`. Không thiếu gì gây hỏng load.

**Nhưng khác nhau ở tokenizer — và đây là lý do thứ hai để chọn `InternVLA-N1`:**

| File | `-System2` | `InternVLA-N1` |
|---|---|---|
| `tokenizer.json` (fast tokenizer) | ❌ **KHÔNG CÓ** | ✓ 0.007 GB |
| `vocab.json` + `merges.txt` | ✓ | ✓ |
| `training_args.bin` | ✓ (pickle) | ❌ không có |

`-System2` chỉ có `vocab.json` + `merges.txt` — tức **tokenizer dạng "slow"**, phải để `transformers`
tự convert sang fast lúc load. Bình thường thì được, nhưng môi trường này là **`transformers` 5.0.0**
(xem 2.2), mà dòng 5.x siết chặt/loại bỏ đường slow-tokenizer. Rủi ro thật:

- `AutoTokenizer.from_pretrained(...)` báo lỗi hoặc đòi `use_fast=False` (mà đường đó có thể đã bị bỏ)
- Hoặc convert được nhưng chậm, và tệ hơn: **im lặng cho ra tokenization khác** bản gốc

`InternVLA-N1` có sẵn `tokenizer.json` → nạp thẳng, không qua bước convert, không có rủi ro này.

**Cộng dồn lý do chọn `InternVLA-N1` thay `-System2`:**
1. Chỉ tốn thêm 0.2GB nhưng có **cả System 1 + System 2** (mục 3.2)
2. Có sẵn **fast tokenizer** → tránh rủi ro với `transformers` 5.0.0 (mục này)
3. Ngày 4 cần dual-system → không phải tải lại 16.79GB lần nữa

→ **Chốt: tải `InternRobotics/InternVLA-N1`.** Không đụng `-System2` tuần này.

**Ghi chú bảo mật nhỏ:** `-System2` có `training_args.bin` — file pickle. `torch.load` từ 2.6+ mặc định
`weights_only=True` nên sẽ từ chối load. **Đừng tắt cờ đó để "cho nó chạy"** — file này là rác của
quá trình train, inference không cần. Bỏ qua.

### 3.7. 💡 `trainer_state.json` — cách xác nhận 7B mà không cần tải weights

Cả hai repo có `trainer_state.json` (4–5MB). File này là log của HuggingFace `Trainer`, thường chứa
`num_train_epochs`, số step, log history, và **đôi khi cả đường dẫn base model**. Tải riêng nó rất rẻ:

```python
from huggingface_hub import hf_hub_download
import json
p = hf_hub_download("InternRobotics/InternVLA-N1", "trainer_state.json")
st = json.load(open(p))
print({k: v for k, v in st.items() if k != "log_history"})
print("so log entry:", len(st.get("log_history", [])))
```

Kết hợp với `config.json` (bước 1.5) là đủ bằng chứng vững để báo mentor về 7B, và biết thêm model
được fine-tune bao nhiêu step — thông tin có ích cho W4 khi chính mình fine-tune.

Việc hai repo đều còn `trainer_state.json`/`training_args.bin` cho thấy checkpoint được upload thẳng
từ thư mục save của `Trainer`, chưa dọn. Không sai, chỉ là dấu hiệu nên đọc `config.json` cẩn thận
thay vì tin README.

### 3.8. `VLN-PE`: loại được ~0.44GB rác train, nhưng 4 thư mục thiếu `config.json`

**Rác của quá trình train, inference không cần** — tổng ~0.44GB:
`optimizer.pt` (0.369 + 0.049 + 0.020), `scheduler.pt`, `rng_state.pth`, `trainer_state.json`

```bash
!hf download InternRobotics/VLN-PE --local-dir /kaggle/working/ckpt/vln_pe \
    --exclude "*optimizer.pt" "*scheduler.pt" "*rng_state.pth" "*trainer_state.json"
```
→ 1.86 GB còn **~1.42 GB**.

**⚠️ Bẫy phải biết trước khi dùng ở Ngày 4:** chỉ **3** thư mục có `config.json`
(`fine_tuned/rdp`, `fine_tuned/seq2seq`, `fine_tuned/cma`). Bốn thư mục sau **có weights nhưng KHÔNG có
`config.json`**:

- `r2r/zero_shot/cma/`
- `r2r/zero_shot/seq2seq/`
- `r2r/fine_tuned/cma_plus/`
- `r2r/fine_tuned/seq2seq_plus/`

→ `from_pretrained()` trỏ thẳng vào các thư mục này sẽ **fail**. Phải dựng config từ code repo hoặc
mượn `config.json` của biến thể tương ứng. Có `meta.yaml` ở root — đọc file này trước, nhiều khả năng
nó mô tả cách map checkpoint với config.

**Hệ quả cho phương án dự phòng Ngày 4:** bảng rủi ro nói "chạy eval với agent baseline nhẹ (`cma`/`rdp`)".
→ **Dùng `r2r/fine_tuned/cma` hoặc `r2r/fine_tuned/rdp`** vì chúng có đủ `config.json`. Tránh bản
`zero_shot` và `*_plus` cho lần chạy đầu — đang cần một thứ chạy được, không phải một thứ để debug.

🔧 **PR candidate:** thiếu `config.json` ở 4/7 thư mục checkpoint của `VLN-PE`, trong khi README không
nhắc gì tới việc phải tự cung cấp config. Người dùng làm theo docs sẽ gặp lỗi khó hiểu.

---

## 3.9. ✅ XÁC NHẬN: base model là Qwen2.5-VL-**7B** — bằng chứng trực tiếp từ `config.json`

Output bước 1.5:

```
InternRobotics/InternVLA-N1        -> qwen2_5_vl | None
   hidden: 3584 layers: 28
InternRobotics/InternVLA-N1-System2 -> qwen2_5_vl | None
   hidden: 3584 layers: 28
```

**Đối chiếu chữ ký kiến trúc chính thức của Qwen2.5-VL:**

| Biến thể | `hidden_size` | `num_hidden_layers` | Khớp? |
|---|---|---|---|
| Qwen2.5-VL-**3B** | 2048 | 36 | ✗ |
| Qwen2.5-VL-**7B** | **3584** | **28** | ✅ **khớp cả hai** |
| Qwen2.5-VL-72B | 8192 | 80 | ✗ |

**→ CHỐT: base model là Qwen2.5-VL-7B.** Hai nguồn bằng chứng độc lập cùng chỉ một kết luận:
1. Số học dung lượng: 16.59 GB ÷ 2 byte (bf16) ≈ 8.3B tham số (mục 3.1)
2. Chữ ký kiến trúc trong `config.json`: 3584 / 28 (mục này)

`model_type` = `qwen2_5_vl` cũng xác nhận đúng dòng Qwen2.5-VL (không phải Qwen2-VL — dòng cũ mới là
dòng có bản 2B). Kế hoạch tuần ghi "Qwen2.5-VL không có bản 2B, nhỏ nhất là 3B" — đúng, nhưng repo này
**thậm chí không dùng bản 3B**.

**`_name_or_path` = `None`:** bình thường, trường này bị xoá khi upload. Không phải lỗi, đừng đi tìm.

**Cả hai repo có kiến trúc VLM y hệt nhau** — củng cố kết luận ở 3.2: chúng dùng chung backbone,
chênh 0.2GB ở shard 4 là phần System 1 gộp thêm.

### Hệ quả cần hành động ngay

| Việc | Trạng thái |
|---|---|
| Sửa yêu cầu server: VRAM ≥16GB → **≥24GB** | ⬜ gửi mail bổ sung hôm nay — **giờ có số ĐO ĐƯỢC**, xem 2.4 |
| Báo Người A: 7B chứ không phải 3B | ⬜ sync cuối giờ |
| Sửa kế hoạch tuần (mục 1.2, 1.3, bảng rủi ro) | ✅ đã sửa |
| Smoke test T4 x2 xác nhận con số | ✅ **đã chạy** — 16.58GB thật, xem 2.4 |

**Tính lại VRAM cho đúng** (số cũ trong kế hoạch dựa trên 3B nên vô dụng):

| Hạng mục | bf16 | Nguồn |
|---|---|---|
| Weights | **16.58 GB** | ✅ **đo thật** trên T4 x2 (2.4) |
| KV-cache + vision token (chuỗi ảnh history) | +2–4 GB | ước lượng |
| **Tối thiểu thực tế** | **~19–21 GB** | |

- **T4 15GB đơn: không thể.** Weights một mình đã vượt.
- **P100 16GB: không thể.** Sát nút nhưng vẫn thiếu.
- **T4 x2 (2×15GB): được**, bắt buộc `device_map="auto"` để shard qua 2 GPU. Đây là cấu hình Kaggle
  duy nhất chạy nổi ở bf16 → **khi smoke test phải chọn T4 x2, không phải P100.**
- **4-bit (bitsandbytes): ~5–6 GB**, chạy được trên 1 GPU đơn — nhưng để dành làm phương án cuối
  theo đúng thang trong kế hoạch, vì nó ảnh hưởng chất lượng output.

---

## 3.10. 🔧 `InternData-N1/preview/` KHÔNG TỒN TẠI — cấu trúc thật khác hẳn tài liệu

**Triệu chứng:** lệnh theo kế hoạch (Phụ lục A.5, P1) chạy "thành công" nhưng không tải gì:

```bash
!hf download InternRobotics/InternData-N1 --repo-type dataset \
    --include "preview/*" --local-dir /kaggle/working/data
```
```
Fetching 0 files: 0it [00:00, ?it/s]
✓ Downloaded  path: /kaggle/working/data
```

**⚠️ Bẫy nghiêm trọng:** `hf download` với `--include` không khớp file nào **KHÔNG báo lỗi** — nó in
`✓ Downloaded` y như thành công. Dấu hiệu duy nhất là dòng `Fetching 0 files` và mọi số đo đều `0.00B`.
Không để ý là đi tiếp với thư mục rỗng, tới Ngày 5 mới phát hiện.

**→ Quy tắc bắt buộc từ giờ: đếm số file khớp TRƯỚC khi download.**
```python
import fnmatch
files = api.list_repo_files("InternRobotics/InternData-N1", repo_type="dataset")
m = fnmatch.filter(files, "<pattern>")
print(f"khop {len(m)} file"); assert m, "PATTERN KHONG KHOP - dung download"
```

**Cấu trúc thật ở cấp gốc** (không có `preview/`, không có `traj_data/` ở gốc):

```
vln_ce/      vln_n1/      vln_pe/      README.md    .gitattributes
```

Tổng **20 829 file**.

### Bẫy này đã nổ LẦN THỨ HAI trong ngày (21/07)

Lần 2: tải 1 tar.gz của `vln_n1` bằng `!hf download --include "<placeholder>"` — placeholder chưa
được thay bằng đường dẫn thật. Kết quả **y hệt**: `Fetching 0 files`, mọi số `0.00B`, `✓ Downloaded`.

**Nguyên nhân sâu hơn lần 1:** dòng `!` là shell, **không nội suy biến Python**. Viết
`--include "{SMALLEST}"` hay dán placeholder đều cho cùng một kết quả im lặng.

**Cách phòng triệt để — dùng API Python thay vì `!hf download` khi cần biến:**
```python
from huggingface_hub import hf_hub_download
tgz = hf_hub_download(REPO, filename=SMALLEST, repo_type="dataset", local_dir=DEST)
assert os.path.getsize(tgz) > 1e6
```
`hf_hub_download` **ném exception** khi sai tên file. `hf download --include` thì không. Đây là
khác biệt quan trọng: một bên fail-fast, một bên fail-silent.

🔧 **PR candidate (mạnh):** tài liệu hướng dẫn `--include "preview/*"` cho một đường dẫn không tồn tại.
Cùng loại với các file không tồn tại đã nêu ở mục 6 của kế hoạch tuần (`challenge_cfg.py`,
`requirements/train.txt`...). Đây là ứng viên tốt cho PR fix docs.

---

## 3.11. 🚨 LeRobot format trong `vln_pe/` — kế hoạch bảo "không đụng `vln_pe`" là SAI với luồng Người B

> ⚠️ **CẬP NHẬT 21/07 — đã thử phủ định kết luận này và THẤT BẠI, tức nó ĐÚNG:**
> `fnmatch.filter(files, "vln_n1/**/meta/info.json")` → **0 file**. Bảng quét ở dưới (122 file
> `meta/info.json`, toàn bộ dưới `vln_pe/`) là chính xác ở mức **danh sách file của repo**.
>
> Dataset card nói cả 3 subset đều LeRobot v2.1 — không mâu thuẫn, nếu file LeRobot của `vln_n1`
> nằm **bên trong `.tar.gz`** như `vln_ce`. ⬜ đang xác minh, xem 3.14.
>
> Bài học: card của dataset **không** thắng được kết quả quét file thật. Suýt đi tải nhầm vì tin card.

**Kết quả quét từ khoá trên toàn bộ 20 829 file:**

| Từ khoá | Số file | Nằm ở đâu |
|---|---|---|
| `meta/info.json` | 122 | **chỉ** `vln_pe/traj_data/r2r_aliengo/<scene>/meta/` |
| `episodes.jsonl` | 122 | **chỉ** `vln_pe/traj_data/...` |
| `tasks.jsonl` | 122 | **chỉ** `vln_pe/traj_data/...` |
| `.parquet` | 5 193 | **chỉ** `vln_pe/traj_data/<...>/data/chunk-000/episode_XXXXXX.parquet` |
| `traj_data` | 20 816 | `vln_ce/traj_data/r2r/<scene>.tar.gz` + `vln_pe/traj_data/...` |

**Hai kết luận:**

1. **`vln_ce/traj_data/` KHÔNG phải LeRobot** — nó là `.tar.gz` đóng gói theo scene
   (`17DRP5sb8fy.tar.gz`, ...). Muốn xem bên trong phải tải rồi giải nén, không soi được từ xa.
2. **Nguồn LeRobot thật duy nhất trong repo là `vln_pe/traj_data/`.** Cấu trúc chuẩn LeRobotDataset:
   ```
   vln_pe/traj_data/r2r_aliengo/<scene_id>/
       meta/info.json  meta/episodes.jsonl  meta/tasks.jsonl  [meta/episodes_stats.jsonl]
       data/chunk-000/episode_000000.parquet ...
       [videos/...]
   ```
   **Mỗi scene là một LeRobotDataset độc lập** (có `meta/` riêng) — 122 scene, ~5193 episode,
   trung bình ~42 episode/scene.

**Xung đột với kế hoạch:** Phụ lục A.4 ghi *"Tuần này chỉ cần `vln_ce` + split `r2r`, không đụng
`vln_pe`, `vln_n1`"*. Chỉ dẫn đó **đúng cho việc eval** (Người A + Ngày 4) nhưng **sai cho Ngày 5**
của Người B: mục tiêu Ngày 5 là chốt schema LeRobot với nhóm SIM, mà schema LeRobot **chỉ có ở
`vln_pe`**. Không có nguồn thay thế.

→ **Quyết định: vẫn lấy `vln_pe`, nhưng chỉ lấy `meta/` của MỘT scene** (vài trăm KB) + 1-2 file
`.parquet` mẫu. Đây là thứ thay thế cho `preview/` 59.5MB mà kế hoạch định dùng.

**Lưu ý về embodiment:** thư mục tên `r2r_aliengo` → dữ liệu render bằng robot **Aliengo** (chó 4 chân)
trong Isaac Sim, không phải camera của agent VLN-CE. Ảnh hưởng tới data contract: chiều cao camera,
hệ toạ độ, action space có thể khác thứ nhóm SIM sinh từ Habitat. **Phải nêu rõ điểm này khi chốt
schema Ngày 5**, đừng bê nguyên `info.json` sang mà không đối chiếu.

### Lệnh tải (rất nhẹ — thay thế cho `preview/`)

```python
SCENE = "17DRP5sb8fy"      # scene bat ky, day la scene dau tien
BASE  = f"vln_pe/traj_data/r2r_aliengo/{SCENE}"

# 1. Kiem tra pattern khop truoc (quy tac o 3.10)
import fnmatch
print(len(fnmatch.filter(files, f"{BASE}/meta/*")), "file meta")

# 2. Toan bo meta/ - vai tram KB, day la SCHEMA that
!hf download InternRobotics/InternData-N1 --repo-type dataset \
    --include "vln_pe/traj_data/r2r_aliengo/17DRP5sb8fy/meta/*" \
    --local-dir /kaggle/working/data

# 3. Mot episode parquet lam mau
!hf download InternRobotics/InternData-N1 --repo-type dataset \
    --include "vln_pe/traj_data/r2r_aliengo/17DRP5sb8fy/data/chunk-000/episode_000000.parquet" \
    --local-dir /kaggle/working/data
```

Đọc ra để lấy nguyên liệu cho `docs/data_contract.md`:
```python
import json, pandas as pd
info = json.load(open(f"/kaggle/working/data/{BASE}/meta/info.json"))
print(json.dumps(info, indent=2)[:3000])          # features, shape/dtype, fps, codebase_version
df = pd.read_parquet(f"/kaggle/working/data/{BASE}/data/chunk-000/episode_000000.parquet")
print(df.dtypes); print(df.head(3)); print("so frame:", len(df))
```

`info.json` chính là **hợp đồng dữ liệu** cần đưa nhóm SIM: `features` (shape/dtype của RGB & depth),
`fps`, tên các cột state/action. `tasks.jsonl` cho format instruction.

### Còn phải xác minh cho `vln_ce`

Kế hoạch A.4 bảo tải `vln_ce/raw_data/r2r/*` — **chưa xác nhận `raw_data/` có tồn tại không**
(đã sai một lần với `preview/`, đừng tin lần hai):
```python
print(len(fnmatch.filter(files, "vln_ce/raw_data/*")), "file trong vln_ce/raw_data")
print(sorted({f.split("/")[1] for f in files if f.startswith("vln_ce/")}))
```

---

## 3.12. ⚠️ `HF_HUB_ENABLE_HF_TRANSFER` đã bị deprecated — đang tải ở tốc độ mặc định

**Cảnh báo gặp phải:**
```
FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as
'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead.
```

Kế hoạch (A.1) hướng dẫn `export HF_HUB_ENABLE_HF_TRANSFER=1` — biến này **không còn tác dụng** ở
`huggingface_hub` bản mới. HF đã chuyển sang backend **Xet**. Tức là nếu không sửa, bạn sẽ kéo
16.79GB ở tốc độ thường thay vì tốc độ cao.

**Fix — set TRƯỚC khi import `huggingface_hub`:**
```python
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
```

Dấu hiệu Xet đang hoạt động: output có dòng `Reconstructing` / `Reconstruction complete` (đã thấy
trong log) — đó là cơ chế dedup theo chunk của Xet, không phải lỗi.

🔧 **PR candidate:** docs InternNav hướng dẫn biến môi trường đã deprecated.

---

## 3.13. 🔧 `meta/info.json` mô tả SAI dữ liệu của chính nó — `features` không khớp cột parquet

**Đã tải & đọc được** (21/07): `vln_pe/traj_data/r2r_aliengo/17DRP5sb8fy/meta/*` + `episode_000000.parquet`.
Chi tiết đầy đủ ở `docs/data_contract.md`. Tóm tắt phần là **lỗi**:

**Triệu chứng:** `info.json` khai báo `features` gồm `observation.camera_intrinsic` (3,3),
`observation.camera_extrinsic` (4,4), `action` float32 (4,4). Parquet thật **không có cột nào
trong ba cột đó**. Thay vào đó là `camera_position`/`camera_orientation`/`camera_yaw`,
`robot_position`/`robot_orientation`/`robot_yaw`, `progress`, `step`, và `observation.action`
kiểu **int64 rời rạc** (giá trị thấy được: 1, 3).

> ✅ **ĐÃ CHỨNG MINH DỨT ĐIỂM (21/07, sau khi đo `vln_n1`):** `info.json` của `vln_n1` khai đúng
> 3 feature `camera_intrinsic`/`camera_extrinsic`/`action` (4×4) và parquet của nó có **đúng 3 cột đó**.
> Tức file `info.json` của `vln_pe` là **bản copy từ `vln_n1` mà quên sửa `features`**. Không còn là
> suy luận từ code — có hai file thật để đối chiếu cạnh nhau. Xem `docs/data_contract.md` mục 2.

**NGUYÊN NHÂN GỐC (đã truy ra từ source code, không phải suy đoán):** `features` này là
**template của NavDP/`vln_n1` bị copy sang `vln_pe` mà không sửa**. `internnav/dataset/navdp_lerobot_dataset.py`
đọc đúng ba trường đó, đúng từng shape:

```python
camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
camera_trajectory = np.array([np.stack(frame) for frame in df['action']], ...).reshape(-1, 4, 4)
```

Còn action rời rạc trong parquet khớp với `internvla_n1_lerobot_dataset.py` / `vlln_lerobot_dataset.py`:
`self.idx2actions = {0: 'STOP', 1: "↑", 2: "←", 3: "→", 5: "↓"}` → giá trị `3, 3, 1` = phải, phải, tiến.
Và `cma_lerobot_dataset.py` đọc đúng `position`/`orientation`/`yaw`/`progress`/`step`.

→ **Ta KHÔNG tải nhầm data. Parquet đúng, metadata sai.** Ba dấu hiệu khác cùng chỉ vào "template
chưa điền": `robot_type: "unknown"`, `splits: {"train": "0:1"}`, `fps: 30`.

**Vì sao nghiêm trọng:** `info.json` chính là thứ ta định đưa nhóm SIM làm hợp đồng dữ liệu
(mục 3.11). Nếu chốt schema theo nó thì:
- Bên SIM sẽ sinh `action` dạng ma trận SE(3) 4×4 liên tục, trong khi dữ liệu thật là **action rời rạc**
  kiểu VLN-CE. Hai bên **không ghép được**, và lỗi chỉ lộ ra lúc train.
- Bên SIM sẽ chờ `camera_intrinsic` — không tồn tại trong subset này.

**Hai mâu thuẫn phụ, cùng loại:**
- `fps: 30` nhưng `timestamp` bước đều **0.166667 s → 6 Hz**. Ai tính thời gian bằng
  `frame_index / fps` sẽ ra **sai 5 lần**. → Contract phải ghi: dùng cột `timestamp`, đừng suy từ `fps`.
- `splits: {"train": "0:1"}` nhưng `total_episodes: 23` → trường `splits` vô dụng.

**Cách xử lý:** **không tin `features` trong `info.json`.** Lấy schema từ `df.dtypes` của parquet thật,
và ghi cả hai (khai báo vs thực tế) vào contract để nhóm SIM thấy chỗ lệch. Đã làm ở
`docs/data_contract.md` mục 3.

**Chưa xác minh:** mới đọc 1 scene / 1 episode / 3 frame. Phải đọc thêm parquet của scene khác để
biết đây là lỗi hệ thống hay riêng scene `17DRP5sb8fy`.

🔧 **PR candidate (mạnh nhất trong ngày):** `meta/info.json` của `InternData-N1/vln_pe` khai báo
`features` của **NavDP** thay vì của chính nó. Đây không phải lỗi docs mà là **lỗi metadata trong
chính dataset** — nặng hơn `preview/` (3.10) vì nó không báo lỗi gì cả, chỉ lặng lẽ dẫn người dùng
tới schema sai. Có bằng chứng đối chiếu từ source code của chính repo → PR rất dễ bảo vệ.

---

## 3.14. 🚨 Repo có ít nhất 3 schema LeRobot khác nhau — `vln_pe` KHÔNG phải schema của InternVLA-N1

Phát hiện phụ khi truy nguyên nhân 3.13, nhưng **ảnh hưởng tới Ngày 5 lớn hơn cả 3.13**.

Đọc `internnav/dataset/*_lerobot_dataset.py` trên GitHub:

| Loader | Cột chính | Action |
|---|---|---|
| `navdp_lerobot_dataset.py` | `camera_intrinsic` (3,3), `camera_extrinsic` (4,4) | 4×4 SE(3) liên tục |
| `cma_lerobot_dataset.py` | `position`, `orientation`, `yaw`, `progress`, `step` | int rời rạc |
| `internvla_n1_lerobot_dataset.py`, `vlln_lerobot_dataset.py` | **`pose.{setting}`**, **`goal.{setting}`**, `relative_goal_frame_id.{setting}` | int rời rạc |

`vln_pe/r2r_aliengo` mà ta tải khớp **hàng giữa** (CMA). Nhưng model đích của cả nhóm là
**InternVLA-N1** — loader của nó cần `pose.{setting}` + `goal.{setting}`, **không có** trong parquet này.

**Hệ quả:** đưa nhóm SIM schema của `vln_pe` → họ sinh data đúng theo nó →
`internvla_n1_lerobot_dataset.py` **vẫn không đọc được**. Lỗi này chỉ lộ ra ở W4 khi fine-tune,
tức là muộn nhất có thể.

### ✅ QUYẾT ĐỊNH (21/07): contract đi theo **`vln_n1/`**, không phải `vln_pe/`

Đã xác nhận model đích là **InternVLA-N1** → subset đích là **`vln_n1/`**.

| Subset | Vai trò từ giờ |
|---|---|
| **`vln_n1/`** | **Nguồn schema chính thức** cho data contract Ngày 5 — ⬜ chưa đo |
| `vln_pe/` | Chỉ còn 2 vai trò: (a) baseline CMA/RDP cho eval Ngày 4 (data **đọc vào**, không phải data SIM sinh ra); (b) bằng chứng cho PR fix docs (3.13) |
| `vln_ce/` | `.tar.gz`, không phải LeRobot — không liên quan contract |

### ⚠️ Trở ngại: `vln_n1/` KHÔNG có `meta/info.json` ở mức danh sách file

Chạy `fnmatch.filter(files, "vln_n1/**/meta/info.json")` → **0 file**. Khớp với bảng quét ở 3.11
(122 `meta/info.json`, toàn bộ dưới `vln_pe/`). Nên **không lấy schema `vln_n1` bằng cách tải
`meta/*` được** như dự định ban đầu.

Giả thuyết đang kiểm: file LeRobot của `vln_n1` nằm **trong `.tar.gz`** như `vln_ce`. Lệnh chẩn đoán:

```python
n1 = [f for f in files if f.startswith("vln_n1/")]
print(len(n1))
print(sorted({"/".join(f.split("/")[:3]) for f in n1})[:40])
from collections import Counter
print(Counter(f.rsplit(".", 1)[-1] for f in n1).most_common(10))
print("\n".join(n1[:20]))
```

→ Kết quả: **toàn `.tar.gz`** (3774 file), xem khối ✅ bên dưới.

🔧 **PR candidate:** dataset card mô tả cả 3 subset cùng một cây thư mục LeRobot
(`traj_data/<scene>/meta/`, `data/`, `videos/`), nhưng thực tế chỉ `vln_pe` có cấu trúc đó ở dạng
file rời. Người làm theo card sẽ tải trượt y như ta.

**Bài học ghi lại:** đã suýt sửa kết luận đúng của mục 3.11 chỉ vì dataset card nói khác.
**Card không thắng được kết quả quét file thật.**

### ✅ KẾT QUẢ (21/07): `vln_n1` = toàn `.tar.gz`, và là schema **System 1**, không phải System 2

3774 file `.tar.gz`, cây `vln_n1/traj_data/<sim>_<camera>/<uuid>.tar.gz`, 6 simulator × 2 camera
(`d435i`, `zed`). Mở `matterport3d_d435i/pLe4wQe7qrG.tar.gz` (248.7 MB) → bên trong **đúng là
LeRobot** (`meta/info.json`, `data/chunk-000/*.parquet`, `videos/*.mp4`).

**Nhưng schema là của NavDP** (`camera_intrinsic` 3×3, `camera_extrinsic` 4×4, `action` 4×4 SE(3)),
**không phải** `pose.{setting}`/`goal.{setting}` như suy đoán ban đầu từ loader. Hợp lý: System 1 của
InternVLA-N1 chính là diffusion policy kiểu NavDP → `vln_n1` là data của **System 1**.

**Intrinsic xác nhận camera thật:** `fx=355.81, fy=351.69, cx=240, cy=135` → ảnh **480×270**,
FOV **68°×42°**, khớp spec Intel RealSense **D435i (69°×42°)**. Hậu tố `_d435i`/`_zed` là model
camera thật → nhóm SIM phải mô phỏng đúng, không phải camera tuỳ ý. ⬜ chưa đo biến thể `_zed`.

→ **Không tồn tại "một schema của InternVLA-N1"** — dual-system thì mỗi system một loại data.
Chi tiết + bảng đối chiếu ở `docs/data_contract.md` mục 2 và 4.

**Vẫn còn thiếu:** subset nào chứa `pose.{setting}`/`goal.{setting}` cho
`internvla_n1_lerobot_dataset.py`. Ứng viên: `vln_ce/traj_data/*.tar.gz`. Xem `data_contract.md` 4.b.

### 3.14.b 🔧 PR candidate mới: `vln_n1` thiếu trường bắt buộc của LeRobot v2.1

Parquet của `vln_n1` chỉ có `index` + 3 cột observation/action. **Thiếu `timestamp`, `frame_index`,
`episode_index`, `task_index`** — đều là trường bắt buộc của LeRobotDataset v2.1, và `vln_pe` thì có đủ.

→ `vln_n1` khai `codebase_version: "v2.1"` nhưng **không đọc được bằng loader LeRobot gốc**, chỉ
loader riêng của InternNav đọc được. Người dùng tin vào `codebase_version` sẽ mất thời gian.

Cộng thêm: `fps: 30` và `splits: {"train": "0:1"}` xuất hiện **y hệt ở cả hai subset** → đây là giá
trị template chưa điền ở mọi nơi, không đáng tin ở bất kỳ subset nào.

**Kết luận cuối về 3.11 (đã giải quyết mâu thuẫn):** "LeRobot chỉ có trong `vln_pe`" **đúng ở mức
danh sách file** — cả 3774 file của `vln_n1` đều là `.tar.gz`, LeRobot nằm *bên trong* archive.
Dataset card mô tả cả 3 subset cùng một cây thư mục file rời → **card sai**, đó là PR candidate ở 3.14.

**Video khác nhau giữa hai subset:** `vln_pe` dùng `.npy`, `vln_n1` dùng `.mp4`. Cả hai parquet đều
**không chứa RGB/depth** — ảnh nằm ở `videos/`. ⬜ chưa mở file video nào để biết shape/dtype.

---

## 3.15. 🔀 (22/07) Có HAI biến thể dual-system, System 1 KHÁC NHAU: NavDP vs NextDiT-async

> 📄 **Bảng so sánh đầy đủ + weight keys + "tại sao 2 bản" + câu hỏi mở để pivot sau này:
> [`docs/checkpoint_variants.md`](docs/checkpoint_variants.md).** Mục 3.15 giữ lại như **bản ghi ngày
> phát hiện**; tài liệu kia là bản tra cứu sống để so sánh về sau.

**Bối cảnh:** demo `scripts/notebooks/inference_only_demo.ipynb` dùng
`internnav.agent.internvla_n1_agent_realworld.InternVLAN1AsyncAgent` và trỏ checkpoint
`InternVLA-N1-DualVLN`. Survey repo đó (HF tree API, 22/07) + đọc `config.json`:

| | `InternVLA-N1` (đã đóng gói `tieulam/internvla-n1-ckpt`) | `InternVLA-N1-DualVLN` |
|---|---|---|
| `model_type` | `qwen2_5_vl` (3.9) | **`internvla_n1`** |
| `architectures` | Qwen2.5-VL | **`InternVLAN1ForCausalLM`** |
| **System 1** | **NavDP** diffusion (`model.language_model.navdp.*`, xem 2.4) | **`system1: "nextdit_async"`** (NextDiT) — không có key `navdp`/`diffusion` |
| Tổng safetensors | 16.79 GB | **16.77 GB** |
| shard 1–3 | 4.966 / 4.991 / 4.933 | 4.968 / 4.991 / 4.933 (≈ trùng → chung S2 backbone) |
| shard 4 (phần S1) | 1.888 GB | 1.875 GB |
| `tokenizer.json` (fast) | ✅ có | ❌ **không** (chỉ `vocab.json`+`merges.txt`, giống `-System2` mục 3.6) |
| train artifacts (`trainer_state`…) | có | không |

**Ba kết luận:**

1. **"-DualVLN ~8GB" là SAI.** Con số đó đọc từ comment trong notebook. Thực tế **16.77GB, full 7B**
   (`hidden_size=3584`, `num_hidden_layers=28` — cùng lớp 7B, xem 3.9).
2. **S2 backbone giống nhau** (shard 1–3 gần trùng byte) → **Ngày 2 KHÔNG cần tải `-DualVLN`.** Dùng
   `InternVLA-N1` đã đóng gói; bản này còn **hơn** vì có sẵn `tokenizer.json` fast (tránh rủi ro
   slow-tokenizer với `transformers` 5.0.0, xem 3.6).
3. **System 1 của hai bản là hai kiến trúc khác nhau:** NavDP (bản mình) vs NextDiT-async (`-DualVLN`).
   Demo `InternVLAN1AsyncAgent` (chữ *Async*) khớp `-DualVLN`, **không** khớp bản NavDP.

**⚠️ Hệ quả cho Ngày 4:** muốn chạy `inference_only_demo.ipynb` / `InternVLAN1AsyncAgent` thì **phải tải
`InternVLA-N1-DualVLN`** (16.77GB riêng) — **không repoint được** sang checkpoint NavDP đã có, vì khác
`config` class (`internvla_n1` vs `qwen2_5_vl`) và khác weight keys (`nextdit` vs `navdp`). Phương án thay:
tìm agent class khớp NavDP thay vì async. Đây là quyết định Ngày 4, không chặn Ngày 2.

**Cập nhật hiểu biết ở 2.4:** kết luận "dual-system phải dùng model class của repo" vẫn đúng, nhưng giờ
chính xác hơn — repo có **≥2** đường dual-system (NavDP diffusion và NextDiT-async), và demo real-world
đi theo đường NextDiT-async.

---

## 3.16. (22/07) Truy xong vị trí code load checkpoint S1+S2 — lộ thêm 2 bẫy

Đọc source `internvla_n1_policy.py` / `internvla_n1_arch.py` / `navdp.py`. Bảng đầy đủ:
`docs/checkpoint_variants.md` **5.b**; áp vào kế hoạch Kaggle: `docs/eval_plan_kaggle_s2.md` **3.1.b**.

- **Điểm load duy nhất cho CẢ S1+S2:** `InternVLAN1Net.__init__` (policy, ~dòng 33–40) gọi
  `InternVLAN1ForCausalLM.from_pretrained(model_config.model_path, ...)`. **S1 không có file weights
  riêng** — nằm trong 4 shard chung; weights `navdp.*` chỉ vào đúng chỗ khi `InternVLAN1MetaModel.__init__`
  đã dựng `self.navdp` (qua `build_navdp` → `NavDP_Policy_DPT_CriticSum_DAT` + `load_model()`).
  Khớp và giải thích trọn phát hiện 2.4 (class thuần không có chỗ chứa → `UNEXPECTED`).
- 🚨 **Bẫy 1:** lời gọi gốc **`device_map={"": device}`** — pin toàn bộ 16.79GB vào MỘT GPU. Trên Kaggle
  T4×2 phải override `device_map="auto"`, nếu không OOM chắc chắn trên 1 T4 (khớp tính toán 3.9).
- 🚨 **Bẫy 2:** nhánh dựng S1 rẽ theo **`config.system1`** (`'navdp' in ...`, lồng `'async' in ...`) —
  mà config bundle NavDP là `model_type: qwen2_5_vl`, ⬜ **chưa thấy field `system1`** (3.15). Thiếu field
  → S1 bị vứt im lặng **dù đã dùng đúng class repo**. Verify sau load: đếm
  `sum(p.numel() for n,p in model.named_parameters() if 'navdp' in n) > 0` + soi log `MISSING`/`UNEXPECTED`.
- `navdp.py::load_model()` còn có đường load **standalone**: `torch.load(self.navdp_pretrained)` khi path
  được set (None → random init, rồi bị `from_pretrained` ghi đè) — tức NavDP có thể chạy tách rời với
  `.ckpt` riêng, liên quan luồng "local" của Người A.

---

## 4. Trạng thái cuối Ngày 1 (21/07/2026)

### ✅ Đã xong

- [x] `HF_HOME` → `/kaggle/temp/hf_cache` (phải `makedirs`), xác nhận không nhân đôi (2.1)
- [x] `HF_XET_HIGH_PERFORMANCE` thay `HF_HUB_ENABLE_HF_TRANSFER` (3.12)
- [x] Survey A.2 + `min_gb=0` → dung lượng thật, phát hiện `-System2` thiếu `tokenizer.json` (3.1, 3.6)
- [x] `config.json` → **base model Qwen2.5-VL-7B** (3584/28) (3.9)
- [x] **Tải `InternVLA-N1` 16.79GB** — 1m51s, đối chiếu 18/18 file với metadata HF
- [x] **Đóng gói + mount lại được** → Kaggle Dataset `tieulam/internvla-n1-ckpt` ← *tiêu chí "xong" của Ngày 1*
- [x] **Smoke test T4 x2 ĐẠT** — 16.58GB, shard đúng 2 GPU (2.4)
- [x] Schema `vln_pe` (file rời) — phát hiện `info.json` mô tả sai parquet (3.13)
- [x] Schema `vln_n1` (archive) — schema **System 1**, `info.json` khớp parquet (3.14)
- [x] Xác minh `scripts/eval/` tồn tại, đúng đường dẫn tài liệu (Ngày 1 mục 2.3)

### ⬜ Còn lại — ưu tiên cao

- [ ] 🚨 **Mail bổ sung anh Huy: VRAM ≥16GB → ≥24GB.** Giờ có số **đo được** (16.58GB thật), không
      còn là ước lượng — xem 2.4. Mail đầu đang đi với thông số sai.
- [ ] Sync Người A: (a) 7B không phải 3B; (b) đường dẫn dataset; (c) `latent_queries` + `navdp.*`
      trong checkpoint khớp phần `internvla_n1_agent.py` mà Người A đang đọc (2.4)
- [ ] Tìm data có `pose.{setting}`/`goal.{setting}` → mở 1 tar.gz của `vln_ce/traj_data/`
      (`docs/data_contract.md` 4.b) — mảnh cuối của data contract

### ⬜ Còn lại — làm được thì tốt

- [ ] Tải `VLN-PE` kèm `--exclude` bỏ rác train, tiết kiệm 0.44GB (3.8)
- [ ] Đọc `VLN-PE/meta.yaml` → map checkpoint ↔ config trước Ngày 4 (3.8)
- [ ] Đo intrinsic biến thể **`_zed`** → đối chiếu `_d435i` (480×270, FOV 68°×42°)
- [ ] Đọc `meta/tasks.jsonl` + `episodes.jsonl` của `vln_n1` → format instruction
- [ ] Mở 1 file video (`vln_n1`: `.mp4`, `vln_pe`: `.npy`) → shape/dtype ảnh
- [ ] Ghi chú đọc `eval.py` / `start_server.py` / `habitat_s2_cfg.py`

### ⬜ Chuyển sang Ngày 3–4

- [ ] Dựng env `internnav-habitat` (`.[habitat]`) trên cloud/server — **việc của Người B**, Ngày 3.
      Nếu `pip install -e ".[habitat]"` gãy → dùng `conda install habitat-sim -c aihabitat`
- [ ] **Ngày 4: dual-system phải dùng model class của repo InternNav**, không dùng `transformers`
      thuần — nếu không System 1 bị bỏ qua im lặng (2.4)
- [ ] Cảnh giác `pip install -e .` hạ cấp `transformers` 5.0.0 → xung đột `torch` (2.2)

---

## 6. Tổng kết Ngày 1 — 8 chỗ tài liệu sai, 5 PR candidate

| # | Tài liệu ghi | Thực tế đo được | Mục |
|---|---|---|---|
| 1 | S2 là Qwen2.5-VL **3B**, ~6-7GB | **7B**, 16.58GB đo thật | 3.1, 3.9, 2.4 |
| 2 | Tải `-System2` cho nhẹ | Chênh **0.2GB**, lại thiếu `tokenizer.json` | 3.2, 3.6 |
| 3 | `HF_HOME=/kaggle/temp/...` | `/kaggle/temp` **không tồn tại sẵn** | 2.1 |
| 4 | `HF_HUB_ENABLE_HF_TRANSFER=1` | **Deprecated**, vô tác dụng | 3.12 |
| 5 | Lấy schema từ `preview/` | **Không tồn tại**; gốc chỉ có `vln_ce/` `vln_n1/` `vln_pe/` | 3.10 |
| 6 | Dataset card: 3 subset cùng cấu trúc | Chỉ `vln_pe` là file rời; `vln_n1` (3774) & `vln_ce` là **`.tar.gz`** | 3.14 |
| 7 | `info.json` là hợp đồng dữ liệu | `vln_pe/info.json` là **template copy từ `vln_n1`**, mô tả sai chính nó | 3.13 |
| 8 | *(giả định)* một schema cho InternVLA-N1 | **Dual-system → nhiều schema.** `vln_n1` = System 1 | 3.14 |

**PR candidate, xếp theo độ mạnh:**

| PR | Nội dung | Bằng chứng |
|---|---|---|
| **A** | `vln_pe/meta/info.json` khai `features` của `vln_n1` → mô tả sai chính nó | ⭐⭐⭐ hai file thật đối chiếu cạnh nhau (3.13) |
| **B** | `vln_n1` khai `codebase_version: v2.1` nhưng thiếu `timestamp`/`frame_index`/`episode_index`/`task_index` | ⭐⭐⭐ loader LeRobot gốc không đọc được (3.14.b) |
| **C** | Docs hướng dẫn `--include "preview/*"` cho đường dẫn không tồn tại; `hf download` in `✓ Downloaded` khi khớp 0 file | ⭐⭐ tái hiện 2 lần (3.10) |
| **D** | Dataset card mô tả sai cấu trúc thư mục của `vln_n1`/`vln_ce` | ⭐⭐ (3.14) |
| **E** | `VLN-PE`: 4/7 thư mục checkpoint thiếu `config.json`, README không nhắc | ⭐ (3.8) |

**Nguyên tắc rút ra cho cả tuần:** *đo trước, tải sau* — và khi **khai báo mâu thuẫn với dữ liệu,
dữ liệu thắng**. Trong ngày hôm nay, `info.json`, dataset card và README đều đã sai ít nhất một lần;
`df.columns` và `fnmatch.filter` thì chưa sai lần nào.

---

## 5. Lỗi HuggingFace — bảng tra nhanh

Ba mã lỗi này trông giống nhau nhưng nguyên nhân hoàn toàn khác:

| Mã | Nguyên nhân thật | Xử lý |
|---|---|---|
| **401** Unauthorized | Vấn đề *token*: chưa set / sai / đã thu hồi | Kiểm tra `HF_TOKEN`, `.strip()` bỏ whitespace thừa |
| **403** GatedRepoError | Token đúng, nhưng *tài khoản* chưa được duyệt | Bấm agree ở trang repo; xem settings/gated-repos |
| **404** Not Found | Thường **không** phải lỗi quyền mà là **sai `--repo-type`** | Dataset phải có `--repo-type dataset` |
