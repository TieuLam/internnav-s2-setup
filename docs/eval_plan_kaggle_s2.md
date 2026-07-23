# Kế hoạch: Full-inference **dual-system (S1 NavDP + S2)** — InternVLA-N1 trên Kaggle

> **Loại tài liệu:** kế hoạch thực hiện (runbook) — viết cho người **ít kinh nghiệm HF / Kaggle /
> InternNav**, nên có [Từ điển thuật ngữ](#0-từ-điển-thuật-ngữ-đọc-mục-này-trước) ở đầu.
> **Ngày viết:** 22/07/2026. **Cập nhật:** đổi phạm vi từ "chỉ S2" → **đầy đủ S1+S2 open-loop** (theo yêu cầu).
> **Nguồn sự thật:** `SETUP_NOTES.md` + `docs/{io_system2, data_contract, vln_subsets_architecture,
> checkpoint_variants}.md` + **đọc code repo** (`internvla_n1_policy.py`, `internvla_n1_agent.py`, 22/07).
> **Nguyên tắc:** *đo trước, tải sau; khi khai báo mâu thuẫn dữ liệu, dữ liệu thắng.*

---

## 0. Từ điển thuật ngữ (đọc mục này trước)

| Thuật ngữ | Giải thích ngắn gọn cho bối cảnh này |
|---|---|
| **HF (HuggingFace)** | Kho model + dataset, "GitHub cho model". Tải bằng thư viện `huggingface_hub`. |
| **HF repo — 2 loại** | `model` repo (checkpoint) và `dataset` repo (dữ liệu). Tải dataset **bắt buộc** `repo_type="dataset"`, quên là lỗi **404**. |
| **Checkpoint / `.safetensors` / shard** | File trọng số đã học. Model lớn chia nhỏ (shard) thành nhiều file, cần `model.safetensors.index.json` ghép lại. Của ta **16.79 GB**. |
| **VLM (Vision-Language Model)** | Model nhận **cả ảnh lẫn chữ**. System 2 là VLM **Qwen2.5-VL-7B**. |
| **InternVLA-N1 = dual-system** | **System 2** (VLM, "suy nghĩ chậm": ảnh+lệnh → điểm đích/hướng) + **System 1** (NavDP, "phản xạ nhanh": sinh **quỹ đạo lái**). Kế hoạch này chạy **cả hai**. |
| **NavDP** | Biến thể System 1 của bản `InternVLA-N1` đã đóng gói. Là **diffusion policy**, dùng **RGB-D** (cần depth). Khác với biến thể `NextDiT-async` của bản `DualVLN`. |
| **`latent` (cầu nối S2→S1)** | Vector "kế hoạch ẩn" mà S2 sinh ra (token đặc biệt `151667` / `latent_queries`), đưa cho S1 để dựng quỹ đạo. |
| **`generate_traj` / `dp_actions`** | Hàm của S1 sinh ra **quỹ đạo** (chuỗi waypoint/pose) từ `latent` + RGB + depth. `dp` = diffusion policy. |
| **Kaggle Notebook / Dataset / Accelerator** | Notebook = môi trường code có GPU. Dataset = dữ liệu mount vào `/kaggle/input/<tên>`. Accelerator = phần cứng (ta cần **T4×2**). |
| **`device_map="auto"`** | Tự chia model ra 2 GPU cho vừa bộ nhớ. Thiếu nó → tràn VRAM (OOM). |
| **VRAM / OOM** | Bộ nhớ GPU / hết bộ nhớ → process **chết ngay** (Kaggle không có swap). |
| **bf16** | Số 16-bit, 2 byte/tham số → 8.3 tỷ tham số ≈ 16.6 GB. |
| **flash-attention** | Kỹ thuật tăng tốc attention. Code repo **hardcode `flash_attention_2`**, nhưng nó **khó chạy trên GPU T4** (kiến trúc sm_75 cũ) → phải ép sang `sdpa`. |
| **`pip install -e .`** | Cài repo InternNav ở chế độ "editable" để dùng **model class riêng của repo** (`InternVLAN1ForCausalLM`). Bắt buộc để nạp được S1. |
| **DepthAnything V2** | Model ước lượng depth từ ảnh RGB — **chỉ cần khi input không có depth thật** (vd camera thật). `vln_ce` **đã có depth sẵn** → eval trên dataset **không cần** nó. |
| **`vln_ce`** | Subset dữ liệu đã render sẵn (RGB **và depth** + câu lệnh + đáp án). Dữ liệu inference/eval của kế hoạch này. |
| **`{setting}`** | Cấu hình camera `f'{height}cm_{pitch}deg'` (vd `60cm_30deg`). `vln_ce` render 5 setting; **tránh `0deg`**. |
| **Open-loop vs closed-loop** | **Open-loop** = cho model xem ảnh render sẵn, KHÔNG có sim phản hồi (chạy trên Kaggle). **Closed-loop** = model lái robot trong Habitat/Isaac, mỗi bước render ảnh mới (cần sim + server, KHÔNG chạy Kaggle). |
| **Ground-truth (GT)** | "Đáp án đúng" trong dữ liệu để chấm điểm: cột `action`, `goal.{setting}` (của S2). Lưu ý `vln_ce` **không có GT quỹ đạo liên tục** cho S1 (đó là ở `vln_n1`). |

---

## 1. Mục tiêu & phạm vi

**Mục tiêu:** notebook Kaggle nạp **đầy đủ dual-system (S1 NavDP + S2)** từ checkpoint đã đóng gói, chạy
**full-inference open-loop** trên dữ liệu **`vln_ce` (R2R/RxR)**: mỗi observation → S2 sinh pixel-goal +
latent → S1 sinh **quỹ đạo** → chấm/visualize.

**Trong phạm vi (IN):**
- ✅ Nạp **cả S1 + S2** bằng **model class của repo** (`InternVLAN1ForCausalLM` qua `InternVLAN1Net`).
- ✅ **Checkpoint NavDP** đã đóng gói: Kaggle Dataset `tieulam/internvla-n1-ckpt` (không tải lại 16.77GB).
- ✅ Dữ liệu `vln_ce` (RGB **+ depth có sẵn**) → **không cần DepthAnything**.
- ✅ Chạy pipeline `s2_step → s1_step_latent → generate_traj` trên **observation tĩnh** (agent `InternVLAN1Agent`
  hỗ trợ, **không cần simulator**).
- ✅ Đầu ra: quỹ đạo S1 (visualize) + chỉ số S2 nơi có GT (pixel-goal L2 / action accuracy).

**Ngoài phạm vi (OUT):**
- ❌ **Closed-loop benchmark SR/SPL** (S1+S2 lái agent trong Habitat/Isaac): cần simulator + server ≥24GB →
  **không chạy Kaggle**. Để dành giai đoạn server.
- ❌ Biến thể **DualVLN / NextDiT-async** (`inference_only_demo.ipynb` gốc dùng bản này) — ta đi **NavDP**.
- ❌ Fine-tune / train.

> ⚠️ **Giới hạn metric phải ghi rõ trong notebook:** `vln_ce` có GT cho **System 2** (action, pixel-goal),
> **không** có GT quỹ đạo liên tục cho **System 1** (thứ đó ở `vln_n1`). Nên với S1, "eval" open-loop ở đây
> **chủ yếu là định tính** (quỹ đạo có hợp lý không) + kiểm "pipeline chạy end-to-end", KHÔNG phải điểm số
> benchmark. Điểm số thật (SR/SPL) chỉ ra được ở closed-loop trên server.

---

## 2. Vì sao lượt trước kế hoạch chỉ có S2 (và nay sửa)

- Lượt trước tôi **khoanh phạm vi hẹp** ở S2 vì S2 nạp được bằng `transformers` thuần (đường dễ nhất trên
  Kaggle), và docs nhóm cố ý hoãn dual-system sang "Ngày 4" do còn blocker.
- **Đọc code đã gỡ nghi ngờ lớn nhất:** agent `InternVLAN1Agent` chạy trên **observation tĩnh, không cần
  sim** → full dual-system **hoàn toàn làm open-loop trên Kaggle được**. Vì vậy kế hoạch mở rộng sang cả S1.
- Đổi lại, có **4 blocker mới** so với đường S2-thuần (xem [§3.3](#33-4-blocker-mới-khi-chạy-đủ-s1s2) và [§7](#7-rủi-ro--cách-xử-lý)).

---

## 3. Những gì ĐÃ kiểm chứng (nền tảng của kế hoạch)

### 3.1. Cơ chế chạy đủ dual-system (đọc code repo 22/07)

Chuỗi gọi để full-inference **một** observation (không cần sim):

```
s2_step(rgb, depth, pose, instruction, camera_intrinsic, look_down)   # System 2
   └─ nếu text có chữ số → pixel-goal:
        traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
        → S2Output.output_latent
s1_step_latent(rgbs, depths, output_latent)                           # System 1 (NavDP)
   └─ dp_actions = self.model.generate_traj(traj_latents=latent, images_dp=rgb, depths_dp=depth)
        → quỹ đạo (chuỗi waypoint/pose)
```

- **Model class:** `InternVLAN1ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
  attn_implementation="flash_attention_2")` — **class của repo**, không phải class HF thuần.
- **Policy class:** `InternVLAN1Net` (chứa `s2_step`, `s1_step_latent`, `parse_actions`, `reset`, ...).
- **Agent:** `InternVLAN1Agent` (`internnav/agent/internvla_n1_agent.py`) — **chạy trên observation tĩnh,
  pose = ma trận đơn vị, KHÔNG cần simulator**. ✅ Đây là chốt chặn cho "open-loop trên Kaggle làm được".
- **Depth là bắt buộc** cho S1 (`generate_traj(..., depths_dp=depth)`) → nhưng `vln_ce` **có sẵn depth**,
  nên **không cần DepthAnything** khi eval trên dataset. (DepthAnything chỉ cần cho input RGB-only, vd camera thật.)

### 3.1.b. Bản đồ vị trí code load checkpoint S1+S2 (đọc source 22/07)

**Kết luận:** S1 và S2 **không load riêng** — cả hai nằm trong **cùng** checkpoint 16.79GB và được nạp bởi
**cùng một lệnh `from_pretrained`**; nhưng module S1 phải được **dựng sẵn trong kiến trúc** thì weights
mới "rơi vào đúng chỗ".

| # | File | Class / method | Vai trò |
|---|---|---|---|
| 1 | `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` (~dòng 33–40) | `InternVLAN1Net.__init__` | **Điểm load duy nhất (cả S1+S2):** `InternVLAN1ForCausalLM.from_pretrained(self.model_config.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": self.model_config.device})` + `AutoTokenizer(..., use_fast=True)` + `AutoProcessor` |
| 2 | `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` | `InternVLAN1MetaModel.__init__` | **Dựng kiến trúc S1**: `elif 'navdp' in config.system1: if 'async' in config.system1: self.navdp = build_navdp(config, memory_size=2)`; tạo `self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))` |
| 3 | `internnav/model/basemodel/internvla_n1/navdp.py` | `NavDP_Policy_DPT_CriticSum_DAT.load_model()` | Loader NavDP **standalone** (tuỳ chọn): `torch.load(self.navdp_pretrained)` nếu path được set; `None` → random init (sau đó bị bước 1 ghi đè bằng `navdp.*` trong checkpoint lớn) |

Trình tự thực tế:

```
InternVLAN1Net.__init__                        (policy — file #1)
  └─ InternVLAN1ForCausalLM.from_pretrained(model_path)
       ├─ dựng kiến trúc: InternVLAN1MetaModel.__init__   (file #2)
       │    ├─ self.navdp = build_navdp(...)   ← rẽ nhánh theo config.system1
       │    └─ self.latent_queries = nn.Parameter(...)
       └─ đổ state_dict từ 4 shard safetensors:
            ├─ S2: backbone Qwen2.5-VL (model.layers.0..27, visual, ...)
            └─ S1: model.language_model.navdp.*  +  latent_queries
```

**Hai hệ quả trực tiếp cho blocker §3.3:**
1. 🚨 **Lời gọi gốc pin MỘT device** (`device_map={"": device}`) → dồn toàn bộ 16.79GB vào 1 GPU →
   OOM chắc chắn trên 1 T4. Đây là bằng chứng cụ thể cho blocker #3: **phải override `device_map="auto"`**.
2. 🚨 **Nhánh dựng S1 rẽ theo `config.system1`** — mà `config.json` bản NavDP đã đóng gói là
   `model_type: qwen2_5_vl` và **chưa thấy field `system1`** trong survey (`checkpoint_variants.md` §3).
   Nếu thiếu field → `self.navdp` **không được dựng** → `navdp.*` bị vứt im lặng **y như** load bằng class
   HF thuần. Đây là cơ chế cụ thể của blocker #4. ⬜ *Chưa xác minh nhánh non-async của `build_navdp`
   (quote chỉ thấy nhánh `'async' in config.system1`).*

### 3.2. Checkpoint & phần cứng (đã smoke-test phần S2)

- **Checkpoint:** `tieulam/internvla-n1-ckpt` (bản `InternVLA-N1` NavDP, 16.79GB), mount tại
  `/kaggle/input/internvla-n1-ckpt/...`. Có sẵn fast `tokenizer.json`.
- **Base:** Qwen2.5-VL-**7B** (`hidden_size=3584`, `num_hidden_layers=28`). Weights S2 đo thật **16.58GB**.
- **Phần cứng Kaggle (ràng buộc cứng):**

| Hạng mục | Giá trị | Hệ quả |
|---|---|---|
| Accelerator | **T4×2 bắt buộc** | 1×T4/P100 không đủ. Cần shard 2 GPU. **S1 + KV-cache cộng thêm** → xem [blocker bộ nhớ §3.3](#33-4-blocker-mới-khi-chạy-đủ-s1s2). |
| `/kaggle/working` | 20GB trần | Checkpoint đã ăn ~16.8GB → chỗ trống ~3.2GB. Dữ liệu + output phải gọn. |
| `/kaggle/temp` | **không tồn tại sẵn** | `os.makedirs(...)` trước khi dùng; để file giải nén tạm ở đây. |
| Swap | **0** | Chạm trần RAM → kernel chết ngay, không cảnh báo. |
| `transformers` | 5.0.0 preinstall | ⚠️ `pip install -e .` repo **có thể hạ cấp** xuống 4.x → xung đột (xem §3.3). |
| Internet | **ON** | Để tải `vln_ce` + cài repo. |

### 3.3. 4 blocker mới khi chạy đủ S1+S2

Đây là phần khó hơn hẳn đường S2-thuần. Phải xử lý theo thứ tự này:

1. **Cài repo InternNav** (`pip install -e .`) để có `InternVLAN1ForCausalLM`/`InternVLAN1Net`.
   - Rủi ro: repo có thể pin `transformers<5` → **hạ cấp** bản 5.0.0 preinstall, kéo xung đột `torch`.
   - Xử lý: đọc `pyproject.toml`/`requirements/*.txt` lấy con số thật; cài xong **restart kernel**.
2. **flash-attention trên T4 (sm_75):** code hardcode `attn_implementation="flash_attention_2"`, mà
   flash-attn 2 **khó/không chạy trên T4**. → **Phải ép `attn_implementation="sdpa"`** (hoặc `"eager"`)
   khi nạp model — nhiều khả năng phải sửa 1 dòng trong code repo hoặc truyền override.
3. **Bộ nhớ:** lời gọi gốc trong repo dùng **`device_map={"": device}`** — pin toàn bộ vào **MỘT GPU**
   (đọc code, [§3.1.b](#31b-bản-đồ-vị-trí-code-load-checkpoint-s1s2-đọc-source-2207)) → 16.79GB **OOM
   trên 1 T4**. Cộng thêm S1 (`generate_traj`) + depth encoder. → **Phải ép `device_map="auto"`**;
   theo dõi VRAM. Nếu vẫn tràn → phương án cuối **4-bit** hoặc **đẩy sang server ≥24GB**.
4. **Class-vs-config mismatch:** checkpoint NavDP có `config.json` `model_type: qwen2_5_vl` (không phải
   `internvla_n1`), nhưng ta nạp bằng `InternVLAN1ForCausalLM`. **Cơ chế cụ thể đã truy ra** ([§3.1.b](#31b-bản-đồ-vị-trí-code-load-checkpoint-s1s2-đọc-source-2207)):
   `InternVLAN1MetaModel.__init__` chỉ dựng `self.navdp` khi **`config.system1`** khớp — config bundle
   NavDP ⬜ chưa thấy field này → nguy cơ S1 bị vứt im lặng dù dùng đúng class repo. → **Verify bắt buộc**
   (Phase B4): đọc `config.json` trước khi load; sau khi load đếm
   `sum(p.numel() for n,p in model.named_parameters() if 'navdp' in n) > 0` + soi log `MISSING`/`UNEXPECTED`.
   Nếu thiếu → patch `config.json` / truyền override rồi load lại.

### 3.4. Dữ liệu `vln_ce` (đã đo — `data_contract.md` 4.b)

- Nguồn: HF dataset `InternRobotics/InternData-N1`, path `vln_ce/traj_data/{r2r,rxr}/<scene>.tar.gz`.
  Nhỏ nhất: `vln_ce/traj_data/rxr/YmJkqBEsHnH.tar.gz` (16.16 MB).
- Bên trong (LeRobot): `data/chunk-000/*.parquet`, `videos/.../*.png`, `meta/*.jsonl`.
- **Có sẵn cả RGB (640×480 uint8) VÀ depth (640×480 uint16, mm)** → đủ đầu vào cho **cả S2 lẫn S1**.
- Cột GT: `action` (int `{1↑,2←,3→,5↓}`, `-1`=start), `goal.{setting}` (pixel `[u,v]`, `(-1,-1)`=không),
  `pose.{setting}` (4×4). Instruction ở `meta/tasks.jsonl`.
- **Chọn setting có góc cúi** (`60cm_30deg`, `125cm_30deg`, ...). **TRÁNH `125cm_0deg`** (goal toàn `-1`).

### 3.5. Hợp đồng I/O System 2 & vấn đề pixel-goal (`io_system2.md`)

- Prompt template, history (`np.linspace(0, i-1, num_history=8)` + frame hiện tại), resize **384×384**,
  generate `max_new_tokens=128, do_sample=False`. Parser: có số → pixel-goal `[row,col]`; không số → action.
- 🚨 **Vấn đề mở:** lần chạy S2 trước ra `←←←←` (nhánh action), **chưa bật pixel-goal**. Nghi phạm: chưa
  resize 384 / thiếu `"you can see <image>."` / cần full policy. **Với dual-system điều này đặc biệt quan
  trọng:** S1 nhận `output_latent` — mà latent chỉ sinh ở **nhánh pixel-goal**. Nếu S2 rơi nhánh action thì
  **không có latent để feed S1**. → [Phase D](#phase-d--gỡ-blocker-pixel-goal--điều-kiện-cần-để-s1-chạy) là **điều
  kiện cần** để dual-system chạy, không chỉ để có metric.

---

## 4. Kiến trúc giải pháp

```
   Kaggle Dataset                         Kaggle Dataset (mới)
   tieulam/internvla-n1-ckpt (NavDP)      <bạn>/vln-ce-r2r-rxr  (RGB + depth + GT)
        │                                        │
        └──────────────────┬──────────────────────┘  mount
                           ▼
        ┌───────────────────────────────────────────────┐
        │   KAGGLE NOTEBOOK (T4×2, Internet ON)          │
        │  0. pip install -e . repo InternNav            │
        │  1. load InternVLAN1ForCausalLM (bf16,         │
        │     device_map=auto, attn=sdpa)  ← S1+S2       │
        │  2. verify navdp.* nạp đủ (không MISSING)      │
        │  3. mỗi frame: rgb+depth+instruction           │
        │       s2_step → output_latent (nhánh pixel)    │
        │       s1_step_latent(rgb,depth,latent) → traj  │
        │  4. chấm S2 (pixel-goal/action) + visualize traj│
        └───────────────────────────────────────────────┘
                           ▼
         /kaggle/working/{results.json, traj_vis/*.png}
```

---

## 5. Kế hoạch từng bước

> ⬜ chưa làm · 🔁 tái sử dụng việc đã có · 🚨 điểm dễ sai.

### Phase A — Chuẩn bị dữ liệu `vln_ce` (notebook CPU riêng, rẻ)

- **A1.** ⬜ Bật Internet. Set cache **trước khi tải**: `HF_HOME=/kaggle/temp/hf_cache` (phải `makedirs`),
  `HF_XET_HIGH_PERFORMANCE=1`.
- **A2.** 🚨 **Đếm file khớp trước khi tải** (`fnmatch.filter`) — tránh bẫy "tải 0 file mà báo thành công".
- **A3.** ⬜ Tải ~5 scene `r2r`+`rxr` bằng `hf_hub_download(..., repo_type="dataset")`, giải nén vào
  `/kaggle/working/vln_ce`.
- **A4.** ⬜ Chốt `SETTING` (vd `60cm_30deg`), dùng thống nhất mọi bước sau.
- **A5.** ⬜ Xây **index từng frame**: `(scene, ep, frame, rgb_path, depth_path, action_GT, goal_GT, pose)`
  từ parquet + `meta/*.jsonl`. Đảm bảo lấy **cả depth path** (S1 cần).
- **A6.** ⬜ Save Version → tạo Kaggle Dataset `vln-ce-r2r-rxr`. Dọn `/kaggle/temp` trước khi save.

### Phase B — Cài repo & nạp đủ dual-system (blocker nặng nhất)

- **B1.** ⬜ Notebook mới: **T4×2**, Internet ON, mount 2 dataset (ckpt + vln_ce).
- **B2.** ⬜ Clone + cài repo, chỉ phần cần cho inference:
  ```bash
  !git clone https://github.com/InternRobotics/InternNav.git --recursive
  # đọc pyproject.toml / requirements TRƯỚC, xem có pin transformers<5 không
  !pip install -e ./InternNav --no-build-isolation   # cân nhắc cài tối thiểu, tránh [habitat]/[isaac]
  ```
  🚨 Nếu bước này **hạ cấp `transformers`** → ghi lại version, **restart kernel**, chạy lại cell baseline.
- **B3.** ⬜ **Nạp model, ép 2 override quan trọng** (khác code gốc):
  ```python
  model = InternVLAN1ForCausalLM.from_pretrained(
      MODEL, torch_dtype=torch.bfloat16,
      attn_implementation="sdpa",       # KHÔNG flash_attention_2 (T4 sm_75)
      device_map="auto")                # ép shard 2 GPU
  print(model.hf_device_map)
  ```
- **B4.** 🚨 **Verify S1 nạp đủ** (blocker #4, cơ chế ở [§3.1.b](#31b-bản-đồ-vị-trí-code-load-checkpoint-s1s2-đọc-source-2207)):
  1. **Trước khi load:** đọc `config.json` trong checkpoint mount — có field `system1` không? (nhánh dựng
     `self.navdp` phụ thuộc nó). Không có → chuẩn bị patch/override.
  2. **Sau khi load:** `sum(p.numel() for n,p in model.named_parameters() if 'navdp' in n)` phải **> 0**,
     và log load **không** có `navdp.*` báo `MISSING`/`UNEXPECTED`.
  Nếu fail → S1 chưa nạp, chỉnh `config.json`/dựng model theo cách repo dùng rồi load lại.
- **B5.** ⬜ Đo VRAM (`torch.cuda.max_memory_allocated`) sau khi nạp. Nếu sát trần → chuẩn bị phương án
  4-bit / server.

### Phase C — Chạy 1 frame qua S2 (smoke test)

- **C1.** ⬜ Đọc 1 frame: RGB → resize 384×384; depth giữ nguyên (đơn vị mm). Lấy `camera_intrinsic`, pose.
- **C2.** ⬜ Gọi `policy.s2_step(rgb, depth, pose, instruction, intrinsic, look_down=False)`; in `output`
  (kỳ vọng có `output_pixel` + `output_latent`).

### Phase D — Gỡ blocker pixel-goal (**điều kiện cần** để S1 chạy)

- **D1.** ⬜ Nếu C2 ra **action, không có latent**: áp nghi phạm `io_system2.md` 4.a — đảm bảo resize 384 +
  câu `"you can see <image>."`; thử nhiều ảnh setting có góc cúi.
- **D2.** ⬜ Nếu **ra pixel-goal + latent** → sang Phase E chạy S1. Kiểm quy ước toạ độ `[row,col]` vs `[u,v]`.
- **D3.** ⬜ Nếu **vẫn action**: dual-system open-loop **bị chặn** (không có latent cho S1). Việc cần làm:
  đọc kỹ `s2_step` xem có tham số bật chế độ pixel / có cần `look_down` / có nhánh riêng khi `mode=dual_system`.
  Đây là điểm điều tra ưu tiên #1 nếu gặp.

### Phase E — Chạy S1 sinh quỹ đạo + eval loop

- **E1.** ⬜ Với mỗi frame (dựng history đúng `np.linspace(0,i-1,8)` như `io_system2.md`): `s2_step` → latent
  → `s1_step_latent(rgbs, depths, latent)` → `dp_actions` (quỹ đạo).
- **E2.** ⬜ Thu output từng frame: `pred_pixel`, `pred_action`, `traj` (dp_actions), kèm GT (`action`, `goal`).
- **E3.** 🚨 Quản lý bộ nhớ: `torch.cuda.empty_cache()` định kỳ; đo thời gian/frame (session GPU ~9h). Bắt
  đầu ~5 scene.

### Phase F — Metric & xuất kết quả

- **F1.** ⬜ **Chỉ số S2 (có GT):** action accuracy (`pred_action[0]` vs `action`) + ma trận nhầm lẫn;
  pixel-goal L2 (map 384→640×480) + success@k, cho frame có `goal != (-1,-1)`.
- **F2.** ⬜ **S1 (không có GT liên tục trong `vln_ce`):** **visualize quỹ đạo** (overlay waypoint lên ảnh /
  vẽ bird-eye) để kiểm định tính; đếm tỉ lệ frame S1 sinh được quỹ đạo hợp lệ (không NaN/rỗng).
- **F3.** ⬜ Lưu `/kaggle/working/results.json` + `summary.json` + ảnh `traj_vis/*.png`.

### Phase G — Kiểm chứng & giới hạn

- **G1.** ⬜ Đối chiếu tay 3–5 frame (instruction + ảnh + pred) — output hợp lý không.
- **G2.** ⬜ Ghi rõ giới hạn: open-loop ≠ closed-loop; S1 **không có GT** trong `vln_ce` nên chỉ đánh giá
  định tính; 1 setting camera; depth `vln_ce` (640×480) khác depth train NavDP (D435i 480×270) → có thể
  lệch tỉ lệ, cần lưu ý.

---

## 6. Định nghĩa "Done"

- [ ] Notebook chạy end-to-end trên T4×2, mount 2 dataset, không tải mạng lúc chạy.
- [ ] Nạp **đủ dual-system** bằng `InternVLAN1ForCausalLM` — **verify `navdp.*` (S1) nạp đủ**, shard 2 GPU.
- [ ] Với ≥1 subset nhỏ: `s2_step → s1_step_latent` chạy được, **S1 xuất quỹ đạo**.
- [ ] Xuất `results.json` + `summary.json` + ≥3 ảnh visualize quỹ đạo.
- [ ] Có chỉ số S2 (action accuracy / pixel-goal L2 nếu bật được) + ghi chú giới hạn rõ ràng.

---

## 7. Rủi ro & cách xử lý

| # | Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|---|
| 1 | **`pip install -e .` hạ cấp `transformers` 5.0.0** | version đổi, xung đột `torch` | Đọc pin thật trong repo; cài đúng bản; **restart kernel**; chạy lại baseline |
| 2 | **flash-attn không chạy trên T4** | lỗi build/import flash_attn, hoặc CUDA error lúc forward | Ép `attn_implementation="sdpa"` (hoặc `"eager"`); có thể phải sửa dòng hardcode trong repo |
| 3 | **OOM** (S1+S2+KV-cache) | kernel chết im (swap=0) | Ép `device_map="auto"`; đo VRAM; 4-bit; **cuối cùng: server ≥24GB** |
| 4 | **S1 không nạp** (class vs config `qwen2_5_vl`) | `navdp.*` báo MISSING/UNEXPECTED | Chỉnh `config.json`/dựng model theo cách repo; đối chiếu `internvla_n1.py` |
| 5 | **Pixel-goal không bật → không có latent cho S1** | S2 ra `←←←…` | [Phase D](#phase-d--gỡ-blocker-pixel-goal--điều-kiện-cần-để-s1-chạy): resize 384 + conjunction; điều tra `s2_step` |
| 6 | Sai `repo_type` | HF 404 | Dataset **phải** `repo_type="dataset"` |
| 7 | Tải 0 file mà báo thành công | `Fetching 0 files`, `0.00B` | Đếm `fnmatch.filter` trước; dùng `hf_hub_download` (ném lỗi) |
| 8 | Tràn đĩa 20GB | `Errno 28 No space left` | `HF_HOME=/kaggle/temp`; giải nén ở `/kaggle/temp`; dọn trước Save Version |
| 9 | Lệch toạ độ / tỉ lệ depth | quỹ đạo/pixel vô lý | Kiểm `[row,col]` vs `[u,v]`; map 384→640×480; lưu ý depth `vln_ce` khác D435i |
| 10 | Chọn nhầm setting `0deg` | `goal` toàn `(-1,-1)` | Dùng setting pitch ≥ 15° |
| 11 | Session GPU khác session CPU | version lệch | Chạy lại cell baseline ở session GPU |

---

## 8. Câu hỏi mở & phụ thuộc

- ⬜ **Blocker #4:** `config.json` bản NavDP có field **`system1`** không? Nhánh dựng `self.navdp` trong
  `InternVLAN1MetaModel.__init__` rẽ theo nó (§3.1.b) — thiếu field là S1 bị vứt im lặng. Verify ở Phase B4.
- ⬜ **Nhánh non-async của `build_navdp`:** quote mới thấy `if 'async' in config.system1` — bản NavDP
  (không async) đi nhánh nào? Đọc đầy đủ `internvla_n1_arch.py` khi triển khai.
- ⬜ **Pixel-goal (Phase D):** có bật được không? Không bật thì không có latent → S1 không chạy.
- ⬜ **Bộ nhớ:** full dual-system + S1 compute có vừa T4×2 không, hay phải lên server ≥24GB?
- ⬜ **`get_policy(policy_name)`:** tên policy đúng cho NavDP là gì (`InternVLAN1Net`)? Config `mode` =
  `dual_system` hay `sync`? — đọc `internnav/configs` + `internvla_n1_agent.py`.
- ⬜ **Depth mismatch:** NavDP train trên depth D435i 480×270; `vln_ce` depth 640×480 → ảnh hưởng tỉ lệ
  quỹ đạo? (định tính chấp nhận được cho milestone đầu.)

---

## 9. Nguồn tham chiếu

- **Code repo (đọc 22/07):** `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
  (`InternVLAN1Net`: `s2_step`, `s1_step_latent`, **điểm load checkpoint ~dòng 33–40**),
  `internnav/agent/internvla_n1_agent.py` (`InternVLAN1Agent`, chạy observation tĩnh),
  `internvla_n1.py` (`InternVLAN1ForCausalLM`, `generate_latents`, `generate_traj`),
  `internvla_n1_arch.py` (`InternVLAN1MetaModel` — dựng `self.navdp` + `latent_queries`, `build_navdp`),
  `navdp.py` (`NavDP_Policy_DPT_CriticSum_DAT.load_model`, `navdp_pretrained`).
- `SETUP_NOTES.md` — môi trường Kaggle, đóng gói checkpoint, smoke-test T4×2, bẫy HF.
- `docs/io_system2.md` — I/O S2, prompt/parser, **vấn đề pixel-goal**.
- `docs/data_contract.md` — schema `vln_ce` (RGB **+ depth**, 5 setting).
- `docs/checkpoint_variants.md` — **NavDP vs DualVLN** (vì sao NavDP cần agent riêng, không phải async).
- `docs/vln_subsets_architecture.md` — `vln_ce`=S2 · `vln_n1`=S1 · `vln_pe`=baseline.
