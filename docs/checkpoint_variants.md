# So sánh biến thể InternVLA-N1: **NavDP** vs **DualVLN**

Tài liệu tham chiếu để **sau này so sánh / cân nhắc pivot** giữa hai bản dual-system. Kế hoạch tuần hiện
đi theo đường **NavDP** (bản `InternVLA-N1` đã đóng gói `tieulam/internvla-n1-ckpt`); DualVLN ghi lại ở
đây để dành, **không** phải hướng đang làm.

- **Ngày ghi:** 22/07/2026
- **Nguồn:** HF model cards + InternNav model zoo/technical report + **đo weight keys thật** (`model.safetensors.index.json`) + `config.json`.
- **Nguyên tắc:** số/khoá là đo thật; chỗ nào suy luận từ kiến trúc sẽ ghi rõ ⬜ *chưa xác minh*.

---

## 1. Ba cấu hình chính thức (khớp 3 repo HF)

Model zoo InternNav liệt kê **3 cấu hình**, ánh xạ đúng 3 repo HuggingFace:

| Repo HF | Tên trong model zoo | Vai trò | Kích thước |
|---|---|---|---|
| `InternRobotics/InternVLA-N1-System2` | InternVLA-N1 (System 2) | VLM đứng riêng, ghép S1 bất kỳ | 16.59 GB |
| `InternRobotics/InternVLA-N1` ⬅ **đang dùng** | Dual System **with NavDP\*** | S1 = NavDP, RGB-**D** | 16.79 GB |
| `InternRobotics/InternVLA-N1-DualVLN` | Dual System **DualVLN** | S1 = NextDiT-async, **RGB-only**, bản mới nhất | 16.77 GB |

---

## 2. Vì sao có hai bản dual-system (theo tài liệu InternRobotics)

- **Technical report:** *"release cung cấp hai cấu hình: InternVLA-N1 (Dual System) with NavDP\* và
  InternVLA-N1 (Dual System) DualVLN."* → **cố ý phát hành hai điểm thiết kế**, không phải bản cũ bị thay.
- **Dấu `*` trong NavDP\*** có định nghĩa: *"NavDP\* indicates joint tuning with System 2"* — tức NavDP đã
  được **đồng huấn luyện** với S2 (khác NavDP standalone).
- **NavDP là một trong nhiều S1 backbone:** docs xếp *"NavDP <InternVLA-N1 (System 1)>"* cạnh **iPlanner,
  ViPlanner, DD-PPO, GNM** → InternNav là framework hỗ trợ nhiều S1 mô-đun; NavDP là bản "nhà trồng",
  DualVLN thay S1 bằng NextDiT.
- **DualVLN** được mô tả (đại ý): *"latest dual-system architecture", "optimized end-to-end", "faster
  convergence"*, **RGB-only**, đạt **SOTA benchmark**.
- **Trục tiến hoá Preview → Official** (từ card chính `InternVLA-N1`): Preview đồng bộ ~2Hz; Official
  **async**, S1 huấn luyện trên bước dày hơn (~25 cm), tốt hơn về mượt/hiệu quả/zero-shot sim2real.
- ⚠️ **Điểm mờ trong docs:** quan hệ 3 repo ↔ NavDP vs NextDiT **không nằm gọn một chỗ** — phải ghép
  model zoo + tech report + 3 model card. **Card chính `InternVLA-N1` KHÔNG gọi tên NavDP** (trừu tượng
  hoá S1) → muốn biết S1 là gì phải đọc `config.json`/weight keys. *(ứng viên PR/doc-clarity nhẹ.)*

---

## 3. So sánh kiến trúc — theo **weight keys thật**

| Thành phần | **NavDP** (`InternVLA-N1`, key `model.language_model.navdp.*`) | **NextDiT-async** (`-DualVLN`, key `model.*` cấp cao) |
|---|---|---|
| **Bộ khử nhiễu (diffusion)** | `decoder.layers.0..15` — **decoder transformer 16 lớp** | `traj_dit` — **DiT 12 lớp**, dual-attn (`attn1` self + `attn2` cross), `time_caption_embed` (timestep+điều kiện kiểu adaLN), `patch_embedder`, `caption_projection` |
| **Mã hoá quan sát** | `rgbd_encoder`: **rgb_model (12) + depth_model (12)** + `former_net (2)` → **RGB *và* Depth** | `rgb_model (12, ViT)` + `rgb_resampler (3, query tokens kiểu Perceiver)` + `memory_encoder (3, có memory_pos)` → **RGB-primary, KHÔNG có nhánh depth riêng** |
| **Đầu ra action** | `action_head` + **`critic_head`** + `goal_compressor` | `action_encoder`/`action_decoder` (tokenise) + `cond_projector` (2 linear) — **KHÔNG có critic** |
| **Cầu nối S2→S1** | `latent_queries` | `latent_queries` (giống) |
| **Cách gắn vào model** | lồng **trong** `model.language_model.navdp.*` | **hạng nhất** ở cấp `model.*` |
| **`config.json`** | `model_type: qwen2_5_vl` (S1 "đi ké" tensor) | `model_type: internvla_n1`, `architectures: InternVLAN1ForCausalLM`, `system1: "nextdit_async"` |
| **`tokenizer.json` (fast)** | ✅ có | ❌ **không** (chỉ `vocab.json`+`merges.txt` → slow tokenizer) |
| **Train artifacts** (`trainer_state`…) | có | không |
| **Tổng safetensors** | 16.79 GB (shard4 = 1.888 GB) | 16.77 GB (shard4 = 1.875 GB) |
| **S2 backbone (shard 1–3)** | 4.966 / 4.991 / 4.933 | 4.968 / 4.991 / 4.933 → **gần trùng byte, chung S2** |

*(Lưu ý: `model.layers.0..27` ở cả hai là **28 lớp LLM của S2** — Qwen2.5-VL 7B, không phải S1.)*

---

## 4. Bốn khác biệt cốt lõi

1. **Backbone khử nhiễu:** NavDP dùng **decoder transformer 16 lớp**; NextDiT dùng **DiT** (kiểu
   Next-DiT/Lumina): patchify quỹ đạo, điều kiện qua `time_caption_embed`, self+cross attention.
2. **Depth:** NavDP có **`depth_model` riêng** (RGB-D hạng nhất); NextDiT **không có nhánh depth** trong
   key lấy được → RGB-only. *(Khớp việc demo phải sinh depth bằng DepthAnything V2.)*
3. **Critic:** NavDP có **`critic_head`** (sinh nhiều quỹ đạo rồi chấm điểm/chọn — chữ ký NavDP);
   NextDiT **bỏ critic**, action encode/decode qua token.
4. **Tích hợp & thực thi:** NavDP "bolt-on" trong module ngôn ngữ, config vẫn là VLM thuần (→ load bằng
   `Qwen2_5_VL` thì `navdp.*` rơi ra, xem `SETUP_NOTES.md` 2.4); NextDiT là **model custom hợp nhất**
   (`InternVLAN1ForCausalLM`) chạy **async**: S2 chạy định kỳ sinh `latent_queries`, `traj_dit` chạy tần
   số cao, `memory_encoder` giữ ngữ cảnh thời gian.

---

## 5. Hệ quả thực tế — khi nào cân nhắc bản nào

| | **NavDP** (đang dùng) | **DualVLN** (để dành) |
|---|---|---|
| Cảm biến | RGB-D (cần depth) | RGB-only |
| Chạy standalone/local | ✅ NavDP vốn standalone-được (hợp luồng Người A, laptop) | ⬜ *nghi là không* — coupled S2, phải verify |
| Data System 1 | `vln_n1` (SE(3) 4×4, có depth) — đã đo | ⬜ *chưa rõ dùng data nào* |
| Đường chạy dual-system | agent NavDP | `inference_only_demo.ipynb` + `InternVLAN1AsyncAgent` |
| Chi phí lên Kaggle | đã đóng gói xong | +tải/đóng gói 16.77GB, +cài repo (hạ cấp `transformers`→4.51), +flash-attn (T4 sm_75 rủi ro), +DepthAnything |
| Tokenizer | fast (an toàn với `transformers` 5.x) | slow → nên pin `transformers==4.51.0` |
| **System 2** | **chung** — `io_system2.md` + data contract `vln_ce` dùng được cho **cả hai** | **chung** |

**Chốt:** S2 giống hệt nhau nên mọi việc S2 (Ngày 2 `io_system2.md`, contract `vln_ce`) **transferable**
giữa hai bản. Khác nhau nằm trọn ở **System 1** và cách đóng gói/chạy.

---

## 5.b. (22/07) Vị trí code load checkpoint S1+S2 — đọc từ source

Truy được chuỗi load đầy đủ của đường dual-system (áp cho **cả hai** biến thể — khác nhau ở giá trị
`config.system1`). **Kết luận: S1 không có file weights riêng** — nằm trong 4 shard chung, được nạp bởi
cùng một `from_pretrained`, với điều kiện module `self.navdp` đã được dựng trước trong kiến trúc.

| # | File | Class / method | Vai trò |
|---|---|---|---|
| 1 | `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` (~dòng 33–40) | `InternVLAN1Net.__init__` | **Điểm load duy nhất:** `InternVLAN1ForCausalLM.from_pretrained(model_config.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": device})` + `AutoTokenizer(use_fast=True)` + `AutoProcessor` |
| 2 | `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` | `InternVLAN1MetaModel.__init__` | **Dựng S1 theo `config.system1`:** `elif 'navdp' in config.system1: if 'async' in config.system1: self.navdp = build_navdp(config, memory_size=2)` — `build_navdp` = `NavDP_Policy_DPT_CriticSum_DAT(memory_size, navdp_version=0.1)` + `navdp.load_model()`. Cũng tạo `self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))` |
| 3 | `internnav/model/basemodel/internvla_n1/navdp.py` | `NavDP_Policy_DPT_CriticSum_DAT.load_model()` | Loader **standalone** (tuỳ chọn): `torch.load(self.navdp_pretrained)` nếu path set; `navdp_pretrained=None` → random init, sau đó bị bước 1 **ghi đè** bằng `model.language_model.navdp.*` trong checkpoint lớn |

**Ba hệ quả:**

1. Khớp phát hiện `SETUP_NOTES.md` 2.4: load bằng class HF thuần thì kiến trúc **không có chỗ chứa**
   `navdp.*` → tensor bị `UNEXPECTED`. Dùng `InternVLAN1ForCausalLM` thì bước 2 dựng chỗ chứa → weights vào đủ.
2. 🚨 Lời gọi gốc **`device_map={"": device}`** pin toàn bộ 16.79GB vào **một** GPU → OOM chắc chắn trên
   1 T4 (15GB). Chạy Kaggle T4×2 phải override `device_map="auto"`.
3. 🚨 Nhánh dựng S1 đọc **`config.system1`** — `-DualVLN` có (`"nextdit_async"`, mục 3), còn bản NavDP
   là `model_type: qwen2_5_vl` và ⬜ **chưa thấy field `system1`** trong survey config. Nếu thiếu →
   `self.navdp` không được dựng → `navdp.*` bị vứt im lặng **y như** load bằng class thuần.
   **Verify bắt buộc trước khi tin "đã nạp đủ dual-system":** đọc `config.json` bundle; sau load đếm
   `sum(p.numel() for n,p in model.named_parameters() if 'navdp' in n) > 0` + soi log `MISSING`/`UNEXPECTED`.

---

## 6. ⬜ Câu hỏi mở phải verify TRƯỚC nếu sau này muốn pivot sang DualVLN

Ghi sẵn để lần sau khỏi điều tra lại từ đầu:

1. **NextDiT S1 có chạy standalone / chạy local (≤6GB) được không?** — quyết định luồng "local" của Người
   A còn tồn tại hay cả nhóm phải lên cloud. *(Đọc `internvla_n1_agent_realworld.py` + xem `traj_dit` có
   tách rời S2 không.)*
2. **DualVLN có thật sự bỏ depth không**, hay pipeline vẫn cần DepthAnything cho mục đích khác? (mâu thuẫn:
   S1 khai RGB-only nhưng demo vẫn dùng DepthAnything V2.)
3. **`InternVLAN1AsyncAgent` có hỗ trợ `device_map`/đa GPU không?** Nếu dồn `cuda:0` → 16.77GB **OOM trên
   1 T4 15GB** → biến **server ≥24GB thành phụ thuộc cứng** cho Ngày 4.
4. **NextDiT S1 huấn luyện trên subset data nào?** `vln_n1` là data của **NavDP** — chưa chắc là data của
   NextDiT. Ảnh hưởng trực tiếp **data contract** với nhóm SIM.
5. **flash-attn 2.7.3 trên T4 (sm_75)** — có bắt buộc không, có ép được `attn_implementation="sdpa"/"eager"` không.
6. **`config.json` bản NavDP có field `system1` không?** (5.b hệ quả 3) — nếu không, phải patch config /
   truyền override trước `from_pretrained`, nếu không S1 bị vứt im lặng **dù dùng đúng class repo**.
   Kèm theo: nhánh **non-async** của `build_navdp` (quote mới thấy `if 'async' in config.system1`) —
   bản NavDP đi nhánh nào?

---

## 7. Nguồn tham chiếu

- `https://huggingface.co/InternRobotics/InternVLA-N1` — model card chính (Preview→Official, async)
- `https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN` — model card DualVLN
- `https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN/raw/main/config.json` — `system1: nextdit_async`
- `.../InternVLA-N1-DualVLN/raw/main/model.safetensors.index.json` — weight keys NextDiT
- InternNav repo README (model zoo) + `scripts/notebooks/inference_only_demo.ipynb`
- `SETUP_NOTES.md` mục 2.4 (weight keys NavDP), 3.1/3.6/3.9 (survey), 3.15 (bản ghi ngày phát hiện)
- `internnav/model/basemodel/internvla_n1/{internvla_n1_policy.py, internvla_n1_arch.py, navdp.py}` —
  chuỗi load checkpoint S1+S2 (mục 5.b, đọc 22/07)
