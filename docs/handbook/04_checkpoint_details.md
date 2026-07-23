# 04 — Checkpoint Details: các bản checkpoint InternVLA-N1 trên HuggingFace — bản nào để eval, bản nào để train

> **File này để làm gì:** liệt kê **từng checkpoint** liên quan InternNav/InternVLA-N1 trên
> HuggingFace (kèm link), giải thích bản nào dùng vào việc gì, bản nào chạy **eval** được ngay,
> bản nào là điểm xuất phát để **train với dữ liệu mới**.
>
> **Nguồn:** danh sách lấy từ HuggingFace API ngày **23/07/2026** (PL-B6 trong
> [05_appendix](05_appendix.md)) + Model Zoo trong README repo InternNav v0.3.1 + các số đo
> 21–22/07 (PL-B1…B5). Chỗ nào chưa xác minh sẽ đánh dấu ⬜.
>
> ⚠️ **Sự kiện phải biết trước khi đọc:** repo `InternRobotics/InternVLA-N1` (bản nhóm đã survey và
> đóng gói ngày 21/07) **đã bị đổi tên trên HF thành `InternVLA-N1-wo-dagger`** — link cũ giờ
> redirect sang tên mới (xác minh 23/07, PL-B6). Các tài liệu cũ trong `docs/` viết trước sự kiện
> này nên vẫn gọi nó là "`InternVLA-N1`".

---

## 0. Bảng tổng quan — nhìn một phát biết bản nào làm gì

7 checkpoint họ InternNav trên HF (tình trạng 23/07/2026):

| # | Repo HF (link) | Nội dung | Size | `config.json` | Eval? | Train? |
|---|---|---|---|---|---|---|
| 1 | [InternVLA-N1-w-NavDP](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP) | **Dual-system: S2 + S1 NavDP** (RGB-D), bản chính thức | 16.78 GB | `internvla_n1`, **`system1: "navdp_async"`** ✅ | ✅ **khuyến nghị** cho dual NavDP | ❌ (là sản phẩm cuối) |
| 2 | [InternVLA-N1-DualVLN](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN) | **Dual-system: S2 + S1 NextDiT** (RGB-only), bản mới nhất, SOTA | 16.77 GB | `internvla_n1`, `system1: "nextdit_async"` ✅ | ✅ (demo notebook chính chủ) | ❌ |
| 3 | [InternVLA-N1-wo-dagger](https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger) *(tên cũ: `InternVLA-N1`)* | Dual-system NavDP, bản **chưa qua DAgger** (yếu hơn #1) | 16.79 GB | `qwen2_5_vl`, **KHÔNG có `system1`** 🚨 | ⚠️ được, nhưng phải patch config | ❌ |
| 4 | [InternVLA-N1-System2](https://huggingface.co/InternRobotics/InternVLA-N1-System2) | **Chỉ System 2** (VLM Qwen2.5-VL-7B fine-tuned) | 16.59 GB | `qwen2_5_vl` | ✅ cho eval S2-only | ✅ **điểm xuất phát của `train_dual_system.sh`** |
| 5 | [InternVLA-N1-System2-wo-dagger](https://huggingface.co/InternRobotics/InternVLA-N1-System2-wo-dagger) | Chỉ System 2, bản chưa DAgger | ~16.6 GB | `qwen2_5_vl` | ✅ (bản yếu hơn #4) | ⬜ |
| 6 | [InternVLA-N1-Preview](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) | Bản preview đầu tiên (dual sync ~2Hz, cũ) | ~16.78 GB | `internvla_n1`; ⬜ `system1` chưa đọc được (401) | ⚠️ chỉ để tham khảo | ❌ |
| 7 | [VLN-PE](https://huggingface.co/InternRobotics/VLN-PE) | **Baseline** Seq2Seq / CMA / RDP (+bản `_plus`, `zero_shot`) | 1.86 GB | riêng từng thư mục, **4/7 thiếu `config.json`** 🚨 | ✅ baseline đối chứng | ✅ có config train trong repo |
| — | Kaggle Dataset [`tieulam/internvla-n1-ckpt`](https://www.kaggle.com/datasets/tieulam/internvla-n1-ckpt) | **Bản mirror của #3** nhóm tự đóng gói 21/07 để khỏi tải lại 16.79 GB mỗi session | 16.79 GB | = #3 (thiếu `system1`) | ⚠️ như #3 | ❌ |

Ngoài ra Model Zoo còn liệt kê **NavDP standalone** (System 1 tách rời) và các baseline VN khác
(iPlanner, ViPlanner, DD-PPO, GNM, ViNT, NoMad) — chúng nằm ở **GitHub repo
[InternRobotics/NavDP](https://github.com/InternRobotics/NavDP)**, không phải HF (theo link trong
README InternNav v0.3.1, mục Model Zoo).

---

## 1. Kiến thức nền để đọc bảng trên (dành cho người mới)

1. **Dual-system** = 2 model trong 1 checkpoint: System 2 (VLM 7B "nghĩ chậm") + System 1
   (policy "phản xạ nhanh" sinh quỹ đạo). Cả hai nằm chung trong 4 file shard — S1 không có file
   riêng (PL-C1). Có thể đọc thêm: [02_code_structure](02_code_structure.md) mục 0–1.
2. **Hai kiến trúc System 1 khác nhau** đang tồn tại song song (đây là phát hành có chủ đích của
   InternRobotics, không phải bản cũ/mới thay nhau — technical report nêu rõ 2 cấu hình):
   - **NavDP** — diffusion policy, cần **RGB-D** (có nhánh encoder depth riêng, có critic chấm điểm
     quỹ đạo). Weight keys: `model.language_model.navdp.*` (đo thật PL-B3).
   - **NextDiT** — DiT (diffusion transformer), **RGB-only**, không critic. Weight keys: `traj_dit`,
     `memory_encoder`, `rgb_resampler`… (đo thật PL-B5).
3. **DAgger** (viết tắt của Dataset Aggregation) — kỹ thuật huấn luyện bổ sung; hậu tố `-wo-dagger`
   = "without DAgger" = bản **chưa** qua bước đó. Benchmark trong README cho thấy bản wo-dagger yếu
   hơn rõ rệt (mục 4 dưới đây).
4. **Field `system1` trong `config.json` quyết định sống còn:** code dựng kiến trúc đọc field này để
   biết có tạo module S1 hay không (`internvla_n1_arch.py:124–143` — PL-C2). Config thiếu field →
   load xong chỉ có S2, **S1 bị vứt im lặng không báo lỗi**. Đây là khác biệt quan trọng nhất giữa
   #1 (có field) và #3 (không có field).

---

## 2. Chi tiết từng bản

### 2.1. `InternVLA-N1-w-NavDP` — bản dual-system NavDP "chuẩn chỉnh" ✅ khuyến nghị cho eval

- **Link:** <https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP>
- **Là gì:** bản chính thức của cấu hình *"InternVLA-N1 (Dual System) with NavDP\*"* trong Model Zoo
  — dấu `*` nghĩa là NavDP đã được **joint-tune cùng System 2** (định nghĩa trong README/tech report).
- **Đăng:** 10/12/2025 (PL-B6). Size 16.78 GB (8.39B params bf16; shard 4.965/4.991/4.933/1.888).
- **`config.json` (fetch 23/07):** `model_type: "internvla_n1"`, `architectures: InternVLAN1ForCausalLM`,
  **`system1: "navdp_async"`**, `n_query: 4` → load bằng `InternVLAN1ForCausalLM` là kiến trúc S1
  được dựng đúng, weights `navdp.*` vào đủ — **không cần patch gì**.
- **Cần biết trước khi dùng:**
  - ❌ **Không có `tokenizer.json`** (fast tokenizer) — chỉ `vocab.json`+`merges.txt` (kiểm đủ 16 file,
    PL-B4) → với `transformers` 5.x có rủi ro đường slow-tokenizer; phương án an toàn là pin
    `transformers==4.51.*` khi dùng bản này (cùng loại rủi ro đã ghi cho DualVLN).
  - Sensor: cần **cả RGB lẫn depth** (NavDP là RGB-D) — data `vln_ce` có sẵn cả hai (PL-D4).
  - ⬜ Nhóm **chưa tải/chạy bản này** — mọi thông tin trên là từ HF API + config, chưa phải số đo runtime.

### 2.2. `InternVLA-N1-DualVLN` — bản dual-system mới nhất, S1 NextDiT, RGB-only

- **Link:** <https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN>
- **Là gì:** cấu hình *"InternVLA-N1 (Dual System) DualVLN"* — kiến trúc dual mới nhất, đạt số
  **tốt nhất trên mọi benchmark** trong README (mục 4). S1 là NextDiT, chỉ cần RGB.
- **Size:** 16.77 GB (không phải ~8GB như comment trong notebook demo — đã đo, PL-B5).
- **`config.json`:** `system1: "nextdit_async"` ✅ (PL-B5) → load bằng class repo là đủ dual-system.
- **Đường chạy chính chủ:** notebook `scripts/notebooks/inference_only_demo.ipynb` + agent
  `InternVLAN1AsyncAgent` (`internnav/agent/internvla_n1_agent_realworld.py:26+`). Config eval mặc
  định của repo v0.3.1 cũng đang trỏ bản này (`habitat_dual_system_cfg.py:9` — file 02 mục 2).
- **Cần biết trước khi dùng:**
  - ❌ Không có `tokenizer.json` (PL-B4) → nên pin `transformers==4.51.*`.
  - Demo realworld dùng thêm **DepthAnything V2** để sinh depth từ RGB camera thật (S1 RGB-only
    nhưng pipeline demo vẫn dựng depth — điểm mâu thuẫn đã ghi ở `../checkpoint_variants.md` §6.2,
    ⬜ chưa giải đáp).
  - Agent realworld cũng pin 1 GPU như code policy (PL-C3) → Kaggle T4×2 phải override.

### 2.3. `InternVLA-N1-wo-dagger` (tên cũ `InternVLA-N1`) — bản nhóm ĐANG có sẵn trên Kaggle

- **Link:** <https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger>
  (link cũ `.../InternVLA-N1` redirect về đây — PL-B6)
- **Là gì:** dual-system NavDP **chưa qua DAgger** — tức cùng kiến trúc với #1 nhưng huấn luyện
  ít bước hơn, điểm benchmark thấp hơn (mục 4). Đây chính là bản nhóm survey/tải/đóng gói ngày 21/07
  thành Kaggle Dataset **`tieulam/internvla-n1-ckpt`** (16.79 GB — PL-B1).
- **Điểm mạnh thực dụng:** ✅ có `tokenizer.json` (fast — an toàn với transformers 5.0.0, PL-B4),
  ✅ có `trainer_state.json` (đọc được lịch sử train), ✅ **đã nằm sẵn trên Kaggle** không tốn lượt
  tải 16.79 GB.
- **Điểm chết người:** 🚨 `config.json` là `model_type: qwen2_5_vl` và **KHÔNG có field `system1`**
  (xác minh 23/07 — PL-B6) → load bằng class repo thì S1 **vẫn** bị vứt im lặng (cơ chế PL-C2).
  Muốn chạy dual-system với bản này phải **patch config** (thêm `"system1": "navdp_async"`; giữ
  nguyên `n_query: 16` có sẵn của nó) rồi **verify** đếm params navdp > 0
  ([01_system_requirements](01_system_requirements.md) mục 3 bước 7). ⬜ Việc patch chưa được chạy thử.
- **Dùng khi nào:** eval S2-only ngay (đã smoke-test thành công — PL-B3), hoặc dual-system nếu chấp
  nhận rủi ro patch + chất lượng wo-dagger; tiết kiệm thời gian tải trên Kaggle.

### 2.4. `InternVLA-N1-System2` và `InternVLA-N1-System2-wo-dagger` — chỉ System 2

- **Link:** <https://huggingface.co/InternRobotics/InternVLA-N1-System2> ·
  <https://huggingface.co/InternRobotics/InternVLA-N1-System2-wo-dagger>
- **Là gì:** VLM Qwen2.5-VL-7B đã fine-tune cho VLN, **không kèm S1** (16.59 GB — đúng bằng phần S2
  của bản dual, shard 1–3 trùng byte — PL-B1). Model Zoo xếp vào mục *"InternVLA-N1 (System 2) +
  Decoupled System1"* — tức ghép với một S1 bất kỳ (NavDP standalone, iPlanner…) qua pixel-goal.
- **Cần biết:** bản `System2` ❌ không có `tokenizer.json` (PL-B4) — trớ trêu là bản
  `System2-wo-dagger` lại ✅ có (kiểm 23/07). Bản `System2` có `training_args.bin` (file pickle —
  đừng cố `torch.load` nó, không cần cho inference).
- **Dùng khi nào:** eval S2 đứng riêng (chấm pixel-goal/action accuracy); hoặc làm **model khởi đầu
  cho train dual-system** (mục 5 — đây là vai trò được script của repo dùng tường minh).

### 2.5. `InternVLA-N1-Preview` — bản preview lịch sử

- **Link:** <https://huggingface.co/InternRobotics/InternVLA-N1-Preview>
- **Là gì:** bản phát hành đầu (tạo 21/07/2025, sửa lần cuối 01/09/2025 — PL-B6). Theo model card
  chính (đọc 22/07): Preview chạy dual **đồng bộ ~2Hz**; bản Official sau đó chuyển **async** và
  S1 được train trên bước dày hơn (~25 cm) → mượt hơn, sim2real tốt hơn.
- `model_type: internvla_n1`, có `tokenizer.json`, ~16.78 GB. ⬜ Giá trị `system1` trong config chưa
  đọc được (endpoint raw trả 401 — PL-B6).
- **Dùng khi nào:** không khuyến nghị cho công việc mới — chỉ để đối chiếu lịch sử.

### 2.6. NavDP standalone (System 1 tách rời) — nằm ở GitHub, không phải HF

- **Link (theo README Model Zoo):** <https://github.com/InternRobotics/NavDP>
- Code InternNav có sẵn đường load standalone: `NavDP_Policy_DPT_CriticSum_DAT.load_model()` đọc
  `navdp_pretrained` (một file `.ckpt` riêng) — `navdp.py:116–125` (PL-C1). Tức S1 có thể chạy
  **không cần** checkpoint 16.79 GB — hướng đi cho máy yếu/laptop.
- ⬜ Nhóm chưa tải/chạy checkpoint NavDP standalone nào — chưa rõ file `.ckpt` phát hành ở đâu
  trong repo NavDP; cần khảo sát khi đi hướng này.

---

## 3. `VLN-PE` — gói baseline (không thuộc dual-system)

- **Link:** <https://huggingface.co/InternRobotics/VLN-PE> — 1.86 GB tổng (PL-B1).
- **Cấu trúc:** `r2r/{fine_tuned,zero_shot}/{cma,rdp,seq2seq,cma_plus,seq2seq_plus}/…` — mỗi thư mục
  một checkpoint nhỏ (`pytorch_model.bin` 0.13–0.57 GB).
- 🚨 **Bẫy đo được (PL trong `SETUP_NOTES.md` 3.8):** chỉ **3** thư mục có `config.json`
  (`fine_tuned/rdp`, `fine_tuned/cma`, `fine_tuned/seq2seq`). Bốn thư mục `zero_shot/*` và `*_plus`
  có weights nhưng **thiếu config** → `from_pretrained` trỏ thẳng vào sẽ fail. Lần đầu chạy baseline:
  **dùng `r2r/fine_tuned/cma` hoặc `r2r/fine_tuned/rdp`**.
- Tải tiết kiệm: `--exclude "*optimizer.pt" "*scheduler.pt" "*rng_state.pth" "*trainer_state.json"`
  (bỏ ~0.44 GB rác train).
- **Vai trò:** đối chứng SR/SPL + "lưới an toàn" khi dual-system tắc — chạy được không cần
  checkpoint 16.79 GB, data đầu vào là `vln_pe` ([03_data_contract](03_data_contract.md) mục 3).

---

## 4. Bản nào EVAL được — bảng quyết định

Số benchmark (từ README repo v0.3.1 — bảng VLN-CE R2R, để thấy chênh lệch giữa các bản):

| Bản | Observation | SR ↑ | SPL ↑ |
|---|---|---|---|
| InternVLA-N1-wo-dagger (Dual, NavDP\*) | RGB-D | 58.2 | 54.0 |
| InternVLA-N1 (Dual, NavDP\*) = `-w-NavDP` | RGB-D | 64.1 | 58.1 |
| InternVLA-N1 (Dual, DualVLN) | RGB | **64.3** | **58.5** |

**Chọn theo mục tiêu:**

| Bạn muốn | Dùng bản | Lý do & việc phải làm |
|---|---|---|
| Eval **S2-only** trên Kaggle ngay hôm nay | `tieulam/internvla-n1-ckpt` (mirror #3) | Đã mount sẵn, có fast tokenizer, smoke-test xong (PL-B3). Load bằng `Qwen2_5_VLForConditionalGeneration` là đủ. |
| Eval **dual-system NavDP** (đường nhóm đang theo) | **#1 `-w-NavDP`** (sạch nhất) hoặc #3 + patch config (đỡ tải 16.78 GB) | #1: config chuẩn sẵn nhưng phải tải mới + lo slow tokenizer. #3: có sẵn trên Kaggle nhưng 🚨 phải patch `system1` + verify (PL-C2, PL-B6) và chấp nhận điểm wo-dagger thấp hơn. |
| Eval **dual-system điểm cao nhất / RGB-only** | #2 `-DualVLN` | Theo notebook demo chính chủ; pin `transformers==4.51.*`; T4 phải né flash-attn (PL-C3, file 01). |
| Baseline đối chứng nhẹ | #7 `VLN-PE` (`fine_tuned/cma` hoặc `rdp`) | Không cần GPU to; tránh 4 thư mục thiếu config (mục 3). |

Mọi đường eval trên Kaggle đều phải qua checklist setup ở [01_system_requirements](01_system_requirements.md)
(T4×2, `device_map="auto"`, `sdpa`, verify S1).

---

## 5. Bản nào để TRAIN với dữ liệu mới — theo đúng script của repo

> Căn cứ: 3 script/config train có thật trong repo v0.3.1 (đọc local 23/07 — PL-D5 và trích dẫn
> dưới đây). Không suy diễn ngoài những gì script viết.

### 5.1. Train System 2 — bắt đầu từ **Qwen gốc**, KHÔNG phải từ checkpoint InternVLA

📄 `scripts/train/qwenvl_train/train_system2.sh` (nguyên văn các dòng chính):

```bash
llm=Qwen/Qwen2.5-VL-7B-Instruct        # ← model khởi đầu: Qwen BASE trên HF
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_...   # data vln_ce
run_name=InternVLA-N1-System2          # ← sản phẩm đầu ra
#SBATCH -N 8  /  --gres=gpu:8          # ← 8 node × 8 GPU = 64 GPU
```

→ Muốn train lại S2 với dữ liệu mới: chuẩn bị data theo **schema `vln_ce`**
([03_data_contract](03_data_contract.md) mục 1), đăng ký tên dataset trong
`internvla_n1_lerobot_dataset.py`, sửa `vln_datasets=`. **Chi phí 64 GPU là ngoài tầm thực tập** —
khuyến nghị của nhóm (ghi trong `../vln_subsets_architecture.md` mục 4): thay bằng chỉnh
prompt/few-shot trên checkpoint S2 có sẵn.

### 5.2. Train Dual-System (S1 + cầu nối) — bắt đầu từ **checkpoint `InternVLA-N1-System2`**

📄 `scripts/train/qwenvl_train/train_dual_system.sh` (nguyên văn các dòng chính):

```bash
llm=Qwen/Qwen2.5-VL-7B-Instruct
# system 1 options: nextdit_async, navdp_async, nextdit     ← comment nguyên văn trong script
system1=nextdit_async
system2_ckpt=checkpoints/InternVLA-N1-System2               # ← MODEL KHỞI ĐẦU

... internnav/trainer/internvla_n1_trainer.py \
    --model_name_or_path "${system2_ckpt}" \                 # load S2 đã train xong
    --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False \   # ĐÓNG BĂNG toàn bộ VLM
    ...
run_name=InternVLA-N1-DualVLN
```

Đọc ra công thức của InternRobotics: **lấy checkpoint System2 (#4) làm nền → đóng băng VLM →
train thêm System 1 + latent connector** trên data `vln_ce` (có sampling `%30`). Đổi
`system1=navdp_async` là cùng script train ra biến thể NavDP (đúng bộ ba lựa chọn trong comment).
→ **Đây là câu trả lời chuẩn cho "bản nào được dùng để train":** checkpoint
**`InternVLA-N1-System2`** là bản duy nhất được script của repo dùng làm điểm xuất phát.

### 5.3. Train System 1 standalone (NavDP) — từ đầu, trên data `vln_n1`

📄 `scripts/train/base_train/configs/navdp.py` (đọc local 23/07):

```python
il=IlCfg(
    epochs=1000, batch_size=32, lr=1e-4,
    load_from_ckpt=False, ckpt_to_load='',                        # mặc định: train từ đầu
    root_dir='data/datasets/InternData-N1/vln_n1/traj_data',      # ← DATA: vln_n1
    image_size=224, memory_size=8, predict_size=24, ...
)
```

→ Train S1 với dữ liệu mới: sinh data theo **schema `vln_n1`** ([03](03_data_contract.md) mục 2 —
action 4×4 SE(3), intrinsic camera thật), trỏ `root_dir`, chạy `scripts/train/base_train/train.py`.
Có tuỳ chọn nạp checkpoint sẵn (`ckpt_to_load` / `navdp_pretrained`) để fine-tune thay vì train từ
đầu. Đây là hướng train **rẻ nhất** (server ≥24GB là đủ, không cần 64 GPU) — khuyến nghị của nhóm
cho giai đoạn W4.

### 5.4. Train baseline — config có sẵn

`scripts/train/base_train/configs/{cma,rdp,seq2seq}.py` (+ bản `_plus`) — data theo schema `vln_pe`.
Dùng khi cần đối chứng do-it-yourself; checkpoint kết quả so được với `VLN-PE` phát hành (mục 3).

---

## 6. Câu hỏi mở còn lại (đừng quên khi làm tiếp)

1. ⬜ Patch `system1` cho bản đóng gói (#3) **chưa chạy thử** — làm Phase B4 của
   `../eval_plan_kaggle_s2.md` để chốt.
2. ⬜ `-w-NavDP` chưa tải — nếu quyết định chuyển sang, đóng gói lại Kaggle Dataset (~16.78 GB,
   tính lượt session). → **Đã có kế hoạch chi tiết:** [06_eval_plan_w_navdp_kaggle.md](06_eval_plan_w_navdp_kaggle.md).
3. ⬜ NavDP standalone `.ckpt` phát hành ở đâu trong repo NavDP GitHub — cần khảo sát cho luồng local.
4. ⬜ `system1` của bản Preview (config bị 401) — chỉ cần nếu ai đó định dùng Preview.
5. ⬜ Nhánh `"navdp"` không-async trong `internvla_n1_arch.py:141–143` **không dựng navdp**
   (PL-C2) — nếu gặp checkpoint nào khai `system1: "navdp"` (không async) thì phải điều tra thêm.
