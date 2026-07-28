# 02 — Hệ thống InternVLA-N1: hai bộ não, ba bộ data, hai giai đoạn huấn luyện

> **File này để làm gì:** cho bạn **bức tranh lớn** — hệ thống gồm những mảnh nào, mảnh nào gọi mảnh
> nào, và vì sao **mỗi bộ não phải ăn một loại data riêng**. Hiểu file này rồi thì 03/04/05 chỉ là
> chi tiết.
>
> Bộ tài liệu: [00_README](00_README.md) · [01_thuat_ngu](01_thuat_ngu.md) ·
> [03_code_train_s2](03_code_train_s2.md)

---

## 1. Ý tưởng gốc: "nghĩ chậm" và "phản xạ nhanh"

Lấy cảm hứng từ tâm lý học (*Thinking, Fast and Slow*):

- **System 2 — nghĩ chậm, có lý trí.** Khi bạn đọc bản đồ và cân nhắc *"à, phải rẽ ở chỗ cái cây
  kia"*. Trong robot: **nhìn ảnh + đọc câu lệnh → quyết định đích đến**.
- **System 1 — phản xạ nhanh, bản năng.** Khi đang đi mà tự né cái ghế không cần suy nghĩ. Trong
  robot: **từ đích đến → vẽ đường đi mượt né vật cản**, làm nhiều lần mỗi giây.

Robot cần **cả hai**: S2 để biết *đi đâu*, S1 để lo *đi thế nào cho khỏi đâm*.

---

## 2. Bản đồ các mảnh code (mảnh nào ở đâu)

```
InternNav/code/
├── scripts/train/                       ← "PHÒNG ĐIỀU KHIỂN": chọn model, đặt tham số, bấm nút
│   ├── base_train/                          • CMA, Seq2Seq, RDP, NavDP (model "cơ bản", 1–8 GPU)
│   │   ├── start_train.sh, train.py
│   │   └── configs/*.py                     • công thức siêu tham số từng model
│   └── qwenvl_train/                        • InternVLA-N1 (7B, cấu hình gốc 64 GPU)
│       ├── train_system2.sh                 ← GIAI ĐOẠN 1: dạy System 2
│       ├── train_dual_system.sh             ← GIAI ĐOẠN 2: gắn + dạy System 1
│       ├── train_system2_vlln.sh            ← biến thể hội thoại nhiều lượt
│       └── zero2.json / zero3.json / zero3_offload.json   • cấu hình DeepSpeed
│
└── internnav/                           ← "ĐỘNG CƠ THẬT"
    ├── trainer/internvla_n1_trainer.py      • vòng lặp huấn luyện N1  ← 03 mổ xẻ file này
    ├── trainer/internvla_n1_argument.py     • định nghĩa mọi cờ dòng lệnh
    ├── dataset/internvla_n1_lerobot_dataset.py  • ĐỌC DATA S2 (NavPixelGoalDataset)
    ├── dataset/navdp_lerobot_dataset.py         • ĐỌC DATA S1 (NavDP_Base_Datset)
    ├── model/basemodel/internvla_n1/            • kiến trúc model (S2 + S1 ghép lại)
    └── configs/model/internvla_n1.py            • ⚠️ config cho EVAL, KHÔNG dùng khi train
```

> **Nguyên tắc thiết kế cần nhớ:** `scripts/train` chỉ *chọn và chỉ đường*; toàn bộ logic thật nằm
> trong package `internnav/`. Muốn đổi thí nghiệm → sửa script/config, **không đụng** code lõi.

---

## 3. Ba bộ data con — không thay thế được cho nhau

Người mới hay tưởng "có một dataset của InternVLA-N1". **Sai.** Bộ `InternData-N1` có **ba bộ con**
nuôi **ba model khác nhau**:

| Bộ con | Nuôi model nào | Dấu hiệu nhận dạng | Loader riêng |
|---|---|---|---|
| **`vln_ce`** | **System 2** | cột `pose.{setting}` / `goal.{setting}`; có sẵn RGB (.jpg) + depth (.png) | `internvla_n1_lerobot_dataset.py` |
| **`vln_n1`** | **System 1** (NavDP) | cột `observation.camera_intrinsic` / `camera_extrinsic`; có `meta/pointcloud.ply` | `navdp_lerobot_dataset.py` |
| **`vln_pe`** | baseline CMA/RDP — **không** thuộc dual-system | cột `observation.robot_position/orientation/yaw/progress/step` | `cma_lerobot_dataset.py` |

> **Bằng chứng:** mỗi loader chỉ đọc đúng bộ cột của một bộ con. Đưa nhầm bộ → loader tìm không thấy
> cột → **bỏ nguyên scene và im lặng** (lỗi bị `try/except` nuốt tại
> [internvla_n1_lerobot_dataset.py:816](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L816))
> → dataset rỗng → train không chạy mà không hiểu vì sao.

---

## 4. Model: ai kế thừa ai

### 4.1. Tên đường dẫn quyết định class model

Trainer **rẽ 3 nhánh bằng cách so khớp chuỗi trong `--model_name_or_path`**
([internvla_n1_trainer.py:149-181](../../../code/internnav/trainer/internvla_n1_trainer.py#L149)):

```python
if 'internvla-n1-system2' in model_name_or_path.lower():
    model = InternVLAN1ForCausalLM.from_pretrained(...)          # model TÙY BIẾN của dự án
elif "qwen2.5" in model_name_or_path.lower():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)  # Qwen2.5-VL GỐC
else:
    model = Qwen2VLForConditionalGeneration.from_pretrained(...)     # Qwen2-VL (đời cũ)
```

> ⚠️ **Bẫy số 1:** nếu bạn đổi tên thư mục checkpoint thành `checkpoints/my_ckpt`, code rơi vào
> nhánh `else` → nạp **nhầm** kiến trúc Qwen2-VL đời cũ + image processor sai. Tên thư mục **phải
> chứa chuỗi** `internvla-n1-system2` (không phân biệt hoa/thường).

### 4.2. `InternVLAN1ForCausalLM` = System 2 + phần lắp thêm

Tại [internvla_n1.py:39](../../../code/internnav/model/basemodel/internvla_n1/internvla_n1.py#L39):

```python
class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
```

- `Qwen2_5_VLForConditionalGeneration` = **System 2** (bộ não Qwen2.5-VL gốc của HuggingFace).
- `InternVLAN1MetaForCausalLM` = phần **mở rộng System 1** do dự án tự viết
  ([internvla_n1_arch.py](../../../code/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py)).

Khi model thuộc loại `internvla-n1`, trainer gọi `initialize_vision_modules()`
([trainer dòng 206](../../../code/internnav/trainer/internvla_n1_trainer.py#L206)) để **lắp System 1**:

| `--system1` chứa | Lắp gì | Ghi chú |
|---|---|---|
| `nextdit` | `build_traj_dit()` — Diffusion Transformer sinh quỹ đạo | mặc định của `train_dual_system.sh` (`nextdit_async`) |
| `navdp` | `build_navdp()` — nạp model NavDP | biến thể |
| `none` | **không lắp gì** | dùng khi chỉ train S2 |

Ngoài ra S1 dùng bộ trích đặc trưng ảnh **DepthAnythingV2**
([internvla_n1_arch.py:28-40](../../../code/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py#L28))
→ cần file trọng số `checkpoints/depth_anything_v2_metric_hypersim_vits.pth`.

Và một tham số học được tên **`latent_queries`** — chính là **cầu nối** S2 → S1.

### 4.3. Bảng ánh xạ: mỗi script gọi model nào

| Script | `--model_name_or_path` | Model nạp | `--system1` | `--pixel_goal_only` | Ý nghĩa |
|---|---|---|---|---|---|
| [train_system2.sh](../../../code/scripts/train/qwenvl_train/train_system2.sh) | `Qwen/Qwen2.5-VL-7B-Instruct` | Qwen2.5-VL **gốc** | `none` | `False` | **Chỉ huấn luyện System 2** |
| [train_dual_system.sh](../../../code/scripts/train/qwenvl_train/train_dual_system.sh) | `checkpoints/InternVLA-N1-System2` | `InternVLAN1ForCausalLM` | `nextdit_async` | `True` | Gắn & huấn luyện **System 1** lên trên S2 đã có |
| [train_system2_vlln.sh](../../../code/scripts/train/qwenvl_train/train_system2_vlln.sh) | `Qwen/Qwen2.5-VL-7B-Instruct` | Qwen2.5-VL gốc | `none` | — | Biến thể hội thoại đa lượt (dùng data `iign_*`) |

---

## 5. Cụm từ gây rối nhất: "action" có HAI nghĩa

| | **action của System 2** | **action của System 1** |
|---|---|---|
| Kiểu | **số nguyên rời rạc** `{0,1,2,3,5}` | **ma trận 4×4** mỗi frame |
| Ý nghĩa | "bấm 1 trong 5 nút" | "vẽ một nét bút liền" = quỹ đạo |
| Ở đâu | cột `action` của `vln_ce` | cột `action` của `vln_n1` |

Khi đọc code hay dataset, luôn tự hỏi *"action này của hệ nào?"*.

---

## 6. Hai giai đoạn huấn luyện (thứ tự bắt buộc)

```
 GIAI ĐOẠN 1 — train_system2.sh                    GIAI ĐOẠN 2 — train_dual_system.sh
 ┌──────────────────────────────┐                  ┌──────────────────────────────────┐
 │ Nạp: Qwen2.5-VL-7B (gốc)     │                  │ Nạp: checkpoint S2 vừa xong      │
 │ Data: vln_ce (r2r/rxr/…)     │  ──checkpoint──► │ ĐÓNG BĂNG toàn bộ S2             │
 │ Học: TOÀN BỘ S2              │                  │ Học: chỉ các module S1           │
 │   tune_mm_vision/mlp/llm=True│                  │   tune_mm_* = False              │
 │ Nhãn: pixel goal / turn/ stop│                  │ Nhãn: + quỹ đạo (traj_poses)     │
 └──────────────────────────────┘                  └──────────────────────────────────┘
```

Cơ chế "học phần nào" nằm ở hàm `set_model()`
([trainer:78-122](../../../code/internnav/trainer/internvla_n1_trainer.py#L78)) — nó bật/tắt
`requires_grad` cho từng nhóm tham số. Chi tiết từng dòng: [03](03_code_train_s2.md) mục 3.

> 💡 Bạn **không bắt buộc** phải chạy giai đoạn 1: dự án phát hành sẵn checkpoint
> `InternVLA-N1-System2` trên HuggingFace, tải về rồi nhảy thẳng vào giai đoạn 2.

---

## 7. Config đến từ đâu (điểm hay nhầm)

Nhánh `qwenvl_train` **không dùng file config `.py`**. Cấu hình đến từ **3 nguồn**:

| Nguồn | Nội dung | File |
|---|---|---|
| **Tham số dòng lệnh** trong `.sh` | gom vào 3 dataclass: `ModelArguments`, `DataArguments`, `TrainingArguments` | [internvla_n1_argument.py](../../../code/internnav/trainer/internvla_n1_argument.py) |
| **DeepSpeed JSON** | cách chia sẻ bộ nhớ GPU | `scripts/train/qwenvl_train/zero2.json` |
| **`data_dict` trong code** | danh sách dataset hợp lệ + cấu hình camera | [internvla_n1_lerobot_dataset.py:127-144](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L127) |

> ⚠️ **Bẫy số 2:** file [internnav/configs/model/internvla_n1.py](../../../code/internnav/configs/model/internvla_n1.py)
> chỉ có vài dòng và dùng cho **giai đoạn đánh giá/triển khai (agent)** — **KHÔNG** liên quan tới
> luồng training. Đừng nhầm hai thứ.

---

## 8. FAQ: data của S1 và S2 có phải cùng một video không?

> *"Nếu tôi quay cùng một căn phòng ở hai buổi khác nhau, mỗi buổi làm data cho một hệ, có dùng được không?"*

**Trả lời ngắn: KHÔNG cần chung video, KHÔNG cần đồng bộ giữa hai hệ.** Nhưng có một loại "đồng bộ"
khác thì **bắt buộc**.

### 8.1. Data gốc vốn đã không chung video

| | Data S2 (`vln_ce`) | Data S1 (`vln_n1`) |
|---|---|---|
| Bộ cảnh | r2r, rxr, scalevln | 3dfront, gibson, hm3d, hssd, matterport3d, replica |
| Camera | cấu hình 60/125 cm có góc cúi | mô phỏng D435i / ZED |

→ Nhóm tác giả **không** quay chung một video rồi tách. Kịch bản "cùng phòng, khác buổi" của bạn
thậm chí *khớp hơn* data gốc.

### 8.2. Cái KHÔNG cần vs cái BẮT BUỘC

| Loại đồng bộ | Cần? | Vì sao |
|---|---|---|
| Giữa data S1 ↔ data S2 (frame t bên này ↔ frame t bên kia) | ❌ **Không** | Hai hệ học hai bài toán độc lập, loader riêng, loss riêng. Chúng chỉ gặp nhau lúc **inference** qua giao diện latent đã thiết kế sẵn. |
| **Bên trong một lượt quay**: RGB + depth + pose + goal của cùng một frame phải cùng khoảnh khắc | ✅ **Bắt buộc** | Depth lệch pha với RGB, hoặc pose không khớp thời điểm ảnh → **nhãn sai mà không báo lỗi** → model học rác. Đây chính là lý do pipeline mcap phải có bước đồng bộ thời gian ([06](06_pipeline_mcap_to_s2.md) giai đoạn B). |

### 8.3. Điều kiện tinh tế: NHẤT QUÁN QUY ƯỚC

Không cần chung video, nhưng để lúc chạy thật S2 bàn giao được cho S1, hai bên nên **cùng quy ước**:
chiều cao & góc cúi camera, model camera/intrinsic, hệ toạ độ, và **đơn vị depth** (S2 chia `1000`,
S1 chia `10000` — mỗi hệ giữ đúng hằng số của mình). Đừng buổi này camera cao 125 cm cúi 30°, buổi
kia cao 30 cm nhìn thẳng.

### 8.4. Bẫy riêng cho "cùng phòng, khác buổi"

Nếu giữa hai buổi **bàn ghế bị xê dịch**: với *train* thì không sao. Chỉ có vấn đề nếu bạn dùng
**chung một `pointcloud.ply`** cho cả hiện trường cũ lẫn mới. Quy tắc: **mỗi lượt quay của S1 phải
kèm bản đồ 3D của đúng hiện trường lúc đó**.

---

Đã có bức tranh lớn. Tiếp theo, đi vào code: [03_code_train_s2](03_code_train_s2.md).
