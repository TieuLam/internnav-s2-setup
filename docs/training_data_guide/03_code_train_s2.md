# 03 — Giải thích code từng phần: **nhánh huấn luyện System 2**

> **File này để làm gì:** đi từ dòng lệnh `bash train_system2.sh` cho tới lúc trọng số được cập nhật,
> **giải thích từng mảnh code làm gì**. Trọng tâm là **nhánh train S2**; chỗ nào chỉ chạy khi train
> dual-system sẽ được đánh dấu 🔵 để bạn biết mà bỏ qua.
>
> Mọi trích dẫn có `file:line` — mở ra tự kiểm chứng được.
> Bộ tài liệu: [02_he_thong](02_he_thong.md) · [04_data_train_s2](04_data_train_s2.md)

---

## 0. Toàn cảnh: 7 chặng của một lần train S2

```
 ①  scripts/train/qwenvl_train/train_system2.sh
     └─ srun torchrun … internnav/trainer/internvla_n1_trainer.py --<hàng chục cờ>
 ②  internvla_n1_trainer.py :: train()                                        [dòng 125]
     ├─ HfArgumentParser → ModelArguments / DataArguments / TrainingArguments  [128]
     ├─ dựng phép biến đổi ảnh (augmentation + resize 384×384)                 [134-147]
     ├─ CHỌN & NẠP MODEL theo tên đường dẫn                                    [149-181]
     ├─ nạp tokenizer                                                          [197]
     ├─ set_model(): bật/tắt học từng phần                                     [207]
     ├─ make_supervised_data_module(): dựng Dataset + Collator                 [216]
     └─ Trainer(...).train()  → lưu checkpoint                                 [217-235]
 ③  internvla_n1_lerobot_dataset.py :: NavPixelGoalDataset                     [822]
     ├─ __init__:  đọc parquet → CẮT thành mẫu pixel_goal / turn / stop        [843-947]
     └─ __getitem__: mở ảnh → dựng đoạn hội thoại → tokenize                   [990-1132]
 ④  DataCollatorForSupervisedDataset: đệm + ghép batch                         [1185]
 ⑤  Trainer của HuggingFace: forward → loss → backward → cập nhật trọng số
 ⑥  DeepSpeed ZeRO-2: chia gradient/optimizer cho các GPU
 ⑦  Lưu ra checkpoints/<run_name>/
```

---

## 1. Chặng ① — `train_system2.sh`: bảng điều khiển

File: [scripts/train/qwenvl_train/train_system2.sh](../../../code/scripts/train/qwenvl_train/train_system2.sh)

```bash
#SBATCH -N 8  --gres=gpu:8        # xin 8 máy × 8 GPU = 64 GPU (cấu hình của nhóm tác giả)
deepspeed=scripts/train/qwenvl_train/zero2.json
llm=Qwen/Qwen2.5-VL-7B-Instruct  # ← quyết định NẠP MODEL NÀO (mục 3)
lr=2e-5 ; vision_tower_lr=5e-6 ; batch_size=2
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,...   # ← tên phải có trong data_dict
run_name=InternVLA-N1-System2 ; output_dir=checkpoints/${run_name}

srun torchrun ... internnav/trainer/internvla_n1_trainer.py --deepspeed ${deepspeed} ...
```

Các cờ **quan trọng nhất cho nhánh S2** (dòng 41-56):

| Cờ | Giá trị ở `train_system2.sh` | Nghĩa |
|---|---|---|
| `--vln_dataset_use` | danh sách dataset | Chọn data. Thêm `%30` vào cuối tên = **chỉ lấy ngẫu nhiên 30%** |
| `--tune_mm_vision / mlp / llm` | `True / True / True` | **Mở học toàn bộ System 2** |
| `--system1` | `"none"` | **Không lắp System 1** |
| `--pixel_goal_only` | `False` | Dùng cả 3 loại mẫu (pixel_goal + turn + stop) |
| `--num_history 8` | | Mỗi mẫu kèm 8 ảnh lịch sử |
| `--sample_step 4` | | Cứ 4 frame lấy 1 mẫu |
| `--resize_h/w 384` | | Ảnh resize về 384×384 trước khi vào model |
| `--data_augmentation True` | | Bật biến đổi màu/độ nét ngẫu nhiên |
| `--model_max_length 8192` | | Độ dài chuỗi token tối đa |

> ⚠️ **Bạn không cần 64 GPU.** Sửa `--nnodes`, `--nproc_per_node`, giảm `batch_size`, hoặc đổi sang
> `zero3_offload.json` để chạy trên ít GPU (đánh đổi tốc độ). Nhưng đây là model 7 tỷ tham số nên
> vẫn cần GPU khá (thường ≥ 24–40 GB VRAM/GPU).

---

## 2. Chặng ② — Ba nhóm tham số (`internvla_n1_argument.py`)

File chỉ 54 dòng, định nghĩa 3 dataclass mà `HfArgumentParser` đọc
([trainer:128](../../../code/internnav/trainer/internvla_n1_trainer.py#L128)):

| Nhóm | Chứa gì | Tham số đáng nhớ (kèm mặc định) |
|---|---|---|
| **`ModelArguments`** ([dòng 7](../../../code/internnav/trainer/internvla_n1_argument.py#L7)) | chọn model & phần nào được học | `model_name_or_path`, `tune_mm_llm/mlp/vision=False`, `system1='nextdit'`, `n_query=4` |
| **`DataArguments`** ([dòng 18](../../../code/internnav/trainer/internvla_n1_argument.py#L18)) | cách xử lý dữ liệu | `vln_dataset_use=''`, `sample_step=4`, `num_history=8`, `predict_step_num=32`, `pixel_goal_only=False`, `resize_h/w=384`, `num_future_steps=4` |
| **`TrainingArguments`** ([dòng 45](../../../code/internnav/trainer/internvla_n1_argument.py#L45)) | vòng lặp huấn luyện (kế thừa HuggingFace) | `model_max_length=512` (script ghi đè thành 8192), `optim='adamw_torch'`, `vision_tower_lr` |

> 💡 Mặc định trong file `.py` **khác** giá trị trong `.sh`. Cái nào thắng? → **`.sh` thắng**, vì cờ
> dòng lệnh ghi đè mặc định. Khi đọc code luôn kiểm cả hai nơi.

---

## 3. Chặng ② — Nạp model & quyết định "học phần nào"

### 3.1. Nạp model — tên đường dẫn quyết định tất cả

[trainer:149-181](../../../code/internnav/trainer/internvla_n1_trainer.py#L149) — xem [02 mục 4.1](02_he_thong.md).
Với `train_system2.sh`, `llm=Qwen/Qwen2.5-VL-7B-Instruct` → khớp `"qwen2.5"` → nạp
`Qwen2_5_VLForConditionalGeneration` **gốc**, và đặt `data_args.model_type = "qwen2.5vl"`.

`model_type` này về sau quyết định dùng hàm tính vị trí token nào
([dataset:831-834](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L831)):
`get_rope_index_25` (cho Qwen2.5) hay `get_rope_index_2`.

Vì `--system1 "none"` và model không phải `internvla-n1`, dòng
[trainer:205-206](../../../code/internnav/trainer/internvla_n1_trainer.py#L205) **không chạy** →
**không có module System 1 nào được dựng**. Đây là điểm mấu chốt khiến nhánh S2 nhẹ hơn nhiều.

### 3.2. `set_model()` — cái công tắc "phần nào được học"

[trainer:78-122](../../../code/internnav/trainer/internvla_n1_trainer.py#L78):

```python
def set_model(model_args, model):
    if model_args.tune_mm_vision:                       # phần MẮT (vision encoder)
        for n, p in model.visual.named_parameters(): p.requires_grad = True
    else: ... = False
    if model_args.tune_mm_mlp:                          # LỚP NỐI mắt ↔ ngôn ngữ
        for n, p in model.visual.merger.named_parameters(): p.requires_grad = True
    else: ... = False
    if model_args.tune_mm_llm:                          # phần NÃO ngôn ngữ + lm_head
        for n, p in model.model.named_parameters(): p.requires_grad = True
        model.lm_head.requires_grad = True
    else: ... = False
    if 'nextdit' in model_args.system1:  # 🔵 chỉ chạy ở dual — mở học các module S1
        ...
```

| | `train_system2.sh` | `train_dual_system.sh` |
|---|---|---|
| `tune_mm_vision / mlp / llm` | `True / True / True` → **học toàn bộ S2** | `False / False / False` → **đóng băng S2** |
| khối `if 'nextdit'` | không chạy (`system1="none"`) | chạy → mở học `action_encoder`, `traj_dit`, `cond_projector`, `memory_encoder`, `rgb_resampler`, `latent_queries`… |

> Sau đó [dòng 220-224](../../../code/internnav/trainer/internvla_n1_trainer.py#L220) in ra bảng
> `tabulate` liệt kê **mọi tham số kèm cột `trainable`** — chạy thật thì nhìn bảng này là biết ngay
> mình có đóng băng đúng chỗ không. Rất đáng dùng để kiểm tra.

### 3.3. Biến đổi ảnh (augmentation)

[trainer:134-147](../../../code/internnav/trainer/internvla_n1_trainer.py#L134):

```python
if data_args.data_augmentation:
    data_args.transform_train = v2.Compose([
        v2.ToImage(), v2.ColorJitter(brightness=0.2, saturation=0.2),
        v2.RandomPosterize(bits=4), v2.RandomAdjustSharpness(sharpness_factor=1.5),
        v2.RandomAutocontrast(), v2.ToPILImage(),
        v2.Resize((data_args.resize_h, data_args.resize_w)),      # 384×384
    ])
else:
    data_args.transform_train = v2.Resize((data_args.resize_h, data_args.resize_w))
```

Nghĩa là: **kể cả khi tắt augmentation, ảnh vẫn bị resize về 384×384**. Bạn **không cần** resize
trước khi tạo data — cứ giữ 640×480 như data gốc.

---

## 4. Chặng ③ — Data: từ parquet thành "bài tập"

File: [internnav/dataset/internvla_n1_lerobot_dataset.py](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py)

### 4.1. `data_dict` — danh bạ dataset ([dòng 50-144](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L50))

Mỗi entry gồm đúng 4 trường:

```python
R2R_125CM_0_30 = {"data_path": "traj_data/r2r", "height": 125, "pitch_1": 0, "pitch_2": 30}
```

11 entry VLN có sẵn: `r2r/rxr/scalevln` × `{125cm_0_30, 125cm_0_45, 60cm_15_15, 60cm_30_30}`.
**Muốn dùng data của mình → phải thêm một entry vào đây**, không có cách nào khác
([dòng 164-169](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L164): tên không có
trong `data_dict` → `raise ValueError`).

- `parse_sampling_rate()` ([147](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L147)):
  hậu tố `%30` ở cuối tên dataset → chỉ lấy ngẫu nhiên 30% số mẫu.
- **Công thức `setting`** ([850](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L850)):

  ```python
  setting = f'{height}cm_{pitch_2}deg'      # ← DÙNG pitch_2 (góc CÚI), không phải pitch_1
  ```

  Đây là chuỗi quyết định **tên cột parquet** và **tên thư mục ảnh**. Sai chuỗi này = mất sạch data.

### 4.2. `get_annotations_from_lerobot_data()` ([752](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L752))

Quét mọi thư mục con của `data_path` (mỗi thư mục = **một scene**), 4 luồng song song. Với mỗi scene:

```python
episodes = read_jsonl(scene_path/"meta"/"episodes.jsonl")           # [765]
for ep in episodes:
    ep_instructions = ep["tasks"][0].split("<INSTRUCTION_SEP>")     # [770]  nhiều lệnh → nhiều bản sao
    parquet = scene/data/chunk-{ep_id//1000:03d}/episode_{ep_id:06d}.parquet   # [772]
    df = pq.read_table(parquet).to_pandas()
    ep_actions = df["action"].tolist()                              # [779]
    if pose_key in df.columns and goal_key in df.columns and rel_key in df.columns:   # [785]
        ep_poses      = df[pose_key].apply(lambda x: x.tolist()).tolist()             # [786]
        ep_pixel_goals= [[df[rel_key][i].tolist(), df[goal_key][i].tolist()] ...]     # [787-789]
    else:
        print("Warning: Missing data …, filling with defaults.")    # [791]  ← KHÔNG điền gì cả!
    assert len(ep_actions) == ep_len                                # [793]
```

🚨 **Hai cái bẫy chết người ở đây:**

1. **Nhánh `else` viết dở.** Nó chỉ in cảnh báo mà **không gán** `ep_poses` / `ep_pixel_goals` → dòng
   [802](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L802) gây `NameError`.
   Lỗi đó lại bị `try/except` ở [816](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L816)
   **nuốt mất** → scene bị **bỏ im lặng**. Nếu đặt sai tên cột, bạn sẽ thấy dataset rỗng mà không có
   traceback nào. → **Cả 3 cột `pose/goal/relative_goal_frame_id` phải cùng tồn tại.**
2. **`.tolist()` được gọi trên từng ô** → giá trị trong parquet phải là **numpy array**, không phải
   `int`/`list` Python thuần. Nghĩa là **dtype của parquet phải đúng** ([04](04_data_train_s2.md) mục 3).

Kết quả là một dict `{"episodes": [...]}`, mỗi episode có:
`id, video (đường dẫn thư mục chunk), instructions, actions, length, poses_{setting}, pixel_goals`.

### 4.3. `NavPixelGoalDataset.__init__()` — **cắt episode thành 3 loại mẫu** ([843-947](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L843))

Đây là **trái tim** của phần data. Đọc chậm đoạn này:

```python
actions = item['actions'][1:] + [0]        # [861] DỊCH TRÁI 1 nhịp, frame cuối thành STOP
pixel_goals = item['pixel_goals']
poses = item[f'poses_{height}cm_{pitch_2}deg']
if len(actions) < 4: continue              # [866] episode quá ngắn → bỏ

for n in range(len(actions)//sample_step + 1):        # [870] cứ 4 frame lấy 1 mẫu
    start = n * sample_step
    if start in (len, len-1): continue                # [871]
    pixel_goal = pixel_goals[start]                   # = [relative_goal_frame_id, [u,v]]

    if pixel_goal[0] == -1:                           # [876] KHÔNG có waypoint nhìn thấy
        if actions[start] == 1: continue              #  đang đi thẳng → bỏ, không tạo mẫu
        else:                                         #  → MẪU "TURN"
            gom actions[start..] cho tới khi gặp 1    # [882-885]
            turn_list.append((..., (start, start+1), turn_actions, None))     # [886]
    else:                                             # CÓ waypoint
        goal_len = pixel_goal[0]
        if goal_len < 3: continue                     # [902] waypoint quá gần → BỎ
        pixel_goal_list.append((..., (start, start+goal_len+1),
                                pixel_goal[1], poses[start:start+goal_len+1]))  # [906]

stop_list.append((..., (len-1, len), 0, None))        # [921] mỗi episode góp 1 mẫu STOP
```

Rồi gộp lại ([936-940](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L936)):

```python
list_data_dict = pixel_goal_list
if not self.pixel_goal_only:                  # train S2
    list_data_dict += turn_list
    list_data_dict += stop_list * 5           # nhân 5 để cân bằng lớp (model đừng quên học DỪNG)
```

**Ba loại mẫu — bảng tra:**

| Loại | Điều kiện | Model học xuất ra gì | Có trong train S2? | Có trong train dual? |
|---|---|---|---|---|
| **pixel_goal** | `rel_id ≥ 3` | `↓` rồi `"u v"` (toạ độ) | ✅ | ✅ (chỉ loại này) |
| **turn** | `rel_id == -1` và không đang đi thẳng | chuỗi `←`/`→` | ✅ | ❌ |
| **stop** | frame cuối episode | `STOP` | ✅ (×5) | ❌ |

Mỗi phần tử của `list_data_dict` là một tuple 10 phần tử — nhớ thứ tự này vì `__getitem__` giải nén
đúng thứ tự đó ([991-1002](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L991)):

```
(ep_id, data_path, video, height, pitch_1, pitch_2, instruction,
 (start_frame_id, end_frame_id), action, pose)
```

> 🔑 **`pose` ở đây đóng vai LÁ CỜ.** `pose is not None` ⇔ "đây là mẫu pixel_goal". Nó được dùng ở
> 3 chỗ: [1034] (có thêm ảnh lookdown), [1069] (chọn kiểu đáp án), [1110] 🔵 (tính nhãn quỹ đạo cho
> dual). **Chỉ chỗ 🔵 mới dùng GIÁ TRỊ SỐ của pose.**

### 4.4. `__getitem__()` — dựng một "bài tập" hoàn chỉnh ([990-1132](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L990))

**Bước 1 — chọn ảnh lịch sử** ([1003-1006]):

```python
history_id = np.unique(np.linspace(0, start_frame_id-1, num_history, dtype=int)).tolist()
```
→ lấy `num_history=8` mốc **rải đều** từ đầu episode tới ngay trước frame hiện tại (không phải 8
frame gần nhất!).

**Bước 2 — mở ảnh** ([1013-1026]):

```python
for id in range(0, end_frame_id):                                   # DUYỆT MỌI FRAME 0..end-1
    image_file = f"{video}/observation.images.rgb.{height}cm_{pitch_1}deg/episode_{ep_id:06d}_{id}.jpg"
    image          = Image.open(image_file)                                       # RGB nhìn thẳng
    lookdown_image = Image.open(image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg'))  # RGB cúi
    depth_image    = Image.open(image_file.replace(...).replace('rgb','depth').replace('.jpg','.png'))
    depth_image, _ = self.preprocess_depth_image_v2(depth_image, depth_scale=1000,
                                                    target_height=224, target_width=224)
```

🚨 **Ba hệ quả bắt buộc phải biết:**
- Ảnh RGB đuôi **`.jpg`**, depth đuôi **`.png`** — đặt sai đuôi là crash.
- **`Image.open(depth)` chạy ở MỌI frame, kể cả khi train S2.** Nên **file depth phải tồn tại** dù
  giá trị của nó không ảnh hưởng nhãn S2 ([04](04_data_train_s2.md) mục 6).
- Vòng lặp chạy từ **frame 0** tới `end_frame_id-1`, không phải chỉ các frame lịch sử → **mọi frame
  trong khoảng đó phải có đủ 3 file ảnh**.

`preprocess_depth_image_v2` ([975-988]): resize về 224×224, **chia 1000** (mm → mét), clip 5 m.

**Bước 3 — dựng đoạn hội thoại giả** ([1044-1081]). Đây chính là "đề bài + đáp án":

> **Human:** *"You are an autonomous navigation assistant. Your task is to `<instruction>`. Where
> should you go next to stay on track? Please output the next waypoint's coordinates in the image.
> Please output STOP when you have successfully completed the task. These are your historical
> observations: `<image>×8`. you can see `<image>`."*

| Loại mẫu | Đáp án model phải sinh ([1069-1081]) |
|---|---|
| pixel_goal (`pose is not None`) | 2 lượt: `↓` → rồi ảnh lookdown → `"340 210"` (toạ độ u v) |
| stop (`action == 0`) | `STOP` |
| turn | chuỗi ký hiệu, vd `"←←←"` |

**Bước 4 — tokenize + che nhãn** (`preprocess_qwen_2_visual`, [189-278]):
- Mỗi `<image>` bị thay bằng `<|vision_start|>` + N lần `<|image_pad|>` + `<|vision_end|>`
  ([228-241]), với N = số patch của ảnh đó.
- **Che nhãn:** phần `system` và `user` bị gán `IGNORE_INDEX = -100` ([261-262]) → **model không bị
  chấm điểm trên câu hỏi**, chỉ bị chấm trên câu trả lời ([264-266]).

**Bước 5 🔵 — nhãn quỹ đạo (chỉ khi `pixel_goal_only=True`)** ([1110-1131]):
lấy ảnh lookdown + depth của đoạn tới đích, và tính `traj_poses` bằng
`get_trajectory_relative_to_frame()` + nội suy spline. **Nhánh train S2 KHÔNG chạy khối này.**

### 4.5. Collator — ghép batch ([1149-1279](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L1149))

`DataCollatorForSupervisedDataset.__call__`:
- đệm `input_ids` bằng `pad_token_id`, đệm `labels` bằng `IGNORE_INDEX` ([1195-1198]);
- cắt về `model_max_length` ([1206-1208]);
- nối `pixel_values` của mọi mẫu thành một tensor dài + `image_grid_thw` ([1216-1219]).
- 🔵 nếu mẫu có `traj_images` thì thêm 4 token `<traj>` vào cuối chuỗi ([1155-1183]) và đóng gói
  `traj_images/traj_depths/traj_poses`.

### 4.6. `make_supervised_data_module()` ([1371](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L1371))

```python
if data_args.iign_dataset_use: train_datasets.append(VLLNDataset(...))     # biến thể VLLN
if data_args.vln_dataset_use:  train_datasets.append(NavPixelGoalDataset(...))
train_dataset = CombinedDataset(train_datasets, shuffle=False)
data_collator = DataCollatorForSupervisedDataset(tokenizer)
```

→ Nhánh S2 chỉ dùng `NavPixelGoalDataset`.

---

## 5. Các hàm hình học phụ (đọc khi cần)

| Hàm | Dòng | Việc nó làm |
|---|---|---|
| `get_trajectory_relative_to_frame(extrinsics, camera_deg)` | [592](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L592) | Dãy ma trận 4×4 camera→world → dãy `(x, y, yaw)` **tương đối với frame đầu**, đã **gỡ bỏ góc cúi** `camera_deg`. Đây là hàm định nghĩa **quy ước hệ toạ độ** mà `pose.{setting}` phải tuân theo. |
| `smooth_and_resample_trajectory(points, sample_length)` | [654](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L654) | Làm mượt quỹ đạo bằng **cubic spline** rồi lấy mẫu lại đều theo quãng đường. |
| `interpolate_and_resample_trajectory(...)` | [571](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L571) | Lọc bước quá ngắn → làm mượt → đổi sang `(dx, dy, dyaw)`. |
| `clip_or_pad(arr, fixed_len)` | [743](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L743) | Cắt/đệm cho đủ `predict_step_num` bước. |

### 🔍 Kiểm chứng quy ước hệ toạ độ (đã chạy thật)

Chạy `get_trajectory_relative_to_frame()` trên `pose.60cm_30deg` của
`vln_ce/traj_data/r2r/17DRP5sb8fy/episode_000000` (action = `[-1,3,3,3,1,3,1,1,2,...]`):

```
[[ 0.     0.     0.   ]      frame 0 = gốc toạ độ
 [ 0.     0.    -0.262]      action 3 (phải) → yaw −15°
 [ 0.     0.    -0.524]      −30°
 [ 0.     0.    -0.785]      −45°
 [ 0.177 -0.177 -0.785]      action 1 (tiến) → đi 0.25 m theo hướng hiện tại
 ...]
```
→ Xác nhận: cột 4 của `pose` là **vị trí camera trong world** (`z = 0.6` = chiều cao 60 cm), ma trận
là **camera → world** với camera theo **quy ước OpenCV**, và bước đi rời rạc là **0.25 m / 15°**.
Chi tiết công thức dựng lại ma trận này: [04](04_data_train_s2.md) mục 5.

---

## 6. Chặng ⑤⑥⑦ — Trainer, resume, lưu

[trainer:217-235](../../../code/internnav/trainer/internvla_n1_trainer.py#L217):

```python
trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
if list(pathlib.Path(output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)     # tự học tiếp nếu có checkpoint dở dang
else:
    trainer.train()
trainer.save_state()
data_args.image_processor.save_pretrained(output_dir)
safe_save_model_for_hf_trainer(trainer, output_dir)
```

> ⚠️ **Bẫy:** nếu `output_dir` **trùng** thư mục checkpoint bạn nạp vào, code sẽ tưởng đó là lần chạy
> dở dang và resume nhầm. **Nơi nạp ≠ nơi lưu.**

---

## 7. Fine-tune hay train từ đầu? — **Luôn là fine-tune**

Cả 3 script đều nạp trọng số có sẵn qua `from_pretrained(...)`. Không có kịch bản khởi tạo ngẫu nhiên.

- `train_system2.sh` = fine-tune **Qwen2.5-VL-7B** (Alibaba đã pretrain trên lượng dữ liệu khổng lồ)
  sang nhiệm vụ điều hướng.
- `train_dual_system.sh` = fine-tune **giai đoạn 2**: nạp lại thành quả giai đoạn 1, đóng băng nó,
  chỉ dạy thêm System 1.

### 7.1. Fine-tune tiếp từ checkpoint `InternVLA-N1-System2`

Đây là tình huống thường gặp nhất (dạy tiếp một model đã biết điều hướng, thay vì bắt đầu từ Qwen gốc).

```bash
huggingface-cli download InternRobotics/InternVLA-N1-System2 \
    --local-dir checkpoints/InternVLA-N1-System2      # GIỮ NGUYÊN TÊN THƯ MỤC
```

Sửa `train_system2.sh` đúng 3 chỗ:

| Dòng | Gốc | Sửa thành | Lý do |
|---|---|---|---|
| [19](../../../code/scripts/train/qwenvl_train/train_system2.sh#L19) | `llm=Qwen/Qwen2.5-VL-7B-Instruct` | `llm=checkpoints/InternVLA-N1-System2` | Nạp từ checkpoint InternNav |
| [33](../../../code/scripts/train/qwenvl_train/train_system2.sh#L33) | `run_name=InternVLA-N1-System2` | `run_name=InternVLA-N1-System2-ft` | **Nơi lưu phải khác nơi nạp** |
| [22](../../../code/scripts/train/qwenvl_train/train_system2.sh#L22) | `lr=2e-5` | `lr=5e-6` *(gợi ý)* | Fine-tune tiếp nên dùng lr nhỏ hơn |

Giữ nguyên `--system1 "none"` và `--tune_mm_vision/mlp/llm True`.

**Vì sao chạy được:** `InternVLAN1ForCausalLM` kế thừa `Qwen2_5_VLForConditionalGeneration`; checkpoint
S2 về bản chất là Qwen2.5-VL đã tinh chỉnh. Với `--system1 "none"`, `initialize_vision_modules` không
dựng module S1 → trọng số khớp sạch.

### 7.2. Fine-tune trên **dữ liệu của bạn** — đúng 3 bước

1. Chuyển data sang **định dạng LeRobot của `vln_ce`** ([04](04_data_train_s2.md), hoặc dùng
   pipeline mcap ở [06](06_pipeline_mcap_to_s2.md)).
2. **Đăng ký** vào `data_dict` ([dòng 127](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L127)):
   ```python
   MYROBOT_125CM_0_30 = {"data_path": "traj_data/myrobot", "height": 125, "pitch_1": 0, "pitch_2": 30}
   data_dict = {..., "myrobot_125cm_0_30": MYROBOT_125CM_0_30}
   ```
3. Trỏ `--vln_dataset_use myrobot_125cm_0_30` trong `.sh`.

---

## 8. Bảng tra nhanh: "muốn đổi X thì sửa ở đâu"

| Muốn đổi | Sửa chỗ nào |
|---|---|
| Model nền / checkpoint khởi đầu | `llm=` trong `.sh` (nhớ luật đặt tên ở mục 3.1) |
| Dataset dùng để train | `vln_datasets=` trong `.sh` + entry trong `data_dict` |
| Chỉ lấy một phần data | thêm `%30` sau tên dataset |
| Đóng băng / mở học từng phần | `--tune_mm_vision/mlp/llm` |
| Mật độ lấy mẫu | `--sample_step` (4 = cứ 4 frame lấy 1) |
| Số ảnh lịch sử | `--num_history` |
| Kích thước ảnh vào model | `--resize_h/--resize_w` |
| Bật/tắt augmentation | `--data_augmentation` |
| Chỉ train mẫu pixel_goal | `--pixel_goal_only True` (dùng cho dual) |
| Tiết kiệm VRAM | đổi `zero2.json` → `zero3.json` / `zero3_offload.json`, giảm `batch_size`, bật `gradient_checkpointing` |
| Cấu hình camera của data mới | `height`/`pitch_1`/`pitch_2` trong entry `data_dict` |

---

Tiếp theo: cấu trúc chi tiết của data S2 và **cái gì bắt buộc / không bắt buộc** →
[04_data_train_s2](04_data_train_s2.md).
Bản song song cho nhánh System 1 → [03b_code_train_s1](03b_code_train_s1.md).
