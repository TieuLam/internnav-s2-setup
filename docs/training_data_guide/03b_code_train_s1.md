# 03b — Giải thích code từng phần: **nhánh huấn luyện System 1** (NavDP)

> **File này để làm gì:** đi từ dòng lệnh `bash start_train.sh --model navdp` cho tới lúc trọng số
> được cập nhật, **giải thích từng mảnh code làm gì**. Đây là bản song song của
> [03_code_train_s2](03_code_train_s2.md) nhưng cho **bộ não phản xạ**.
>
> Mọi trích dẫn có `file:line` — mở ra tự kiểm chứng được. Các con số "đo thật" lấy từ scene
> `vln_n1/traij_data/3dfront_d435i/00154c06-2ee2-408a-9664-b8fd74742897` (episode 0, 78 frame).
> Bộ tài liệu: [02_he_thong](02_he_thong.md) · [05_data_train_s1](05_data_train_s1.md) ·
> [06b_pipeline_mcap_to_s1](06b_pipeline_mcap_to_s1.md)

---

## 0. Đọc trước: nhánh S1 **không giống gì** nhánh S2

Nếu bạn vừa đọc [03](03_code_train_s2.md), hãy xoá bớt kỳ vọng — hai nhánh dùng **hai bộ khung
hoàn toàn khác nhau**, chỉ trùng nhau ở chỗ cùng gọi `transformers.Trainer`.

| | **Nhánh S2** (`qwenvl_train`) | **Nhánh S1** (`base_train`) |
|---|---|---|
| Script | `train_system2.sh` | `start_train.sh --model navdp` |
| Cách nhận tham số | **cờ dòng lệnh** → `HfArgumentParser` | **file config `.py`** → pydantic `ExpCfg` |
| Model | Qwen2.5-VL 7B (VLM khổng lồ) | NavDPNet (~vài chục triệu tham số) |
| Nạp trọng số | **luôn** `from_pretrained` một checkpoint có sẵn | `ckpt_to_load=''` → **train từ đầu** (mục 4.2) |
| Song song | DeepSpeed ZeRO-2, 64 GPU | DDP thuần, 8 GPU |
| Loss | cross-entropy sinh token | **4 thành phần** (diffusion + critic + aux), mục 8 |
| Ngôn ngữ | ✅ trung tâm | ❌ **không dùng một chữ nào** |
| Trainer | `transformers.Trainer` gần như nguyên bản | `NavDPTrainer` **ghi đè 6 phương thức** |

👉 Hệ quả thực dụng: **các mẹo của nhánh S2 không áp dụng được ở đây.** Muốn đổi tham số S1 thì sửa
[scripts/train/base_train/configs/navdp.py](../../../code/scripts/train/base_train/configs/navdp.py),
**không phải** sửa `.sh`.

---

## 1. Toàn cảnh: 7 chặng của một lần train S1

```
 ①  scripts/train/base_train/start_train.sh --model navdp
     └─ export CUDA_VISIBLE_DEVICES=0..7 ; torchrun --nproc_per_node=8   [47-50, 69-80]
 ②  scripts/train/base_train/train.py :: __main__                        [284-325]
     ├─ tyro.cli(TrainCfg)  → chỉ đọc đúng 2 cờ: --name, --model-name    [286]
     ├─ supported_cfg['navdp'] → (navdp_exp_cfg, "NavDP_Policy")         [303]
     └─ get_policy/get_config → NavDPNet / NavDPModelConfig              [310]
 ③  main() :: khởi tạo phân tán + nạp model                              [75-148]
     ├─ dist.init_process_group('nccl')                                  [104-110]
     ├─ NavDPNet.from_pretrained(ckpt_to_load, config=…)                 [127]
     ├─ đóng băng mọi tham số tên chứa 'mask_token'                      [130-132]
     └─ bọc DistributedDataParallel                                      [145-148]
 ④  NavDP_Base_Datset(...)  ← quét thư mục scene, KHÔNG đọc ảnh          [170-182]
     └─ navdp_lerobot_dataset.py:74-137
 ⑤  TrainingArguments + NavDPTrainer(...)                                [225-259]
 ⑥  trainer.train()                                                      [266]
     ├─ get_train_dataloader()  → DistributedSampler + collate           [navdp_trainer:171]
     ├─ __getitem__ dựng 1 mẫu (đây mới là chỗ đọc ảnh)  [dataset:416-570]
     ├─ compute_loss()  → 4 thành phần loss                [navdp_trainer:26-122]
     └─ save_model() mỗi epoch                             [navdp_trainer:188-205]
 ⑦  checkpoints/<name>/ckpts/checkpoint-<step>navdp.ckpt   ← ⚠️ tên file dính liền, mục 9.3
```

---

## 2. Chặng ① — `start_train.sh`: chỉ là cái nút bấm

File: [scripts/train/base_train/start_train.sh](../../../code/scripts/train/base_train/start_train.sh)

```bash
bash scripts/train/base_train/start_train.sh --name my_navdp --model navdp
```

Script chỉ làm **ba việc**, không chứa siêu tham số nào:

| Việc | Dòng | Chi tiết |
|---|---|---|
| Chọn GPU | [47-50](../../../code/scripts/train/base_train/start_train.sh#L47) | `navdp` → `CUDA_VISIBLE_DEVICES=0..7`, `NUM_GPUS=8` (**hard-code**) |
| Bật cờ debug NCCL | [64-66](../../../code/scripts/train/base_train/start_train.sh#L64) | `NCCL_DEBUG=INFO` — log rất dài, tắt được |
| Gọi launcher | [69-80](../../../code/scripts/train/base_train/start_train.sh#L69) | **chỉ `navdp` dùng `torchrun`**; các model khác chạy `python` thuần |

> ⚠️ **Muốn chạy 1 GPU** thì phải sửa thẳng vào `case "navdp")`: đổi thành
> `CUDA_VISIBLE_DEVICES=0` và `NUM_GPUS=1`. Không có cờ dòng lệnh nào làm được việc đó.
> Nhớ sửa **cả** `torch_gpu_ids` trong config (mục 3.2), nếu không assert ở
> [train.py:319](../../../code/scripts/train/base_train/train.py#L319) sẽ không khớp thực tế.

> 💡 `--master_port` bị khai báo **hai lần** ([73](../../../code/scripts/train/base_train/start_train.sh#L73)
> và [77](../../../code/scripts/train/base_train/start_train.sh#L77)) — cái sau (`12345`) thắng. Vô hại,
> nhưng biết để khỏi tưởng mình đọc nhầm.

---

## 3. Chặng ② — Cấu hình đến từ đâu

### 3.1. `train.py` chỉ nhận đúng 2 cờ

[train.py:33-37](../../../code/scripts/train/base_train/train.py#L33):

```python
class TrainCfg(BaseModel):
    name: str = 'cma_train'        # tên thí nghiệm → tên thư mục checkpoint
    model_name: str = 'cma'        # chọn model
```

`tyro.cli(TrainCfg)` ([286](../../../code/scripts/train/base_train/train.py#L286)) chỉ dựng đúng 2
tham số này. **Mọi siêu tham số khác nằm trong file config.** Bảng tra ở
[297-304](../../../code/scripts/train/base_train/train.py#L297):

```python
supported_cfg = {..., 'navdp': [navdp_exp_cfg, "NavDP_Policy"]}
model_class, model_config_class = get_policy("NavDP_Policy"), get_config("NavDP_Policy")
```

→ `NavDPNet` + `NavDPModelConfig`
([model/__init__.py](../../../code/internnav/model/__init__.py)).

### 3.2. `configs/navdp.py` — nơi chứa **toàn bộ** siêu tham số

[scripts/train/base_train/configs/navdp.py](../../../code/scripts/train/base_train/configs/navdp.py):

```python
navdp_exp_cfg = ExpCfg(
    name='navdp_train', model_name='navdp',
    torch_gpu_ids=[0],                              # ⚠️ mục 3.3
    output_dir='checkpoints/%s/ckpts',              # %s ← thay bằng --name
    il=IlCfg(
        epochs=1000, batch_size=32, lr=1e-4, num_workers=8,
        ckpt_to_load='',                            # ⚠️ RỖNG = train từ đầu (mục 4.2)
        root_dir='data/datasets/InternData-N1/vln_n1/traj_data',   # nơi đọc data
        dataset_navdp='data/datasets/navdp_dataset_lerobot.json',  # ⚠️ mục 5.2
        image_size=224, memory_size=8, predict_size=24, pixel_channel=4,
        temporal_depth=16, heads=8, token_dim=384, channels=3, dropout=0.1,
        scene_scale=1.0, preload=False, random_digit=False, prior_sample=False,
        scratch=False, finetune=False,              # ⚠️ mục 4.4
    ),
    model=navdp_cfg,
)
```

Nhóm theo vai trò:

| Nhóm | Tham số | Ảnh hưởng |
|---|---|---|
| **Hình dạng dữ liệu** | `image_size=224`, `memory_size=8`, `predict_size=24`, `pixel_channel=4` | Phải **khớp** với data và với model. Đổi là phải train lại từ đầu. |
| **Kiến trúc** | `token_dim=384`, `heads=8`, `temporal_depth=16`, `dropout=0.1` | Kích thước Transformer decoder |
| **Lấy mẫu data** | `scene_scale`, `preload`, `random_digit`, `prior_sample` | Mục 5 |
| **Vòng lặp học** | `epochs=1000`, `batch_size=32`, `lr=1e-4`, `num_workers=8` | `batch_size` là **mỗi GPU** |
| **Đường dẫn** | `root_dir`, `dataset_navdp`, `ckpt_to_load`, `output_dir` | Mục 5.2, 9 |

> 📌 `weight_decay=1e-4` và `warmup_ratio=0.05` có trong config nhưng **bị bỏ qua** — `NavDPTrainer`
> tự tạo optimizer bằng `torch.optim.Adam(params, lr=lr)`, không truyền `weight_decay`
> ([navdp_trainer:145](../../../code/internnav/trainer/navdp_trainer.py#L145)), và tự tạo scheduler
> riêng ([154-157](../../../code/internnav/trainer/navdp_trainer.py#L154)). Đừng mất công chỉnh hai
> tham số đó.

### 3.3. 🚨 `torch_gpu_ids=[0]` mâu thuẫn với `NUM_GPUS=8`

[train.py:312-322](../../../code/scripts/train/base_train/train.py#L312):

```python
exp_cfg.num_gpus  = len(exp_cfg.torch_gpu_ids)     # = 1 vì config ghi [0]
exp_cfg.world_size = exp_cfg.num_gpus
assert exp_cfg.num_gpus <= available_gpus
```

Trong khi `start_train.sh` khởi 8 tiến trình. Kết quả: `config.world_size` **sai** (=1), nhưng số
GPU thật vẫn là 8 vì độ song song do `torchrun` + biến môi trường `WORLD_SIZE` quyết định
([train.py:94](../../../code/scripts/train/base_train/train.py#L94)), **không** do config. Trường
`config.world_size` chỉ là số trang trí — nhưng đừng dựa vào nó để tính batch toàn cục.

**Batch toàn cục thật = `batch_size` × số tiến trình = 32 × 8 = 256.**

---

## 4. Chặng ③ — Nạp model

### 4.1. `NavDPModelConfig` bọc nguyên cục config

[navdp_policy.py:19-31](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L19): toàn bộ
`ExpCfg` được nhét vào một trường `model_cfg` của `PretrainedConfig`. Vì thế trong `__init__` model
mới đọc kiểu `self.config.model_cfg['il']['image_size']`
([75-85](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L75)) — hơi lạ mắt nhưng đúng.

### 4.2. `from_pretrained('')` = **train từ đầu**

[navdp_policy.py:37-64](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L37):

```python
if os.path.isdir(path):                       # thư mục → nạp pytorch_model.bin
    model.load_state_dict(torch.load(path/'pytorch_model.bin'))
elif path is None or len(path) == 0:          # ← ckpt_to_load='' rơi vào đây
    pass                                      #   KHÔNG nạp gì cả
else:                                         # file .ckpt/.pth → nạp strict=False
    model.load_state_dict(torch.load(path), strict=False)
```

> 🔑 **Đây là khác biệt lớn nhất so với S2.** [00_README](00_README.md) nói "không bao giờ train từ
> đầu" — câu đó đúng cho **hai script `qwenvl_train`**. Nhánh NavDP thì mặc định `ckpt_to_load=''`
> nên **mọi module đều khởi tạo ngẫu nhiên**, trừ đúng một thứ: xương sống thị giác.

### 4.3. Thứ **duy nhất** luôn có trọng số sẵn: DepthAnythingV2

[navdp_backbone.py:229-232](../../../code/internnav/model/encoder/navdp_backbone.py#L229):

```python
self.rgb_model = DepthAnythingV2(**model_configs['vits'])
self.rgb_model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vits.pth"), strict=False)
self.rgb_model = self.rgb_model.pretrained.float()      # chỉ giữ phần ViT encoder
```

**Bốn** encoder đều dựng từ `DepthAnythingV2` bản `vits`:

| Encoder | Nơi định nghĩa | Nạp `.pth`? | Đầu vào |
|---|---|---|---|
| `rgbd_encoder.rgb_model` | [backbone:229](../../../code/internnav/model/encoder/navdp_backbone.py#L229) | ✅ | 8 ảnh RGB bộ nhớ |
| `rgbd_encoder.depth_model` | [backbone:239](../../../code/internnav/model/encoder/navdp_backbone.py#L239) | ❌ ngẫu nhiên | 1 ảnh depth (nhân 3 kênh) |
| `image_encoder` (ImageGoalBackbone) | [backbone:329](../../../code/internnav/model/encoder/navdp_backbone.py#L329) | ❌ ngẫu nhiên | **6 kênh** (ảnh đích ‖ ảnh hiện tại) |
| `pixel_encoder` (PixelGoalBackbone) | [backbone:392](../../../code/internnav/model/encoder/navdp_backbone.py#L392) | ❌ ngẫu nhiên | **`pixel_channel` kênh** (=4) |

> 🚨 **Thiếu file `checkpoints/depth_anything_v2_vits.pth` là crash ngay lúc dựng model**, không có
> nhánh dự phòng. Lưu ý tên khác với file mà nhánh dual dùng
> (`depth_anything_v2_metric_hypersim_vits.pth`, [02](02_he_thong.md) mục 4.2) — **hai file khác
> nhau**, đừng đổi tên cho nhau.
>
> Ba encoder còn lại đổi số kênh đầu vào của `patch_embed.proj` nên **không thể** nạp trọng số gốc
> ([backbone:331-337](../../../code/internnav/model/encoder/navdp_backbone.py#L331),
> [394-400](../../../code/internnav/model/encoder/navdp_backbone.py#L394)).

### 4.4. `finetune=False` → **đóng băng mắt RGB**

[navdp_policy.py:95-98](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L95):

```python
if not self.finetune:
    for p in self.rgbd_encoder.rgb_model.parameters(): p.requires_grad = False
    self.rgbd_encoder.rgb_model.eval()
```

Và trong `forward` của backbone, token ảnh bị `detach()`
([backbone:267-268](../../../code/internnav/model/encoder/navdp_backbone.py#L267)) → gradient **không**
chảy ngược vào encoder RGB. Đây là lựa chọn hợp lý: giữ nguyên đặc trưng thị giác đã pretrain, chỉ
học phần điều hướng.

Thêm một chỗ đóng băng nữa ở [train.py:130-132](../../../code/scripts/train/base_train/train.py#L130):
mọi tham số có tên chứa `mask_token` bị `requires_grad=False` (token này thuộc ViT của
DepthAnythingV2, không dùng tới → nếu để mở sẽ báo lỗi "tham số không nhận gradient" trong DDP).

---

## 5. Chặng ④ — Dataset: `NavDP_Base_Datset`

File: [internnav/dataset/navdp_lerobot_dataset.py](../../../code/internnav/dataset/navdp_lerobot_dataset.py).
Cấu trúc dữ liệu mà nó đọc: [05_data_train_s1](05_data_train_s1.md).

### 5.1. `train.py` truyền tham số **theo vị trí** — dễ nhầm

[train.py:170-182](../../../code/scripts/train/base_train/train.py#L170) so với chữ ký
[dataset:34-51](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L34):

| Vị trí | Tham số nhận | Giá trị từ config |
|---|---|---|
| 1 | `root_dirs` | `root_dir` |
| 2 | **`preload_path`** | `dataset_navdp` ← **không phải** "dataset" mà là đường dẫn file cache |
| 3–6 | `memory_size, predict_size, batch_size, image_size` | 8, 24, 32, 224 |
| 7 | `scene_data_scale` | `scene_scale` = 1.0 |
| kwargs | `pixel_channel, preload, random_digit, prior_sample` | 4, False, False, False |

> ⚠️ `trajectory_data_scale`, `action_dim`, `debug` **không được truyền** → giữ mặc định
> (`1.0`, `3`, `False`). Riêng `trajectory_data_scale` và `debug` **không được dùng ở đâu cả** —
> code chết.

### 5.2. `__init__` — quét thư mục, **không** đọc ảnh

[dataset:74-131](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L74):

```python
for group_dir in os.listdir(root_dirs):                 # 3dfront_d435i, gibson_zed, …
    for scene_dir in os.listdir(root/group):            # từng scene uuid
        chunk_name  = os.listdir(root/group/scene/'data')[0]      # [82] ⚠️ CHỈ LẤY CHUNK ĐẦU
        episode_info = jsonlines(scene/'meta/episodes_stats.jsonl')   # [85-88]
        rgb_paths   = sorted(os.listdir(scene/f'videos/{chunk}/observation.images.rgb/'))   # [92]
        depth_paths = sorted(os.listdir(scene/f'videos/{chunk}/observation.images.depth/')) # [97]
        data_paths  = sorted(os.listdir(scene/f'data/{chunk}'))                             # [99]
        for episode_idx, episode in enumerate(episode_info):        # [101]
            lo, hi = episode['image_index']['min'], episode['image_index']['max']
            self.trajectory_rgb_path.append(rgb_paths[lo:hi+1])     # [104]
            self.trajectory_data_dir.append(data_paths[episode_idx])# [108] ← GHÉP THEO THỨ TỰ
```

**Ba ràng buộc sinh ra từ đúng 6 dòng này** — nắm chắc trước khi tự tạo data
([06b](06b_pipeline_mcap_to_s1.md) mục 5):

1. **`image_index` là chỉ số TOÀN CỤC** trong danh sách ảnh đã `sorted()` của **cả scene**, không
   phải chỉ số trong episode. Đo thật trên scene mẫu:

   ```
   ep0: image_index 0…77      (78 ảnh)
   ep1: image_index 78…182    (105 ảnh)   ← nối tiếp, không reset
   ep2: image_index 183…305
   ```

2. **Ảnh của mọi episode nằm CHUNG một thư mục phẳng**, và `sorted()` phải cho đúng thứ tự
   episode-rồi-frame → tên file **bắt buộc đệm 0**: `episode_000000_007.jpg`
   ([05](05_data_train_s1.md) mục 2.1).

3. **Parquet ghép với episode theo thứ tự liệt kê**, dùng `episode_idx` (số thứ tự dòng trong
   `episodes_stats.jsonl`), **không** dùng trường `episode_index` bên trong. Thứ tự dòng trong
   `episodes_stats.jsonl` phải khớp thứ tự `sorted()` của thư mục parquet.

Thêm hai cái bẫy:

- **`chunk_name = os.listdir(...)[0]`** — chỉ lấy **một** chunk, và `os.listdir` **không đảm bảo thứ
  tự**. Scene có `chunk-000` và `chunk-001` (>1000 episode) sẽ mất dữ liệu **im lặng**.
  → Giữ mỗi scene trong **đúng một chunk**.
- **`preload_path` bị ghi đè mỗi lần chạy với `preload=False`**
  ([118-125](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L118)): code `json.dump` kết
  quả quét ra file `data/datasets/navdp_dataset_lerobot.json`. **Thư mục cha phải tồn tại**, nếu
  không sẽ `FileNotFoundError` sau khi đã quét xong (rất tốn thời gian mới báo lỗi). Lần sau đặt
  `preload=True` để nạp lại cache trong ~1 giây.

### 5.3. Nhân bản ×50 — vì sao không phải là "học lại 50 lần"

[dataset:128-137](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L128):

```python
self.trajectory_data_dir = self.trajectory_data_dir * 50
```

`__len__` = số episode × 50. Nghe như overfit, nhưng **không**: mỗi lần `__getitem__` được gọi, nó
**bốc ngẫu nhiên một cửa sổ khác** của cùng episode (mục 6.1) và **xoay ngẫu nhiên** quỹ đạo phụ
(mục 6.4). Cùng một `index` gọi 2 lần cho ra 2 mẫu khác nhau.

> 📌 Hệ quả cho việc tính lịch train: "1 epoch" ở đây = **50 lượt** quét qua tập episode. Với
> `epochs=1000` trong config, con số thật là 50 000 lượt. Đây là lý do `save_strategy='epoch'` mà
> vẫn ra checkpoint đều đặn.

### 5.4. `__getitem__` — 6 bước dựng một mẫu

[dataset:416-570](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L416):

```
① đọc parquet             → K (3×3), E_base (4×4), quỹ đạo A[i] (4×4 × n)   [424-429]
② đọc pointcloud.ply      → lọc điểm màu (0,0,128) = vật cản                [431]
③ bốc 3 mốc ngẫu nhiên    pixel_start ≤ memory_start < target               [433-439]
④ dựng quan sát           8 ảnh bộ nhớ + 1 depth + 3 kiểu "đích"            [448-516]
⑤ dựng nhãn               pred_actions, augment_actions (24×3)              [454-525]
⑥ chấm điểm an toàn       pred_critic, augment_critic (2 số vô hướng)       [471-494]
```

**Bước ③ — ba mốc thời gian** ([433-439]):

```python
pixel_start_choice  = np.random.randint(0, L//2)                  # nửa đầu episode
target_choice       = np.random.randint(pixel_start + 1, L - 1)   # đích, luôn ở sau
memory_start_choice = np.random.randint(pixel_start, target)      # "hiện tại"
```

| Mốc | Vai trò |
|---|---|
| `memory_start_choice` | **"bây giờ"** — gốc toạ độ của nhãn, nơi lấy depth và 8 ảnh bộ nhớ |
| `target_choice` | **đích** — dùng cho `point_goal` và ảnh `image_goal` |
| `pixel_start_choice` | khung ảnh **quá khứ** để chiếu `pixel_goal` lên |

> 🚨 Episode phải có **ít nhất 4 frame**: `randint(0, L//2)` cần `L ≥ 2`, và `randint(p+1, L-1)` cần
> `L-1 > p+1`. Episode 2–3 frame gây `ValueError` **giữa lúc train**, không phải lúc khởi động.

**Bước ④ — 8 ảnh bộ nhớ, giãn cách 4 frame** ([215-222](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L215)):

```python
memory_index = np.arange(start - (memory_size-1)*digit, start+1, digit)   # digit = 4
# = [start-28, start-24, …, start-4, start]  → 8 mốc
context_image[outrange_sum:] = …    # chỉ số âm bị bỏ, phần đầu giữ ẢNH ĐEN (zeros)
context_depth = process_depth(depth_paths[start])                          # CHỈ 1 ảnh depth
```

→ Khác S2 (rải đều từ đầu episode): S1 nhìn **28 frame gần nhất**. Với data gốc 30 fps ≈ **1 giây**.
Frame đầu episode được đệm bằng ảnh đen.

**Ba kiểu "đích" cùng lúc** — model học cả ba để lúc chạy thật dùng kiểu nào cũng được:

| Đích | Dựng thế nào | Hình dạng |
|---|---|---|
| `point_goal` | `target_xyt_actions[-1]` — toạ độ `(x, y, θ)` của đích trong hệ robot | `(3,)` |
| `image_goal` | ảnh tại `target` **nối kênh** với ảnh tại `memory_start` | `(224,224,6)` |
| `pixel_goal` | ảnh tại `pixel_start` + **mặt nạ trắng** đánh dấu đích | `(224,224,4)` |

`process_pixel_goal` ([224-266](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L224)) vẽ
một **hình chữ nhật trắng kích thước ngẫu nhiên 6–12 px** quanh điểm chiếu, rồi resize xuống 224.
Nếu điểm rơi ngoài khung → mặt nạ toàn 0 và `visible_flag=False`.

> ⚠️ `pixel_flag` được trả về ở [569] nhưng **`navdp_collate_fn` vứt đi** ([573-586]) — nó chỉ gom 9
> phần tử đầu. Nghĩa là **mẫu có mặt nạ rỗng vẫn được train bình thường**, model phải tự học "không
> có dấu thì bỏ qua kênh này".

> ⚠️ Hàm chiếu **không kiểm tra điểm có ở trước mặt camera không** (không có `if z <= 0`). Đích nằm
> sau lưng (sau một cú quay đầu) vẫn có thể cho ra toạ độ pixel nằm trong khung → **nhãn sai âm
> thầm**. Khác với pipeline S2 vốn kiểm tra rõ ràng ([06](06_pipeline_mcap_to_s2.md) mục C.3).

**Chi tiết `pixel_channel`** ([521-522]):

```python
if self.pixel_channel == 7:
    pixel_goal = np.concatenate((pixel_goal, memory_images[-1]), axis=-1)
```

`4` = (ảnh quá khứ 3 kênh + mặt nạ 1 kênh) — đích được đánh dấu **trên khung hiện tại**.
`7` = thêm ảnh hiện tại — dùng khi đích được đánh dấu ở khung **quá khứ** (điều hướng bất đồng bộ).
Config đang để **4**, và số này phải khớp `PixelGoalBackbone(pixel_channel=…)`
([navdp_policy:89-91](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L89)).

---

## 6. Hình học — phần đáng đọc kỹ nhất

Đây là chỗ quyết định data tự tạo có dùng được hay không. Mọi khẳng định dưới đây **đo trực tiếp**
trên `episode_000000.parquet` của scene mẫu.

### 6.1. Quy ước trục của `vln_n1` (đo thật, không phải suy đoán)

Mỗi `action[i]` là ma trận 4×4 **camera → world**:

```
action[i] = [ Rz(yaw_i) · R_mount | C_i ]
            [        0  0  0      |  1  ]
```

trong đó `R_mount = camera_extrinsic[:3,:3]`. Đo được trên scene mẫu:

```
R_mount = [[1, 0,  0],        C_i = (x, y, 0.35698)     ← z KHÔNG ĐỔI suốt episode
           [0, 0, -1],                                    = chiều cao lắp camera
           [0, 1,  0]]
```

Kiểm chứng bằng số: `action[i][:3,:3] @ inv(R_mount)` cho ra **ma trận xoay quanh trục z thuần tuý**
ở mọi frame (đo: yaw = +10.56° tại frame 0, −4.92° tại frame 20, −62.29° tại frame 77), và cột 2
của `action` **luôn đúng bằng** `(0,0,1)` ở cả 78 frame → **camera không có góc cúi (pitch = 0)**
trong bộ data này.

Ba cột của phần xoay, giải nghĩa:

| Cột | Là trục nào của camera | Kiểm chứng |
|---|---|---|
| 0 | **phải** | vuông góc hướng đi (tích vô hướng với vector di chuyển = **0.007**) |
| 1 | **lên** | luôn = `(0,0,1)` |
| 2 | **LÙI** (camera nhìn theo **−z**) | tích vô hướng với vector di chuyển = **−1.000** |

> 🔧 **Đính chính [05](05_data_train_s1.md) mục 3.1:** tài liệu đó ghi "cột 3 = trước". Đo lại cho
> thấy cột 3 chỉ **ra sau**; hướng đi là **−cột 3**. Tức quy ước là **x-phải, y-lên, z-lùi** (kiểu
> OpenGL/Blender), thuận tay phải (det = +1). Vẫn khác `vln_ce` (OpenCV: x-phải, y-**xuống**,
> z-**trước**) — kết luận "đừng bê pose từ hệ này sang hệ kia" **vẫn đúng**, chỉ là lệch **180°
> quanh trục x** chứ không phải chỉ lật trục y.

Số liệu quỹ đạo đo thật (episode 0, 78 frame):

| Chỉ số | Giá trị |
|---|---|
| Bước đi mỗi frame | trung vị **0.0386 m**, trung bình 0.0369 m (min 0.0285 – max 0.0411) |
| `fps` (từ `info.json`) | 30 |
| → tốc độ | ≈ **1.1 m/s** |
| Tổng quãng đường | 2.84 m |
| Độ cao camera | 0.357 m, **hằng số** |

→ **Quỹ đạo `vln_n1` là liên tục và dày** (0.04 m/frame), hoàn toàn khác `vln_ce` (rời rạc
0.25 m/bước, 15°/nấc). Đây là điều kiện tiên quyết khi tự tạo data ([06b](06b_pipeline_mcap_to_s1.md)
mục 4.4).

### 6.2. `relative_pose()` — đổi sang hệ robot

[dataset:268-288](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L268):

```python
R_base = R_start @ inv(base_extrinsic[:3,:3])     # ① GỠ BỎ phép xoay do LẮP camera
homo_RT = [[R_base, T_start], [0,1]]
T_frame = inv(homo_RT) @ [P_world, 1]             # ② đưa điểm về hệ "thân robot"
T_frame = [T_frame[1], -T_frame[0], T_frame[2]]   # ③ ĐỔI THỨ TỰ TRỤC
```

Bước ① là lý do `observation.camera_extrinsic` **bắt buộc phải đúng**: nó là **ma trận hiệu chuẩn
lắp camera**, không phải "pose của frame đầu". Nhân với nghịch đảo của nó biến pose-camera thành
pose-thân-xe.

Bước ③ cho ra hệ toạ độ cuối cùng mà **mọi nhãn của S1 sống trong đó**:

```
x = TRƯỚC (+)      y = TRÁI (+)      z = CAO (+)
```

Kiểm chứng (chạy lại đúng công thức trên data thật, `start = 0`):

```
[[0.     0.      0.]        frame 0 = gốc
 [0.0402 -0.0003 0.]        tiến 4 cm
 [0.0804 -0.0011 0.]
 ...
 [2.4139 -1.2343 0.]]       frame 77: tiến 2.41 m, lệch PHẢI 1.23 m
```

z luôn = 0 vì camera giữ nguyên độ cao ✓.

### 6.3. `xyz_to_xyt()` — thêm góc

[dataset:312-320](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L312): `θ` là góc giữa
**vector di chuyển đầu tiên** (`init_vector`) và vector di chuyển tại bước hiện tại, tính bằng
`arctan2(cross, dot)`. → `θ` là **hướng đi tương đối**, không phải yaw tuyệt đối. Mảng kết quả ngắn
hơn đầu vào 1 phần tử.

### 6.4. `process_actions()` — nhãn chính + bản "xoay lệch"

[dataset:322-384](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L322):

1. Đưa mọi pose trong cửa sổ `[start, end]` về hệ robot → `label_actions`.
2. **Xoay ngẫu nhiên toàn bộ quỹ đạo ±60°** ([338-346]).
3. Đưa ngược về world → làm mượt bằng **cubic spline** → lấy lại về hệ robot ([350-382]).
4. Lấy mẫu tại `action_indexes = clip(arange(25) * 4, 0, len-2)` ([383]).

Quỹ đạo xoay này **không phải augmentation của nhãn** — nó là **mẫu âm** để dạy critic: "đường này
lệch khỏi đường chuyên gia, hãy chấm điểm nó" (mục 7).

### 6.5. Nhãn cuối cùng: **hiệu số × 4**

[dataset:524-525](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L524):

```python
pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0     # 25 mốc → 24 hiệu số
```

Model **không** dự đoán vị trí tuyệt đối mà dự đoán **24 bước dịch chuyển**. Lúc chạy thật, quỹ đạo
được dựng lại bằng `torch.cumsum(naction / 4.0)`
([navdp_policy:319](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L319)).

Ba con số **quan trọng nhất của cả tài liệu này**, tính ra từ `predict_size=24`, `pred_digit=4`:

| Đại lượng | Công thức | Đo thật trên scene mẫu |
|---|---|---|
| Cửa sổ dự đoán | `24 × 4 = 96` frame | 3.2 giây @30 fps |
| Tầm nhìn xa | 96 × bước đi | ≈ **3.5 m** |
| Biên độ nhãn | `‖4 × Δ(4 frame)‖` | max **0.65 / 0.44 / 0.73** cho `(x, y, θ)` |

> 🚨 **Ràng buộc cứng: nhãn phải nằm trong `[-1, 1]`.** `DDPMScheduler` đặt `clip_sample=True`
> ([navdp_policy:119-121](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L119)) → lúc
> sinh, mẫu bị kẹp về `[-1,1]`. Nhãn vượt ngưỡng thì model **không bao giờ sinh ra được**.
> Quy đổi: **dịch chuyển giữa 2 frame ≤ 0.0625 m** (vì `4 × 4 × d ≤ 1`). Data gốc 0.037 m → an toàn.
> Robot của bạn chạy 1 m/s mà log 10 Hz → 0.1 m/frame → **vỡ ngưỡng**. Cách xử lý:
> [06b](06b_pipeline_mcap_to_s1.md) mục 4.4.

> 📌 **Episode ngắn hơn 96 frame thì đuôi nhãn bằng 0.** `action_indexes` bị `clip` nên các mốc cuối
> trùng nhau → hiệu số = 0. Đo thật trên episode 78 frame: 3 hàng cuối của `pred_actions` đúng bằng
> `[0,0,0]`. Model hiểu đó là "dừng lại" — hợp lý, nhưng **đừng để dataset toàn episode ngắn**, nếu
> không model sẽ học thói quen dừng sớm.

---

## 7. Critic — "giám khảo" chấm độ an toàn

[dataset:471-494](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L471):

```python
pred_distance = |traj_xy - obstacle_xy|.sum(-1).min(-1)     # KHOẢNG CÁCH L1, chỉ dùng x,y
pred_critic   = -5.0 * (pred_distance < 0.1).mean()          # phạt: tỉ lệ điểm quá sát vật cản
              + 0.5 * (pred_distance[1:] - pred_distance[:-1]).sum()   # thưởng: rời xa vật cản
```

Ba điều đáng nhớ:

1. **Khoảng cách là L1 (Manhattan), không phải Euclid** — `np.abs(...).sum(axis=-1)`. Ngưỡng `0.1`
   vì thế là "tổng lệch x + lệch y < 10 cm", chặt hơn khoảng cách Euclid.
2. **Chỉ dùng `x, y`** (`[:, 0:2]`) → **toạ độ z của `pointcloud.ply` hoàn toàn không ảnh hưởng**.
   Đo thật: điểm vật cản trong scene mẫu có `z ∈ [−0.100, 0.100]` — một lát mỏng quanh mặt sàn.
   → Bạn chỉ cần một **bản đồ occupancy 2D** ([05](05_data_train_s1.md) mục 5.2).
3. **Không có điểm vật cản → `critic = 2.0` cố định** ([490-494]) → nhánh critic mất hoàn toàn tín
   hiệu học, mà **không có lỗi nào báo ra**.

Đo lại `pointcloud.ply` của scene mẫu (88 750 điểm): `[102,102,102]` xám 49 014 điểm (đi được),
**`[0,0,128]` 27 276 điểm (vật cản)**, `[125,255,122]` 2 591 điểm — khớp đúng số điểm bộ lọc
`|color − (0,0,0.5)| < 0.05` chọn ra.

---

## 8. Chặng ⑥ — Loss: bốn thành phần

[navdp_trainer.py:90-100](../../../code/internnav/trainer/navdp_trainer.py#L90):

```python
ng_action_loss = (pred_ng - ng_noise).square().mean()      # nhánh KHÔNG có đích
mg_action_loss = (pred_mg - mg_noise).square().mean()      # nhánh CÓ đích (trộn ngẫu nhiên)
action_loss    = 0.5 * mg_action_loss + 0.5 * ng_action_loss
aux_loss       = 0.5*(batch_pg - imagegoal_aux_pred).square().mean() \
               + 0.5*(batch_pg - pixelgoal_aux_pred).square().mean()
critic_loss    = (critic_pred  - batch_label_critic).square().mean() \
               + (augment_pred - batch_augment_critic).square().mean()

loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
```

| Thành phần | Hệ số | Dạy model điều gì |
|---|---|---|
| `ng_action_loss` | 0.8 × 0.5 | Đi tiếp **khi không biết đích** (khám phá) — điều kiện đầu vào bị thay bằng vector 0 ([navdp_policy:216](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L216)) |
| `mg_action_loss` | 0.8 × 0.5 | Đi tới đích, với **đích được bốc ngẫu nhiên** trong 3 kiểu |
| `critic_loss` | 0.2 | Chấm điểm an toàn cho quỹ đạo thật **và** quỹ đạo bị xoay lệch |
| `aux_loss` | 0.5 | Ép `image_goal` và `pixel_goal` **giải mã ra được `point_goal`** — tức ba kiểu đích phải nằm chung một không gian nghĩa |

> 🔑 **Đây là mô hình diffusion**, nên nhãn hồi quy **không phải quỹ đạo** mà là **nhiễu**:
> [navdp_policy:148-157](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L148) cộng
> nhiễu vào nhãn ở một bước thời gian ngẫu nhiên (`num_train_timesteps=10` — rất ít, nên sinh nhanh),
> model học **đoán lại phần nhiễu đó**. Vì vậy **`loss` của S1 không so sánh được với `loss` của
> S2**, và loss ~0.9 lúc đầu là bình thường.

**Trộn đích ngẫu nhiên** ([navdp_policy:219-235](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L219)):
3 khe điều kiện, mỗi khe chọn 1 trong 3 kiểu đích theo công thức `batch_index % 27` — **tất định
theo vị trí trong batch**, không phải random thật. Nghĩa là **batch phải đủ lớn** (≥27) thì model
mới thấy đủ 27 tổ hợp. `batch_size=32` vừa đủ; giảm xuống 8 để tiết kiệm VRAM là **âm thầm bỏ mất
2/3 số tổ hợp**.

### 8.1. `psutil` — phụ thuộc ẩn, in log mỗi step

[navdp_trainer.py:39-54](../../../code/internnav/trainer/navdp_trainer.py#L39): `compute_loss`
`import psutil` và in số tiến trình con **mỗi bước train**. Thiếu `psutil` → crash ở step đầu tiên
(không phải lúc khởi động). Đây rõ ràng là code debug còn sót; xoá được nếu log quá ồn.

---

## 9. `NavDPTrainer` ghi đè những gì

[internnav/trainer/navdp_trainer.py](../../../code/internnav/trainer/navdp_trainer.py) kế thừa
`BaseTrainer` (→ `transformers.Trainer`) và ghi đè:

| Phương thức | Dòng | Hành vi |
|---|---|---|
| `compute_loss` | [26](../../../code/internnav/trainer/navdp_trainer.py#L26) | Mục 8 |
| `create_optimizer` | [124](../../../code/internnav/trainer/navdp_trainer.py#L124) | `torch.optim.Adam(lr=config.il.lr)` — **bỏ qua `weight_decay`** và `optim='adamw_torch'` trong `TrainingArguments` |
| `create_scheduler` | [154](../../../code/internnav/trainer/navdp_trainer.py#L154) | `LinearLR` 1.0 → 0.5 trong **10 000 iteration** — **bỏ qua** `lr_scheduler_type='cosine'` |
| `create_optimizer_and_scheduler` | [159](../../../code/internnav/trainer/navdp_trainer.py#L159) | Chặn hoàn toàn cơ chế của HF |
| `get_train_dataloader` | [171](../../../code/internnav/trainer/navdp_trainer.py#L171) | Luôn dựng `DistributedSampler` (kể cả 1 GPU), `drop_last=True` |
| `save_model` | [188](../../../code/internnav/trainer/navdp_trainer.py#L188) | Mục 9.3 |

### 9.1. Hệ quả: nửa số cờ trong `TrainingArguments` là trang trí

[train.py:225-254](../../../code/scripts/train/base_train/train.py#L225) khai báo `optim`,
`learning_rate`, `lr_scheduler_type`, `weight_decay`… nhưng **đều bị ghi đè**. Những cờ **thật sự có
tác dụng**:

```python
bf16=False, tf32=False            # ⚠️ train ở FP32 → tốn VRAM; đổi bf16=True nếu GPU hỗ trợ
gradient_checkpointing=False      # bật lên nếu thiếu VRAM
save_strategy='epoch'             # ⚠️ save_steps=5 BỊ BỎ QUA khi strategy='epoch'
save_total_limit=8                # giữ 8 checkpoint gần nhất
dataloader_drop_last=True
ddp_find_unused_parameters=True   # cần, vì nhánh no-goal/critic không dùng hết tham số
```

> ⚠️ `save_interval_epochs=5` trong config được gán vào `save_steps`, nhưng `save_strategy='epoch'`
> nên `save_steps` vô hiệu → **lưu sau MỖI epoch**, không phải mỗi 5 epoch.

### 9.2. Điểm nghẽn tốc độ nằm ở `__getitem__`

Mỗi mẫu phải: đọc parquet, **đọc lại `pointcloud.ply` (~2.4 MB, 88 750 điểm)**, mở 9 ảnh, chạy 2 lần
cubic spline. Không có cache. Dataset tự in thời gian trung bình mỗi
`batch_size` mẫu ([dataset:544-548](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L544)) —
nhìn con số này để biết có nghẽn dataloader không. Nếu nghẽn: tăng `num_workers`, hoặc chuyển
`pointcloud.ply` sang dạng `.npy` chỉ chứa điểm vật cản.

### 9.3. 🚨 Lỗi tên file checkpoint

[navdp_trainer.py:203](../../../code/internnav/trainer/navdp_trainer.py#L203):

```python
torch.save(model_to_save.state_dict(), output_dir + "navdp.ckpt")     # ← THIẾU dấu '/'
```

HuggingFace truyền `output_dir = checkpoints/<name>/ckpts/checkpoint-500` (không có `/` cuối) →
file thật được ghi ra là:

```
checkpoints/<name>/ckpts/checkpoint-500navdp.ckpt      ← nằm NGANG HÀNG thư mục, không nằm TRONG
checkpoints/<name>/ckpts/checkpoint-500/               ← thư mục rỗng (chỉ có trainer_state.json)
```

→ Lúc đi tìm checkpoint, đừng ngạc nhiên khi thư mục `checkpoint-500/` trống. Sửa thành
`os.path.join(output_dir, "navdp.ckpt")` nếu muốn gọn.

Ngoài ra file lưu là **`state_dict` thuần** (không phải định dạng `save_pretrained`) → khi nạp lại
phải đi vào nhánh `else` của `from_pretrained` (mục 4.2), tức **truyền đường dẫn tới đúng file
`.ckpt`**, không phải thư mục.

---

## 10. Fine-tune từ checkpoint có sẵn — 2 bước

Trái với S2, ở đây fine-tune là **lựa chọn**, không phải mặc định.

```python
# scripts/train/base_train/configs/navdp.py
il=IlCfg(
    ckpt_to_load='checkpoints/navdp/navdp.ckpt',   # file .ckpt, KHÔNG phải thư mục
    lr=2e-5,                                       # giảm lr khi học tiếp
    ...
)
```

và đổi `name` khi chạy để **nơi lưu khác nơi nạp**:

```bash
bash scripts/train/base_train/start_train.sh --name navdp_ft --model navdp
```

> 📌 Nạp bằng `strict=False` ([navdp_policy:60](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L60))
> → **khoá thiếu/thừa bị bỏ qua im lặng**. Nếu bạn đổi `token_dim`/`memory_size`/`predict_size` so
> với checkpoint, phần lớn trọng số sẽ **không** được nạp mà **không có lỗi**. In `incompatible_keys`
> (code đã in sẵn) và **đọc nó**.

---

## 11. Bảng tra nhanh: "muốn đổi X thì sửa ở đâu"

| Muốn đổi | Sửa chỗ nào |
|---|---|
| Số GPU | `case "navdp")` trong [start_train.sh:47](../../../code/scripts/train/base_train/start_train.sh#L47) **và** `torch_gpu_ids` trong config |
| Thư mục data | `il.root_dir` |
| File cache danh sách episode | `il.dataset_navdp` (+ đặt `preload=True` cho lần chạy sau) |
| Checkpoint khởi đầu | `il.ckpt_to_load` (đường dẫn tới **file** `.ckpt`) |
| Tốc độ học | `il.lr` (scheduler cố định `LinearLR 1.0→0.5 / 10k iter`, sửa ở [navdp_trainer:156](../../../code/internnav/trainer/navdp_trainer.py#L156)) |
| Batch mỗi GPU | `il.batch_size` — **đừng để < 27**, xem mục 8 |
| Tầm nhìn xa của quỹ đạo | `il.predict_size` (và `pred_digit=4` hard-code ở [dataset:446](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L446)) |
| Độ dài bộ nhớ | `il.memory_size` (đổi là **phải train lại**: `cond_pos_embed` phụ thuộc nó) |
| Kích thước ảnh | `il.image_size` (224 — khớp patch 14×14 của ViT, đổi phải cẩn thận) |
| Số kênh pixel-goal | `il.pixel_channel` (4 hoặc 7, phải khớp cả dataset lẫn model) |
| Dùng ít scene hơn | `il.scene_scale` (<1.0 = lấy thưa; >1.0 = **lặp lại** scene) |
| Giãn cách bộ nhớ ngẫu nhiên | `il.random_digit=True` → `digit ∈ [2,8)` thay vì cố định 4 |
| Tiết kiệm VRAM | `bf16=True` / `gradient_checkpointing=True` tại [train.py:229-231](../../../code/scripts/train/base_train/train.py#L229) |

---

## 12. Bảng bẫy đã phát hiện

| # | Bẫy | Triệu chứng | Cách tránh |
|---|---|---|---|
| 1 | Thiếu `checkpoints/depth_anything_v2_vits.pth` | Crash lúc dựng model | Tải đúng bản `vits` (khác bản `metric_hypersim` của nhánh dual) |
| 2 | Thiếu `open3d` / `jsonlines` / `psutil` | Crash ở import, hoặc ở **step train đầu tiên** (psutil) | `pip install open3d jsonlines psutil` |
| 3 | Thư mục cha của `dataset_navdp` chưa tồn tại | Quét xong hết mới `FileNotFoundError` | `mkdir -p data/datasets` |
| 4 | Đường dẫn giải nén là `traij_data` (lỗi chính tả trong archive gốc) mà config ghi `traj_data` | `root_dir` rỗng → dataset 0 mẫu | Đổi tên thư mục hoặc sửa `root_dir` |
| 5 | Tên ảnh không đệm 0 | `sorted()` sai thứ tự → ảnh ghép nhầm episode, **không báo lỗi** | `episode_{i:06d}_{f:03d}.jpg` |
| 6 | `image_index` tính theo từng episode thay vì toàn cục | Ảnh lệch hẳn sang episode khác | Xem mục 5.2 |
| 7 | Scene có nhiều hơn 1 chunk | Mất data im lặng | Gộp về `chunk-000` |
| 8 | Episode < 4 frame | `ValueError` giữa lúc train | Lọc bỏ khi tạo data |
| 9 | Bước đi > 0.0625 m/frame | Nhãn vượt `[-1,1]`, model không sinh lại được | Lấy mẫu lại quỹ đạo, mục 6.5 |
| 10 | `pointcloud.ply` không có điểm màu `(0,0,128)` | Không crash, `critic = 2.0` hằng số → mất khả năng né vật cản | Kiểm bằng cách đếm điểm sau bộ lọc |
| 11 | `batch_size` < 27 | Model chỉ thấy một phần tổ hợp đích | Giữ ≥ 32 |
| 12 | Tìm checkpoint trong `checkpoint-N/` | Thư mục rỗng | File thật là `checkpoint-Nnavdp.ckpt`, mục 9.3 |
| 13 | `prior_sample=True` | `TypeError: rank_steps() missing 2 args` ([dataset:434](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L434) gọi thiếu tham số) | Để `False` |

---

## 13. Chạy thử tối thiểu (1 GPU)

```bash
# 1. phụ thuộc
pip install open3d jsonlines psutil tyro

# 2. trọng số thị giác
mkdir -p checkpoints && \
  wget -O checkpoints/depth_anything_v2_vits.pth <url_depth_anything_v2_vits>

# 3. data (giải nén archive vln_n1 vào đúng đường dẫn root_dir)
mkdir -p data/datasets/InternData-N1/vln_n1/traj_data data/datasets

# 4. sửa configs/navdp.py:  torch_gpu_ids=[0], batch_size=32, epochs=1
# 5. sửa start_train.sh:    CUDA_VISIBLE_DEVICES=0 ; NUM_GPUS=1
bash scripts/train/base_train/start_train.sh --name navdp_smoke --model navdp
```

Dấu hiệu **chạy đúng**:

- log in `Total trainable parameters: …` (từ [navdp_trainer:150](../../../code/internnav/trainer/navdp_trainer.py#L150));
- có dòng `__getitem__ pid=… avg_time(last 32)=…s` → dataloader đọc được data;
- `loss` hữu hạn và giảm dần;
- sau epoch đầu xuất hiện file `checkpoints/navdp_smoke/ckpts/checkpoint-<N>navdp.ckpt`.

---

Tiếp theo: tự sinh data S1 từ log robot → [06b_pipeline_mcap_to_s1](06b_pipeline_mcap_to_s1.md).
Cấu trúc dữ liệu S1 (bảng bắt buộc/không bắt buộc): [05_data_train_s1](05_data_train_s1.md).
