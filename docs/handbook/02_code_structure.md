# 02 — Cấu trúc code InternVLA-N1: checkpoint load ở đâu, dữ liệu đi vào S1/S2 thế nào, hai system nối nhau và chạy bất đồng bộ ra sao

> **File này để làm gì:** trả lời 6 câu hỏi về cấu trúc code của đường dual-system trong repo
> `InternRobotics/InternNav`, mỗi câu có **code snippet nguyên văn + đường dẫn file + số dòng**:
> 1. [Checkpoint bắt đầu được load ở đâu?](#1-checkpoint-bắt-đầu-được-load-ở-đâu)
> 2. [Đường dẫn model được đọc từ đâu?](#2-đường-dẫn-model-được-đọc-từ-đâu)
> 3. [Gọi function nào để truyền dữ liệu vào System 2?](#3-gọi-function-nào-để-truyền-dữ-liệu-vào-system-2)
> 4. [Gọi function nào để truyền dữ liệu vào System 1?](#4-gọi-function-nào-để-truyền-dữ-liệu-vào-system-1)
> 5. [System 2 kết nối với System 1 như thế nào?](#5-system-2-kết-nối-với-system-1-như-thế-nào)
> 6. [System 2 và System 1 chạy bất đồng bộ ra sao?](#6-system-2-và-system-1-chạy-bất-đồng-bộ-ra-sao)
>
> **Số dòng** ghi theo bản clone local `InternNav/code` (InternNav **v0.3.1**, commit `7a5c624`) —
> đã xác minh từng snippet ngày 23/07/2026. Bằng chứng chi tiết: các mục **PL-C** trong
> [05_appendix.md](05_appendix.md).

---

## 0. Bức tranh tổng thể trước khi đọc chi tiết

InternVLA-N1 là model **dual-system** — hai "bộ não" trong một checkpoint:

- **System 2 (S2)** — VLM Qwen2.5-VL-7B, "suy nghĩ chậm": nhìn ảnh + đọc câu lệnh → quyết định
  *đi tới điểm nào trên ảnh* (pixel goal) hoặc *hành động rời rạc* (tiến/rẽ trái/rẽ phải/cúi/dừng).
- **System 1 (S1)** — NavDP, diffusion policy, "phản xạ nhanh": nhận "kế hoạch ẩn" (latent) từ S2
  + RGB + depth → sinh **quỹ đạo di chuyển** (chuỗi waypoint).

Các file cần biết (đường dẫn tính từ gốc repo InternNav):

| File | Vai trò |
|---|---|
| `scripts/eval/eval.py` | Entry point eval: đọc file config, dựng Evaluator |
| `scripts/eval/configs/habitat_dual_system_cfg.py` | Config mẫu — **nơi khai `model_path`** |
| `internnav/agent/internvla_n1_agent.py` | `InternVLAN1Agent` — vòng đời một step: nhận obs, điều phối S1/S2, thread |
| `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` | `InternVLAN1Net` — **nơi load checkpoint**, chứa `s2_step` / `s1_step_latent` |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py` | `InternVLAN1ForCausalLM` — model class kế thừa Qwen2.5-VL, thêm `generate_latents` / `generate_traj` |
| `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` | `InternVLAN1MetaModel` — dựng module S1 (`self.navdp`) + `latent_queries` theo `config.system1` |
| `internnav/model/basemodel/internvla_n1/navdp.py` | Kiến trúc NavDP (S1) + loader standalone |
| `internnav/model/utils/vln_utils.py` | Dataclass `S2Input/S2Output/S1Input/S1Output` — "hợp đồng dữ liệu" giữa agent và policy |

Chuỗi gọi rút gọn:

```
eval.py --config habitat_dual_system_cfg.py
  └─ Evaluator → InternVLAN1Agent.__init__          (agent)
       └─ get_policy('InternVLAN1_Policy') → InternVLAN1Net.__init__     (policy)
            └─ InternVLAN1ForCausalLM.from_pretrained(model_path)        ← LOAD CHECKPOINT (S1+S2)
  mỗi step:
  agent.step(obs{rgb, depth, instruction})
    ├─ (thread riêng)  policy.s2_step(...)   → pixel goal + latent, HOẶC action rời rạc
    └─ (main thread)   policy.s1_step_latent(rgbs, depths, latent) → quỹ đạo → action
```

---

## 1. Checkpoint bắt đầu được load ở đâu?

### 1.1. Điểm load duy nhất — `InternVLAN1Net.__init__`

📄 `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`, **dòng 33–43**:

```python
self.model = InternVLAN1ForCausalLM.from_pretrained(
    self.model_config.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": self.model_config.device},
)

self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path, use_fast=True)
self.processor = AutoProcessor.from_pretrained(self.model_config.model_path)
```

Đây là **lệnh load duy nhất cho CẢ S1 lẫn S2** — S1 không có file weights riêng, toàn bộ nằm chung
trong 4 shard safetensors của checkpoint (bằng chứng: PL-B1, PL-B3, PL-C1).

Hai điều phải biết về lời gọi này khi chạy trên Kaggle:
- `attn_implementation="flash_attention_2"` hardcode → T4 không chạy được, phải ép `"sdpa"`.
- `device_map={"": device}` pin **toàn bộ 16.79 GB vào MỘT GPU** → OOM trên 1 T4; phải ép `"auto"`
  (PL-C3). Cách override: [01_system_requirements](01_system_requirements.md) mục 3 bước 6.

### 1.2. Vì sao load một lệnh mà ra được hai system — kiến trúc phải "dựng chỗ chứa" trước

`from_pretrained` làm 2 việc theo thứ tự: (a) dựng kiến trúc model từ `config.json`,
(b) đổ weights từ shard vào các module trùng tên. Bước (a) đi qua:

📄 `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`, **dòng 121–145**
(`InternVLAN1MetaModel.__init__`):

```python
if hasattr(config, "system1"):                                    # dòng 124 — CỔNG QUAN TRỌNG
    self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))

    if 'nextdit' in config.system1:
        self.traj_dit, self.noise_scheduler = build_traj_dit(config)
        ...
    elif 'navdp' in config.system1:
        if 'async' in config.system1:
            self.navdp = build_navdp(config, memory_size=2)       # dòng 143 — dựng S1 NavDP
```

và `build_navdp` (cùng file, dòng 10–15) tạo `NavDP_Policy_DPT_CriticSum_DAT` rồi gọi
`navdp.load_model()` — hàm này ở 📄 `navdp.py` dòng 116–125: nếu `navdp_pretrained=None` thì chỉ
random init (weights thật sẽ được `from_pretrained` ghi đè ở bước (b)); nếu được set path thì
`torch.load` một file `.ckpt` riêng — đây là đường chạy **NavDP standalone** không cần checkpoint 16.79 GB.

**Hệ quả thực tế (đọc kỹ, dễ mất nhiều giờ debug):**
- `config.json` **không có field `system1`** → dòng 124 False → không dựng `navdp`/`latent_queries`
  → khi đổ weights, toàn bộ tensor S1 không có chỗ chứa → bị bỏ qua **im lặng**.
- Checkpoint đã đóng gói (`tieulam/internvla-n1-ckpt`) rơi đúng vào trường hợp này — config của nó
  thiếu `system1` (xác minh 23/07 — PL-B6). Checkpoint `InternVLA-N1-w-NavDP` thì có sẵn
  `system1: "navdp_async"`. Chi tiết chọn bản nào: [04_checkpoint_details](04_checkpoint_details.md).
- Load bằng class HF thuần (`Qwen2_5_VLForConditionalGeneration`) thì **chắc chắn** mất S1 —
  kiến trúc Qwen không có chỗ chứa `navdp.*` (~120 tensor `UNEXPECTED` đo thật — PL-B3).

---

## 2. Đường dẫn model được đọc từ đâu?

Đường dẫn checkpoint **không hardcode trong model** — nó chảy từ file config qua 4 trạm:

**Trạm 1 — file config eval.** 📄 `scripts/eval/configs/habitat_dual_system_cfg.py`, dòng 4–17:

```python
eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",                            # dual_system hoặc system2
            "model_path": "checkpoints/InternVLA-N1-DualVLN", # ← ĐƯỜNG DẪN CHECKPOINT KHAI Ở ĐÂY
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            ...
        },
    ),
    ...
)
```

(Bản v0.3.1 mặc định trỏ checkpoint DualVLN; muốn chạy bản NavDP thì đổi giá trị này thành đường dẫn
checkpoint NavDP của bạn.)

**Trạm 2 — entry point đọc file config bằng importlib.** 📄 `scripts/eval/eval.py`:

```python
def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    ...
    return getattr(config_module, attr_name)

evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')   # --config <file.py>
evaluator = Evaluator.init(evaluator_cfg)
```

**Trạm 3 — agent gói `model_settings` thành `ModelCfg` rồi đưa cho policy.**
📄 `internnav/agent/internvla_n1_agent.py`, dòng 33–42:

```python
vln_sensor_config = self.config.model_settings
self._model_settings = ModelCfg(**vln_sensor_config)          # model_path nằm trong đây
...
policy = get_policy(self._model_settings.policy_name)         # 'InternVLAN1_Policy' → InternVLAN1Net (PL-C6)
policy_config = get_config(self._model_settings.policy_name)
model_config = {'model': self._model_settings.model_dump()}
self.policy = policy(config=policy_config(model_cfg=model_config))
```

**Trạm 4 — policy lấy ra và dùng.** 📄 `internvla_n1_policy.py`, dòng 31 → 34:

```python
self.model_config = ModelCfg(**config.model_cfg['model'])
self.model = InternVLAN1ForCausalLM.from_pretrained(self.model_config.model_path, ...)
```

`ModelCfg` (📄 `internnav/configs/model/base_encoders.py`, dòng 181+) là pydantic model với
`extra='allow'` — nên mọi key bạn viết trong dict `model_settings` đều đi xuyên suốt tới policy.

> **Tóm tắt cho người mới:** muốn đổi checkpoint, chỉ cần sửa **một chỗ** — key `"model_path"` trong
> file config eval (Trạm 1), hoặc tự dựng `ModelCfg` tương đương khi gọi policy trực tiếp trong notebook.

---

## 3. Gọi function nào để truyền dữ liệu vào System 2?

### 3.1. Function: `InternVLAN1Net.s2_step(rgb, depth, pose, instruction, intrinsic, look_down)`

📄 `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`, **dòng 110**:

```python
def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
```

Nơi gọi nó trong luồng chuẩn — thread S2 của agent, 📄 `internvla_n1_agent.py`, dòng 158–166:

```python
with self.s2_agent_lock:
    current_s2_output = self.policy.s2_step(
        self.s2_input.rgb,          # np.ndarray HxWx3 (RGB hiện tại)
        self.s2_input.depth,        # np.ndarray depth (S2 không đưa vào prompt, giữ cho bước sau)
        self.s2_input.pose,         # agent truyền ma trận đơn vị 4x4 (PL-C5)
        self.s2_input.instruction,  # câu lệnh tiếng Anh, ví dụ từ meta/tasks.jsonl của vln_ce
        self.camera_intrinsic,      # ma trận nội tại camera — agent tự dựng từ width/height/hfov (dòng 45–47)
        self.s2_input.look_down,    # True nếu là frame "cúi nhìn xuống"
    )
```

Hợp đồng dữ liệu vào/ra ở tầng agent — 📄 `internnav/model/utils/vln_utils.py`, dòng 140–165:

```python
@dataclass
class S2Input:
    idx: Optional[int] = -1
    instruction: Optional[str] = None
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    pose: Optional[Tuple[float, float, float]] = None
    look_down: Optional[bool] = False
    should_infer: Optional[bool] = False

@dataclass
class S2Output:
    output_action: Optional[np.ndarray] = None      # chuỗi action rời rạc {0,1,2,3,5}
    output_pixel: Optional[np.ndarray] = None       # pixel goal [row, col]
    output_latent: Optional[torch.Tensor] = None    # "kế hoạch ẩn" → feed System 1
    rgb_memory: Optional[np.ndarray] = None         # RGB tại frame chấm goal (cho S1)
    depth_memory: Optional[np.ndarray] = None       # depth tại frame chấm goal (cho S1)
    ...
```

### 3.2. Bên trong `s2_step` làm gì với dữ liệu của bạn (tóm tắt có số dòng)

1. **Resize ảnh về 384×384** và tích vào `rgb_list` làm history (dòng 113–116; số 384 từ config
   `resize_w/h` — mục 2 Trạm 1).
2. **Dựng prompt**: template *"You are an autonomous navigation assistant. Your task is to
   `<instruction>`… Please output the next waypoint's coordinates in the image…"* (dòng 64, đã thay
   instruction ở dòng 124) + history lấy mẫu đều `np.linspace(0, episode_idx-1, num_history=8)`
   (dòng 130) + câu chốt `"you can see <image>."` (dòng 148–149).
3. **Generate**: `max_new_tokens=128, do_sample=False` (dòng 169–176).
4. **Parse output** (dòng 184–196):

```python
if bool(re.search(r'\d', self.llm_output)):          # text CÓ chữ số → nhánh pixel goal
    coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
    pixel_goal = [int(coord[1]), int(coord[0])]      # ⚠️ đảo thứ tự → [row, col]
    output.output_pixel = np.array(pixel_goal)
    ...
    traj_latents = self.model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)
    output.output_latent = traj_latents              # ← thứ System 1 cần
else:                                                # KHÔNG có số → nhánh action rời rạc
    action_seq = self.parse_actions(self.llm_output) # map 'STOP','↑','←','→','↓' → [0,1,2,3,5]
    output.output_action = action_seq
```

> ⚠️ Ba bẫy parser đã ghi nhận: (a) đảo toạ độ `[row,col]` trong khi data `vln_ce` lưu goal `[u,v]`
> — ✅ đo 23/07 (PL-E3): toạ độ model nhả ở **không gian 640×480 gốc** (dù input đã resize 384), cùng
> hệ với GT → khi so sánh chỉ cần đảo thứ tự, **KHÔNG scale 384→640**; (b) chỉ cần *một chữ số bất
> kỳ* trong text là bị coi là pixel goal; (c) text không số + không ký tự action → trả list rỗng,
> không báo lỗi. Chi tiết: `../io_system2.md` mục 3.d.
> Chạy thật 23/07: S2 **ra được pixel+latent** qua cơ chế look-down 2 nhịp (file 06 Phase E, PL-E3)
> — nghi vấn PL-E1 cũ ("chỉ ra action") đã đóng.

---

## 4. Gọi function nào để truyền dữ liệu vào System 1?

### 4.1. Function: `InternVLAN1Net.s1_step_latent(rgb, depth, latent)`

📄 `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`, **dòng 200–215** (nguyên văn):

```python
def s1_step_latent(self, rgb, depth, latent):
    with torch.no_grad():
        dp_actions = self.model.generate_traj(
            traj_latents=latent, images_dp=rgb, depths_dp=depth
        )  # use_aysnc based on MODEL

    if self.continuous_traj:
        action_list = traj_to_actions(dp_actions)          # quỹ đạo liên tục → chuỗi action
    else:
        random_choice = np.random.choice(dp_actions.shape[0])
        action_list = chunk_token(dp_actions[random_choice])

    action_list = [x for x in action_list if x != 0]
    output = S1Output(idx=action_list[:4])                 # lấy tối đa 4 action đầu
    return output
```

Ba đầu vào: `latent` (từ `S2Output.output_latent` — mục 5), `rgb`, `depth`. Lõi tính toán nằm ở
`generate_traj` — 📄 `internvla_n1.py`, dòng 349+; với NavDP nó rẽ vào nhánh dòng 434–436:

```python
elif 'navdp' in self.get_system1_type():
    if 'async' in self.get_system1_type():
        all_trajs = self.model.navdp.predict_pointgoal_action_async(...)
```

(một lần nữa thấy `config.system1` quyết định đường chạy — PL-C2).

### 4.2. Agent chuẩn bị dữ liệu cho S1 như thế nào (quan trọng khi bạn tự feed data)

📄 `internvla_n1_agent.py`, dòng 304–336 — nhánh `partial_async` (đủ tiền xử lý):

```python
if self.s2_output.output_latent is not None:
    if mode != 'sync':
        # cặp ảnh: (1) frame lúc S2 chấm goal — "rgb_memory", (2) frame hiện tại
        processed_pixel_rgb  = np.array(Image.fromarray(self.s2_output.rgb_memory).resize((224, 224))) / 255.0
        processed_pixel_depth = np.array(Image.fromarray(self.s2_output.depth_memory[:, :, 0]).resize((224, 224))) * 10.0
        processed_pixel_depth[processed_pixel_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold  # clip 5.0 m

        processed_rgb   = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
        processed_depth = np.array(Image.fromarray(depth[:, :, 0]).resize((224, 224))) * 10.0   # "should be 0-10m"
        processed_depth[processed_depth > self.sys1_depth_threshold] = self.sys1_depth_threshold

        rgbs   = torch.stack([...]).unsqueeze(0).to(self.device)                 # [1, 2, 224, 224, 3]
        depths = torch.stack([...]).unsqueeze(0).unsqueeze(-1).to(self.device)   # [1, 2, 224, 224, 1]
        self.s1_output = self.policy.s1_step_latent(rgbs, depths, self.s2_output.output_latent)
    else:
        self.s1_output = self.policy.s1_step_latent(rgb, depth * 10000.0, self.s2_output.output_latent)
```

Những con số phải nhớ khi tự chuẩn bị input cho S1:
- Ảnh resize **224×224**, RGB chia 255 về [0,1].
- Depth nhân 10 → **mét trong khoảng 0–10 m** (comment nguyên văn trong code: *"should be 0-10m"* —
  vì depth của Habitat được chuẩn hoá [0,1]), rồi **clip tại 5.0 m** (`sys1_depth_threshold`, dòng 59).
- S1 nhận **cặp 2 frame**: frame lúc S2 chấm goal (`rgb_memory`/`depth_memory` — agent lưu ở dòng
  201–202) + frame hiện tại. Shape cuối: rgbs `[1, 2, 224, 224, 3]`, depths `[1, 2, 224, 224, 1]`.
- ⚠️ Nếu dữ liệu của bạn là depth **milimét** (như PNG của `vln_ce` — PL-D4) thì phép "×10" ở trên
  KHÔNG áp dụng nguyên xi — phải tự quy về mét trước. Xem hướng dẫn cụ thể ở
  [03_data_contract](03_data_contract.md) mục 4.

### 4.3. Cấu trúc output của S1 — mổ xẻ `S1Output` (đọc code 23/07/2026)

**a. Dataclass** — 📄 `internnav/model/utils/vln_utils.py:177–184`:

```python
@dataclass
class S1Output:
    idx: Optional[list] = None                # ← field DUY NHẤT được điền trong đường s1_step_latent
    trajectory: Optional[np.ndarray] = None   # None — dự phòng cho agent khác
    linear_velocity: Optional[float] = None   # None — dành cho robot thật (điều khiển vận tốc)
    angular_velocity: Optional[float] = None  # None — như trên
    vis_image: Optional[np.ndarray] = None    # None — ảnh debug
```

`s1_step_latent` chỉ tạo `S1Output(idx=action_list[:4])` (`internvla_n1_policy.py:214`) — 4 field
còn lại tồn tại vì dataclass dùng chung với agent realworld. **Đừng trông đợi `trajectory` có giá
trị** ở đường này; muốn waypoint xem mục (d).

**b. Bảng mã của `idx`** — list **tối đa 4 phần tử**, mỗi phần tử ∈ {1, 2, 3}:

| Mã | Nghĩa | Độ lớn vật lý | Nguồn số |
|---|---|---|---|
| `1` | ↑ tiến thẳng | **0.25 m** | `step_size=0.25` (`vln_utils.py:87`) |
| `2` | ← quay trái | **15°** | `turn_angle_deg=15` (`vln_utils.py:87`) |
| `3` | → quay phải | **15°** | như trên |
| `0` | stop | — | **bị lọc** trước khi trả (`internvla_n1_policy.py:212`) |

Mã `5` (look-down) **không bao giờ** xuất hiện từ S1 — nó là chuyện riêng của S2 (mục 3.2).
Ví dụ đọc: `idx == [2, 2, 1, 1]` = "quay trái 30° rồi tiến 0.5 m".

**c. Đường biến đổi latent → idx** (nhánh `navdp_async` + `continuous_traj=True` — cấu hình của
config chính chủ `h1_internvla_n1_async_cfg.py`):

| # | Bước | Code | Tensor/kết quả |
|---|---|---|---|
| 1 | Điều kiện hoá: latent S2 → goal embedding (`vlm_embed_mlp`+`goal_compressor`); cặp RGB-D → `rgbd_encoder` | `navdp.py:237–241` | latent (1, 4, 3584); rgbs [1,2,224,224,3], depths [1,2,224,224,1] |
| 2 | Diffusion: khởi tạo noise rồi khử nhiễu 20 bước DDPM | `navdp.py:242–253` | **(32, 32, 3)** — 32 ứng viên × 32 bước **(dx, dy, dyaw)** — ✅ `predict_size=32` đo runtime 23/07 (s1_traj trả (33, 2) = T+1) |
| 3 | Un-normalize (`xy /= 4`), cộng dồn delta thành đường xy, rồi **trung bình cộng cả 32** thành 1 quỹ đạo | `vln_utils.py:128–132` | (T+1, 2) — mét, xuất phát (0,0). ⚠️ **KHÔNG dùng critic** — `critic_head` có trong kiến trúc (`navdp.py:80`) nhưng đường async không gọi |
| 4 | "Robot ảo" bám quỹ đạo → mã rời rạc: quay từng nấc 15°, tiến 0.25 m/bước; dừng khi cách đích < 0.2 m hoặc tiến thêm lại xa đích hơn | `vln_utils.py:87–126` | list mã {0,1,2,3} độ dài tuỳ quỹ đạo |
| 5 | Lọc bỏ `0`, cắt **4 phần tử đầu** | `internvla_n1_policy.py:212–214` | `S1Output(idx=[...])` |

Con số 4 ở bước 5 khớp nhịp async (mục 6): S1 chỉ được tin tối đa 4 bước
(`num_future_steps: 4`) trước khi S2 kịp suy nghĩ lại.

**d. Liên hệ với data contract ([03_data_contract](03_data_contract.md)):**

- `idx` dùng **cùng bảng mã** với cột GT `action` của `vln_ce` (03 mục 1.2, 4.4) → có thể so thô
  `idx[0]` với GT từng frame như một tín hiệu định lượng phụ — nhưng nhớ giới hạn open-loop
  (03 mục 4.4: quỹ đạo S1 không có GT liên tục trong `vln_ce`).
- GT quỹ đạo liên tục thật sự chỉ có ở `vln_n1` (cột `action` SE(3) 4×4 — 03 mục 2.2–2.3); muốn
  chấm số cho waypoint phải tự quy đổi SE(3) → (dx, dy, dyaw).
- Muốn **waypoint để vẽ** thay vì action rời rạc: `traj_to_actions(trajs, use_discrate_action=False)`
  trả thẳng quỹ đạo trung bình (T+1, 2) mét — dùng cho visualize (file 06 bước F2).
  ⚠️ Hàm này un-normalize **in-place** (chia 4 trực tiếp vào tensor đầu vào) — đừng gọi 2 lần trên
  cùng một tensor.
- Nhánh `continuous_traj=False` (không dùng trong config của ta): chọn **ngẫu nhiên 1 trong 32**
  ứng viên rồi lượng tử hoá từng bước bằng `chunk_token` (`vln_utils.py:36–60`) — cùng bảng mã.

**e. Hiện trạng các field còn lại — và điều kiện để tự điền (grep toàn repo 23/07):**

Không một dòng code nào trong v0.3.1 điền `trajectory` / `linear_velocity` / `angular_velocity`;
riêng `vis_image` có *người tiêu thụ* (agent lưu PNG + ghi video — `internvla_n1_agent.py:395–399`)
nhưng không có *người sản xuất*.

| Field | Hiện trạng | Muốn có thì |
|---|---|---|
| `trajectory` | **Tính xong nội bộ rồi vứt** — quỹ đạo trung bình chỉ là bước trung gian sinh `idx` (bước 3–4 bảng trên) | Rẻ nhất: gọi `generate_traj` + `traj_to_actions(..., use_discrate_action=False)` bên ngoài policy — file 06 Phase F có sẵn hàm `s1_step_full` kiểu này |
| `linear/angular_velocity` | **Không tồn tại ở bất kỳ đâu trong luồng S1** | Tự viết: chọn chu kỳ điều khiển `dt` (giả định ngoài code) rồi tính v ≈ ‖Δxy‖/dt, ω ≈ Δyaw/dt; Δyaw nằm ở kênh `dp_actions[:,:,2]` (tự mean qua 32 ứng viên). Trong open-loop hai số này chỉ mang tính minh hoạ |
| `vis_image` | "Ổ cắm" có sẵn phía agent, không ai cắm | Tự vẽ (overlay waypoint); chỉ có ích khi chạy qua agent — đường policy-trực-tiếp tự visualize ngoài notebook |

⚠️ **Đừng nhầm về controller:** gắn robot controller vào cũng KHÔNG làm `S1Output` trả velocity.
Luồng thật của repo: `S1Output.idx → agent/env → controller đổi thành v, ω
(h1_vln_move_by_speed_controller.py — Isaac) → khớp robot` — vận tốc sinh ra **ở controller, phía
hạ nguồn**, và không ghi ngược về `S1Output`.

---

## 5. System 2 kết nối với System 1 như thế nào?

Cầu nối là **latent** — tensor "kế hoạch ẩn" S2 sinh ra, S1 tiêu thụ. Ba mảnh ghép:

**Mảnh 1 — tham số học được `latent_queries` + token riêng.**
📄 `internvla_n1_arch.py` dòng 125 tạo `self.latent_queries = nn.Parameter(torch.randn(1, n_query, hidden_size))`;
📄 `internvla_n1.py` dòng 18 khai token id dành riêng cho trajectory:

```python
TRAJ_TOKEN_INDEX = 151667      # token "latent trajectory" — cầu nối S2 → S1
IMAGE_TOKEN_INDEX = 151655     # <|image_pad|> của Qwen2.5-VL
```

(Trong checkpoint, tensor này chính là `model.language_model.latent_queries` từng hiện nguyên hình
trong log `UNEXPECTED` — PL-B3.)

**Mảnh 2 — S2 sinh latent: `generate_latents`.** Chỉ chạy ở **nhánh pixel-goal** của `s2_step`
(mục 3.2 bước 4). 📄 `internvla_n1.py`, dòng 320–347 (rút gọn):

```python
def generate_latents(self, input_ids, pixel_values, image_grid_thw):
    text_embeds = self.get_model().embed_tokens(input_ids)
    latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
    # nối N_QUERY token TRAJ vào cuối chuỗi, nhét latent_queries vào vị trí đó
    input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY])...], dim=1)
    ...
    text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
    outputs = self.model(inputs_embeds=text_embeds, ..., output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]   # lấy đúng N_QUERY vector cuối
    return hidden_states                                          # ← ĐÂY là latent đưa cho S1
```

Diễn giải cho người mới: sau khi VLM đã "đọc" ảnh + lệnh + tự sinh câu trả lời, ta nối thêm vài
token đặc biệt (số lượng = `n_query` trong config) vào cuối, chạy thêm một lượt forward, rồi lấy
hidden state tại các vị trí đó làm "bản kế hoạch nén" — đó là `output_latent`.

**Mảnh 3 — trao tay ở agent rồi S1 tiêu thụ.** `s2_step` gắn latent vào `S2Output.output_latent`
(policy dòng 191–192); agent chép sang state chung (agent dòng 197–202, kèm `rgb_memory`/`depth_memory`);
main thread kiểm tra `if self.s2_output.output_latent is not None:` (agent dòng 304) rồi gọi
`s1_step_latent(rgbs, depths, latent)` → bên trong, `generate_traj(traj_latents=latent, ...)` đưa
latent vào NavDP làm điều kiện khử nhiễu, sinh quỹ đạo (mục 4).

**Điều kiện sống còn:** latent **chỉ tồn tại khi S2 đi nhánh pixel-goal**. Nếu S2 trả action thuần
(như lần chạy thật PL-E1) thì `output_latent = None` → S1 không bao giờ được gọi — agent khi đó chỉ
phát lại chuỗi action của S2 (agent dòng 280–299).

---

## 6. System 2 và System 1 chạy bất đồng bộ ra sao?

> Bằng chứng đầy đủ số dòng cho toàn mục: PL-C4.

### 6.1. Phân vai thread

- **S2 chạy trong thread riêng** (daemon), khởi động ngay khi tạo agent —
  📄 `internvla_n1_agent.py` dòng 76 (`self._start_s2_thread()`) và dòng 133–208:

```python
def _start_s2_thread(self):
    def s2_thread_func():
        while True:
            should_infer = self.s2_input.should_infer      # chờ cờ hiệu
            if should_infer:
                with self.s2_input_lock:
                    self.s2_input.should_infer = False
                    s2_infer_idx = self.s2_input.idx
            else:
                time.sleep(0.5)                            # chưa cần thì ngủ
                continue
            ...
            with self.s2_agent_lock:
                current_s2_output = self.policy.s2_step(...)   # dòng 159–166
            ...
            with self.s2_output_lock:                      # ghi kết quả dưới lock
                self.s2_output.output_pixel  = current_s2_output.output_pixel
                self.s2_output.output_action = current_s2_output.output_action
                self.s2_output.output_latent = current_s2_output.output_latent
                self.s2_output.rgb_memory    = self.s2_input.rgb
                self.s2_output.depth_memory  = self.s2_input.depth
                self.s2_output.is_infering   = False

    self.s2_thread = threading.Thread(target=s2_thread_func)
    self.s2_thread.daemon = True
    self.s2_thread.start()
```

- **S1 chạy trong main thread** — comment nguyên văn dòng 269: `# S1 inference is done in the main thread`.
- Ba lock (`s2_input_lock`, `s2_output_lock`, `s2_agent_lock`, dòng 71–73) bảo vệ dữ liệu chung
  giữa hai thread.

### 6.2. Hai chế độ — quyết định bởi `should_infer_s2(mode)`

📄 `internvla_n1_agent.py`, dòng 210–241. Docstring nguyên văn trong code:

```python
def should_infer_s2(self, mode="partial_async"):
    """Function: Enables the sys2 inference thread depending on the mode.
    mode: just support 2 modes: "sync" and "partial_async".
    "sync": Synchronous mode (navdp_version >= 0.0), Sys1 and Sys2 execute in a sequential inference chain.
    "partial_async": Asynchronous mode (navdp_version > 0.0, e.g., 0.1),
                     Sys2 performs a single inference, while Sys1 performs multiple inference cycles.
    """
```

| | `sync` | `partial_async` |
|---|---|---|
| Nhịp S2 | mỗi khi hết action tồn (chuỗi tuần tự S2→S1) | **thưa** — chỉ khi một trong 2 điều kiện dưới |
| Nhịp S1 | 1 lần / 1 output S2 | **nhiều vòng** trên cùng 1 latent |
| Điều kiện gọi lại S2 | `output_action is None` | `dual_forward_step >= sys2_max_forward_step` (mặc định **8** — dòng 37) HOẶC cả ba output đều `None` (dòng 230–240) |

Diễn giải: ở `partial_async`, S2 (chậm, ~VLM 7B) chấm một pixel-goal rồi "ngủ"; S1 (nhanh) dùng
latent đó lái tiếp nhiều bước (đếm bằng `dual_forward_step`); đủ 8 bước — hoặc S1/S2 hết sạch
output — thì S2 mới được đánh thức để chấm goal mới. Mode đọc từ config `infer_mode`
(dòng 36, mặc định `'sync'`).

### 6.3. Trình tự một lần `agent.step(obs)` (ghép tất cả lại)

```
step(obs)                                        # dòng 243
 ├─ rgb/depth/instruction lấy từ obs; pose = ma trận đơn vị (dòng 250 — PL-C5)
 ├─ should_infer_s2(mode)?                       # dòng 253
 │    ├─ CÓ  → ghi s2_input + set cờ should_infer=True (dòng 255–263) → thread S2 tự nhặt việc
 │    └─ KHÔNG → policy.step_no_infer(rgb, ...)  # dòng 268 — vẫn nạp ảnh vào history cho đúng
 ├─ CHỜ: while s2_output.is_infering: sleep(0.5) # dòng 270–274 — điểm "hẹn" giữa 2 thread
 ├─ Nếu s2_output.output_action ≠ None:          # dòng 280 — nhánh action thuần
 │    └─ phát action đầu tiên, giữ phần còn lại; action 5 (↓) → bật look_down cho frame sau
 └─ Ngược lại nếu output_latent ≠ None:          # dòng 304 — nhánh dual-system thật sự
      ├─ tiền xử lý cặp ảnh 224×224 (mục 4.2)
      ├─ s1_step_latent(rgbs, depths, latent) → S1Output.idx (tối đa 4 action)
      └─ phát action đầu, action thừa gửi lại vào s2_output.output_action;
         tăng dual_forward_step / sys1_infer_times (dòng 345–364)
```

> Lưu ý phiên bản: cơ chế trên là của `InternVLAN1Agent` (bản NavDP, agent chính trong repo).
> Biến thể **DualVLN** dùng agent khác — `InternVLAN1AsyncAgent` trong
> `internnav/agent/internvla_n1_agent_realworld.py` (dòng 26+, cũng load
> `InternVLAN1ForCausalLM.from_pretrained(args.model_path, ...)` và cũng pin 1 device) — kiến trúc S1
> của nó là NextDiT, không phải NavDP. Xem [04_checkpoint_details](04_checkpoint_details.md) mục 2.2.

---

## 7. Tóm tắt 1 trang (in ra dán màn hình)

| Câu hỏi | Trả lời ngắn | File:dòng |
|---|---|---|
| Checkpoint load ở đâu? | `InternVLAN1ForCausalLM.from_pretrained(model_path)` trong `InternVLAN1Net.__init__` — một lệnh cho cả S1+S2 | `internvla_n1_policy.py:33–38` |
| S1 vào model bằng cách nào? | `InternVLAN1MetaModel.__init__` dựng `self.navdp` **nếu** config có `system1` chứa `navdp`+`async` | `internvla_n1_arch.py:121–145` |
| Đường dẫn model đọc từ đâu? | Key `"model_path"` trong `model_settings` của file config eval → `AgentCfg` → `ModelCfg` → policy | `habitat_dual_system_cfg.py:9` → `internvla_n1_agent.py:33–42` → `internvla_n1_policy.py:31,34` |
| Đưa dữ liệu vào S2? | `policy.s2_step(rgb, depth, pose, instruction, intrinsic, look_down)` → `S2Output{output_pixel, output_latent, output_action}` | `internvla_n1_policy.py:110`; gọi tại `internvla_n1_agent.py:159–166` |
| Đưa dữ liệu vào S1? | `policy.s1_step_latent(rgbs, depths, latent)` → `generate_traj` → `S1Output.idx` | `internvla_n1_policy.py:200–215`; tiền xử lý tại `internvla_n1_agent.py:304–336` |
| S2 nối S1 bằng gì? | `output_latent` — sinh bởi `generate_latents` (token 151667 + `latent_queries`), chỉ có ở nhánh pixel-goal | `internvla_n1.py:18,320–347`; `internvla_n1_policy.py:184–192` |
| Bất đồng bộ ra sao? | S2 = thread daemon riêng + cờ `should_infer`; S1 = main thread; mode `partial_async`: 1 lần S2 / tối đa 8 bước S1 (`sys2_max_forward_step`) | `internvla_n1_agent.py:133–208, 210–241, 269+` |
