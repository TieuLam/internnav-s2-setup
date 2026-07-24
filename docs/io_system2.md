# I/O System 2 (VLM) — InternVLA-N1

**Mục tiêu tài liệu:** bảng I/O của System 2 (VLM Qwen2.5-VL-7B) — **prompt template**, cách nối
ảnh history, **output parser** (trích pixel goal / action ra sao, xử lý sai format), kèm 1 snippet
chạy được. Ghi lại **số liệu đo từ source code**, không phải từ docs/README.

> **Trạng thái (22/07, Ngày 2 — HOÀN THÀNH):** phần **đọc code** (prompt, parser, generate, config) +
> phần **chạy thật `generate()`** (Khối 2, T4 x2) **đều xong**. 🚨 Kết quả thật: model ra **action
> `←←←←`**, **chưa** bật chế độ pixel-goal — chi tiết + nghi phạm ở **mục 4.a**.

**Nguồn code (đọc trên GitHub `InternRobotics/InternNav@main`, repo chưa clone local):**

| # | File | Cho ta biết |
|---|---|---|
| 1 | `internnav/agent/internvla_n1_agent.py` | agent gọi S2 qua `self.policy.s2_step(...)` |
| 2 | `internnav/model/utils/vln_utils.py` | dataclass `S2Input` / `S2Output` (hợp đồng I/O) |
| 3 | `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` | **prompt template, conjunctions, actions2idx, parser, s2_step** — file lõi |
| 4 | `internnav/model/basemodel/internvla_n1/internvla_n1.py` | `InternVLAN1ForCausalLM`, token index, `generate_latents` |
| 5 | `scripts/eval/configs/habitat_dual_system_cfg.py` | config: `resize_w/h=384`, `num_history=8`, `mode=dual_system` |

---

## 1. Tổng quan — S2 có **hai chế độ output** trong CÙNG một lần gọi

S2 **không** cố định trả một loại output. Trong một lần `s2_step()`, tuỳ text VLM sinh ra mà rẽ nhánh:

| Nhánh | Điều kiện (đọc từ text VLM) | Output chính | Dùng cho |
|---|---|---|---|
| **Pixel goal** | text **có chữ số** (`re.search(r'\d', ...)`) | `output_pixel` `[row, col]` + `output_latent` | feed sang **System 1** (diffusion policy) |
| **Action rời rạc** | text **không** có chữ số | `output_action` (list int) | điều khiển trực tiếp `{↑,←,→,↓}` |
| **STOP** | text chứa `"STOP"` | `output_action = [0]` | kết thúc episode |

🎯 **Điểm mấu chốt cho data contract (khớp `data_contract.md` mục 4.b):** pixel goal `[u,v]` mà S2 sinh
ra chính là cột `goal.{setting}` trong `vln_ce/`. Còn action rời rạc `{1,2,3,5}` chính là cột `action`.
**Cùng một VLM sinh cả hai** — nên `vln_ce` có cả `goal` lẫn `action` trong cùng parquet là hợp lý.

Token đặc biệt trong model (`internvla_n1.py`):
```python
IMAGE_TOKEN_INDEX = 151655     # <|image_pad|> của Qwen2.5-VL
TRAJ_TOKEN_INDEX  = 151667     # token latent trajectory (cầu nối S2 → S1)
```
`InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM)` — tức model
**kế thừa Qwen2.5-VL** rồi bồi thêm phần trajectory/latent. (Khớp phát hiện Ngày 1: load bằng
`Qwen2_5_VLForConditionalGeneration` thuần thì phần bồi thêm `navdp.*`/`latent_queries` bị bỏ qua.)

---

## 2. INPUT — S2 nhận gì

### 2.a. Hợp đồng I/O ở tầng agent (`vln_utils.py`)

```python
@dataclass
class S2Input:
    idx: int = -1
    instruction: str = None       # câu lệnh tiếng Anh tự nhiên (1 câu)
    rgb: np.ndarray = None        # ảnh RGB hiện tại
    depth: np.ndarray = None      # depth (dùng ở bước dựng trajectory, không vào prompt VLM)
    pose: Tuple[float,float,float] = None
    look_down: bool = False       # frame "nhìn xuống" — xử lý riêng (mục 2.d)
    should_infer: bool = False
```

Agent gọi (`internvla_n1_agent.py`), S2 chạy trong **thread riêng** (cơ chế `partial_async` — S2 chạy
định kỳ, không phải mỗi bước):
```python
current_s2_output = self.policy.s2_step(
    self.s2_input.rgb, self.s2_input.depth, self.s2_input.pose,
    self.s2_input.instruction, self.camera_intrinsic, self.s2_input.look_down,
)
```

### 2.b. Prompt template (verbatim từ `internvla_n1_policy.py`)

Conversation **không có system role** — chỉ có cặp `human`/`gpt`:
```python
self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
# answer = "" (rỗng, để model generate)
```

Prompt template gốc (`<instruction>.` bị `.replace()` bằng câu lệnh thật):
```
You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go
next to stay on track? Please output the next waypoint's coordinates in the image. Please output
STOP when you have successfully completed the task.
```

→ Model được **yêu cầu tường minh** trả **toạ độ pixel waypoint**, hoặc **STOP** khi xong. Đây là lý do
parser mặc định coi "có số = pixel goal".

### 2.c. Nối ảnh history vào prompt

Sau câu template, code chèn history rồi câu "you can see \<image\>":
```python
DEFAULT_IMAGE_TOKEN = "<image>"

if self.episode_idx == 0:
    history_id = []                                   # frame đầu: KHÔNG có history
else:
    history_id = np.unique(np.linspace(0, self.episode_idx - 1,
                           self.num_history, dtype=np.int32)).tolist()
    placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
    sources[0]["value"] += f' These are your historical observations: {placeholder}.'

# câu dẫn ảnh hiện tại — luôn dùng conjunction[0]:
prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN     # "you can see <image>"
sources[0]["value"] += f" {prompt}."

self.input_images = [self.rgb_list[i] for i in history_id] + cur_images   # history + frame hiện tại
```

`self.conjunctions` (7 biến thể, nhưng runtime chỉ dùng `[0]`):
```python
self.conjunctions = ['you can see ', 'in front of you is ', 'there is ', 'you can spot ',
                     'you are toward the ', 'ahead of you is ', 'in your sight is ']
```

**Cách lấy history:** `np.linspace(0, episode_idx-1, num_history)` → **hạ mẫu đều** tối đa
`num_history=8` frame quá khứ (không phải 8 frame gần nhất). `np.unique` bỏ trùng khi episode còn ngắn.
Chuỗi ảnh cuối cùng = `history (≤8) + 1 frame hiện tại`.

Text `<image>` (dạng chuỗi) được `split_and_clean(...)` (import từ `vln_utils`) tách thành các entry
`{"type":"image","image":...}` để processor Qwen hiểu — model tự bung thành vision token `151655`.

### 2.d. `look_down` — frame nhìn xuống xử lý riêng

```python
else:  # look_down == True
    self.input_images.append(image)
    input_img_id = -1
    sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
    self.conversation_history.append(
        {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]})
```
Frame look_down **không** vào `rgb_list` (không tính là history), mà **nối tiếp hội thoại** — dùng để
model "cúi nhìn" xác nhận goal sát chân. (Liên hệ `data_contract.md` 4.b: setting camera có góc cúi
`pitch≥15°` mới thấy sàn để chấm pixel goal; `125cm_0deg` nhìn thẳng → goal toàn `-1`.)

### 2.e. Tiền xử lý ảnh & processor

```python
image = Image.fromarray(rgb).convert('RGB')
image = image.resize((self.resize_w, self.resize_h))     # 384 × 384 (config)

text = self.processor.apply_chat_template(
    self.conversation_history, tokenize=False, add_generation_prompt=True)
inputs = self.processor(text=[text], images=self.input_images,
                        return_tensors="pt").to(self.device)
```

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `resize_w` × `resize_h` | **384 × 384** | `habitat_dual_system_cfg.py` |
| `num_history` | **8** | `habitat_dual_system_cfg.py` |
| `mode` | `dual_system` (hoặc `system2`) | config |
| image token | `<image>` (text) → `151655` (`<|image_pad|>`) | policy + `internvla_n1.py` |
| `min_pixels` / `max_pixels` | **không set trong code này** | — (⬜ đo default processor ở Khối 2) |

> ⚠️ RGB gốc `vln_ce` là **640×480** (`data_contract.md` 4.b) nhưng S2 **resize về 384×384** (vuông,
> **méo tỉ lệ 4:3 → 1:1**). Nhóm SIM cần biết: pixel goal S2 sinh nằm trong hệ toạ độ ảnh **sau resize**,
> không phải ảnh gốc. Đây là chỗ dễ lệch toạ độ khi map ngược về ảnh robot.

---

## 3. OUTPUT + PARSER — S2 trả gì và trích ra sao

### 3.a. Generate call (verbatim)

```python
output_ids = self.model.generate(
    **inputs,
    max_new_tokens=128,          # ⚠️ HARD-CODE 128 trong s2_step (config ghi max_new_tokens=1024 — lệch)
    do_sample=False,             # greedy, tất định
    use_cache=True,
    past_key_values=None,
    return_dict_in_generate=True,
).sequences

self.llm_output = self.processor.tokenizer.decode(
    output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)   # bỏ phần prompt
```

### 3.b. Parser — decision logic (verbatim)

```python
if bool(re.search(r'\d', self.llm_output)):          # (1) CÓ CHỮ SỐ → pixel goal
    coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
    pixel_goal = [int(coord[1]), int(coord[0])]      # ⚠️ ĐẢO thứ tự: [số thứ 2, số thứ 1]
    image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
    traj_latents = self.model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)
    output.output_pixel  = np.array(pixel_goal)
    output.output_latent = traj_latents
else:                                                 # (2) KHÔNG có số → action rời rạc
    action_seq = self.parse_actions(self.llm_output)
    output.output_action = action_seq
```

`parse_actions` — khớp ký tự mũi tên / `STOP` (regex ghép từ key của `actions2idx`):
```python
self.actions2idx = OrderedDict({'STOP':[0], "↑":[1], "←":[2], "→":[3], "↓":[5]})

def parse_actions(self, output):
    action_patterns = '|'.join(re.escape(a) for a in self.actions2idx)   # "STOP|↑|←|→|↓"
    matches = re.compile(action_patterns).findall(output)
    actions = [self.actions2idx[m] for m in matches]
    return list(itertools.chain.from_iterable(actions))
```

### 3.c. Bảng tổng hợp OUTPUT

| Trường (`S2Output`) | dtype | Ý nghĩa | Điền ở nhánh |
|---|---|---|---|
| `output_pixel` | `np.ndarray [2]` | pixel goal **`[row, col]`** (đã đảo từ text) | pixel |
| `output_latent` | `torch.Tensor` | latent trajectory (token `151667`) → **feed S1** | pixel |
| `output_action` | `list[int]` | chuỗi action `{0,1,2,3,5}` | action / STOP |
| `output_trajectory` | `np.ndarray` | quỹ đạo (dựng ở bước sau, cần depth) | — |
| `rgb_memory`/`depth_memory` | `np.ndarray` | RGB/depth tại frame chấm goal (cho S1) | pixel |

### 3.d. 🚨 Điểm parser dễ gây bug — phải nêu

1. **Đảo toạ độ:** `pixel_goal = [coord[1], coord[0]]`. VLM nói theo `(x, y)` = `(cột, hàng)`, code lưu
   thành `[hàng, cột]` (= `[v, u]`, thuận index numpy `img[row, col]`). ⚠️ Nhưng `data_contract.md` 4.b
   đo cột `goal.{setting}` là **`[u, v]`** (pixel `[cột, hàng]`).
   ✅ **ĐÃ ĐỐI CHIẾU 23/07/2026 với 98 frame output thật** (PL-E3 trong `handbook/05_appendix.md`,
   checkpoint `-w-NavDP`): VLM nhả `"u v"` trong **không gian 640×480 gốc** — bằng chứng: 20 frame có
   toạ độ vượt 384 (max 563), không thể là không gian 384×384 dù ảnh input bị resize 384. Tức **cùng
   hệ toạ độ với GT**, không lệch quy ước — khi so sánh chỉ cần đảo lại thứ tự (`u=coord[0],
   v=coord[1]`), **KHÔNG scale 384→640** (scale nhầm sẽ thổi L2 từ 41.8 lên 206 px). Kết quả đo:
   median lệch GT chỉ **20 px**.
2. **Parser cực lỏng:** chỉ cần text **có bất kỳ chữ số nào** là coi là pixel goal, rồi lấy **2 số đầu
   tiên** bất kể ngữ cảnh. Nếu VLM viết "*turn 90 degrees*" → "90" bị hiểu nhầm thành toạ độ. Không có
   validation format `(x, y)`.
3. **Không có nhánh xử lý format sai tường minh:** nếu text không số và không chứa ký tự action nào →
   `parse_actions` trả **list rỗng** → agent không có lệnh. Không raise, không fallback.
4. **`max_new_tokens=128` hard-code trong `s2_step`** ≠ `max_new_tokens=1024` trong config → khả năng
   config chỉ áp cho nhánh khác (traj/S1). Ứng viên PR fix docs/consistency.

---

## 4. Snippet chạy được (khung — output thô điền ở Khối 2)

Cho **S2 đơn thuần** (không cần cài repo InternNav, tránh `pip install -e .` hạ cấp `transformers`):
class HF thuần là đủ (phát hiện Ngày 1). Snippet dựng đúng prompt template ở trên:

```python
import torch, re
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

MODEL = "/kaggle/input/internvla-n1-ckpt/..."   # mount tieulam/internvla-n1-ckpt
proc = AutoProcessor.from_pretrained(MODEL)      # KHONG trust_remote_code
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")   # dtype=, KHONG torch_dtype=

INSTRUCTION = "walk forward and turn left at the door"   # 1 cau that tu vln_ce/meta/tasks.jsonl
img = Image.open("frame_rgb.png").convert("RGB").resize((384, 384))   # RGB 640x480 vln_ce -> 384

prompt = ("You are an autonomous navigation assistant. Your task is to " + INSTRUCTION +
          ". Where should you go next to stay on track? Please output the next waypoint's "
          "coordinates in the image. Please output STOP when you have successfully completed "
          "the task. you can see <image>.")

messages = [{"role":"user","content":[{"type":"image","image":img},
                                       {"type":"text","text":prompt}]}]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], images=[img], return_tensors="pt").to(m.device)
out = m.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
llm_output = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(repr(llm_output))

# --- parser (khop internvla_n1_policy.py) ---
if re.search(r'\d', llm_output):
    coord = [int(c) for c in re.findall(r'\d+', llm_output)]
    pixel_goal = [coord[1], coord[0]]        # [row, col]
    print("PIXEL GOAL:", pixel_goal)
else:
    idx = {'STOP':0, "↑":1, "←":2, "→":3, "↓":5}
    print("ACTIONS:", [idx[m] for m in re.findall('STOP|↑|←|→|↓', llm_output)])
```

### 4.a. KẾT QUẢ ĐO THẬT (22/07, Khối 2, T4 x2)

| Lần | Prompt | Ảnh | `llm_output` thô | Nhánh parser | Kết quả |
|---|---|---|---|---|---|
| 1 | tự dựng (chỉ instruction) | 1 RGB `vln_ce` (setting có pitch), **chưa resize 384** | `←←←←` | action (không số) | `[2,2,2,2]` = rẽ trái ×4 |
| 2 | **prompt thật** (xin coordinates) | như trên | `←←←←` | action | `pixel_goal = []` (không có số để parse) |

🚨 **Phát hiện:** dù prompt thật **yêu cầu toạ độ pixel**, model vẫn rơi vào **nhánh action** (`←←←←`) —
không sinh chữ số nên `output_pixel` rỗng. Model **nói đúng bộ từ vựng action** nhưng **chưa bật được chế
độ pixel-goal** trong setup hiện tại. Đây là bằng chứng trực tiếp cho "S2 hai chế độ" ở mục 1: lần chạy
này đi **nhánh action**, không phải pixel.

**Nghi phạm — setup lệch so với `s2_step` thật (mục 2.e), xếp theo khả năng:**
1. **Chưa `resize(384, 384)`** — đưa ảnh 640×480 gốc; policy resize vuông 384 → số vision token lệch, ảnh
   off-distribution. **Thử lại điều này TRƯỚC.**
2. **Thiếu câu dẫn `"you can see <image>."`** (conjunction[0], mục 2.c) nối cuối prompt.
3. Có thể chế độ pixel cần **full policy** (`generate_latents`, token `151667`) — giả thuyết Ngày 4.

→ **Việc tiếp (rẻ, cùng session):** chạy lại **đúng snippet mục 4** (đã có `resize(384,384)` + `you can see
<image>.`); nếu vẫn arrows → thử nhiều ảnh khác + đối chiếu cột `action` của parquet frame đó.

**VRAM generate:** ⬜ chưa đo (`torch.cuda.max_memory_allocated(0/1)`).

---

## 5. Việc còn treo (chuyển tiếp Khối 2 / Ngày 3)

- [x] **Chạy `generate()` thật** (Khối 2, T4 x2) → ra `←←←←`, đi **nhánh action** (chưa bật pixel). Mục 4.a.
- [ ] 🚨 **Chạy lại đúng snippet mục 4** (có `resize(384,384)` + `you can see <image>.`) → xem có bật
      pixel-goal không. **Nghi phạm số 1** (mục 4.a).
- [ ] Kiểm chứng **đảo toạ độ** `[row,col]` (runtime) vs `[u,v]` (data `vln_ce`) — mục 3.d.1. *(Chưa làm
      được — cần có output pixel trước.)*
- [ ] Đối chiếu cột `action` của parquet frame đó với `←` model đoán (ground-truth check).
- [ ] Đo `min_pixels`/`max_pixels` default của `AutoProcessor` (Qwen2.5-VL) — không set trong repo.
- [ ] Đọc `split_and_clean` trong `vln_utils.py` (cách tách `<image>` → content entry) — chi tiết hoá.
- [ ] Xác nhận `max_new_tokens` 128 vs 1024 áp cho nhánh nào (ứng viên PR).
- [ ] Đo VRAM generate (`max_memory_allocated`) — số cho mail server.
- [ ] (Ngày 4 — full policy) Khi load bằng `InternVLAN1ForCausalLM`: **verify `config.system1`** có trong
      `config.json` bundle NavDP + đếm params `navdp.*` > 0 sau load. Nhánh dựng S1 nằm ở
      `internvla_n1_arch.py` (`InternVLAN1MetaModel.__init__`); lời gọi load gốc pin **một** device
      (`device_map={"": device}`) → Kaggle phải override `"auto"`. Chi tiết: `checkpoint_variants.md` 5.b.

---

## 6. Nguồn (file:dòng đã truy)

- Prompt / conjunctions / `actions2idx` / `parse_actions` / parser / `s2_step`:
  `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
- `S2Input` / `S2Output` / `S1Input` / `S1Output`, `split_and_clean`: `internnav/model/utils/vln_utils.py`
- `s2_step` được agent gọi: `internnav/agent/internvla_n1_agent.py`
- Token index `151655`/`151667`, `InternVLAN1ForCausalLM`, `generate_latents`:
  `internnav/model/basemodel/internvla_n1/internvla_n1.py`
- Config `resize=384`, `num_history=8`, `mode=dual_system`, `max_new_tokens=1024`:
  `scripts/eval/configs/habitat_dual_system_cfg.py`
- Chuỗi load checkpoint S1+S2 (điểm load `InternVLAN1Net.__init__` ~dòng 33–40; dựng `self.navdp` +
  `latent_queries` trong `InternVLAN1MetaModel`; `navdp_pretrained` standalone):
  `internvla_n1_policy.py` + `internvla_n1_arch.py` + `navdp.py` — xem `checkpoint_variants.md` 5.b
