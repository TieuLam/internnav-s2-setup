# Data format của System 1 (NavDP)

Tài liệu này mô tả format input/output của System 1, ý nghĩa của trajectory và
cách trajectory được chuyển thành action.

> **Trạng thái xác minh**
>
> - NavDP standalone point-goal: **đã chạy thật** trên sample
>   `InternData-N1-preview/vln_n1`.
> - NavDP joint trong dual-system: **đã đối chiếu code**, chưa chạy
>   end-to-end với checkpoint InternVLA-N1 7B.

## 1. Hai chế độ System 1

System 1 có hai contract khác nhau:

| Chế độ | Goal đưa vào S1 | Dữ liệu quan sát |
|---|---|---|
| NavDP standalone | Point-goal `(x,y,theta)` | 8 RGB history + 1 current depth |
| NavDP joint trong dual-system | Learned latent do S2 sinh | Cặp RGB-D tại plan-frame và current-frame |

Không được đưa pixel coordinate `[row,col]` trực tiếp vào API point-goal
standalone. Trong dual-system, pixel goal được S2 dùng để tạo latent và xác định
frame neo cho S1.

---

## 2. Input của NavDP standalone

API đã kiểm chứng:

```python
NavDPNet.predict_pointgoal_batch_action_vel(
    goal_point,
    input_images,
    input_depths,
    ...
)
```

### 2.1. Tensor contract

| Input | Shape | Dtype/range | Ý nghĩa |
|---|---:|---|---|
| `goal_point` | `[B,3]` | `float32` | `(x,y,theta)` trong robot-local frame |
| `input_images` | `[B,8,224,224,3]` | `float32`, `[0,1]` | Tám RGB history frame, channels-last |
| `input_depths` | `[B,224,224,1]` | `float32`, mét | Một depth frame hiện tại |

Trong đó:

- `B` là batch size.
- Khi infer một sample: `B=1`.
- RGB dùng layout `NHWC`, không phải `NCHW`.
- RGB được ImageNet-normalize bên trong backbone.
- Các history frame còn thiếu ở đầu episode được pad bằng zero.
- Standalone chỉ nhận **một current-depth frame**. Không lặp depth thành tám
  frame.

### 2.2. Hệ tọa độ point-goal

Robot bắt đầu tại:

```text
(x,y,theta) = (0,0,0)
```

Quy ước loader hiện tại:

- `x > 0`: phía trước robot.
- `y < 0`: phía bên phải robot.
- `y > 0`: phía bên trái robot.
- `theta`: heading tương đối, đơn vị radian.
- `x,y`: đơn vị mét.

Ví dụ:

```python
goal_point = torch.tensor(
    [[2.0, -0.5, -0.26]],
    dtype=torch.float32,
    device="cuda:0",
)
```

Ý nghĩa:

- goal ở khoảng `2.0 m` phía trước;
- lệch `0.5 m` sang phải;
- heading đích lệch khoảng `-0.26 rad ≈ -15°`.

### 2.3. RGB history

Ví dụ shape:

```python
input_images.shape
# torch.Size([1, 8, 224, 224, 3])
```

Thứ tự thời gian:

```text
input_images[0,0] = frame cũ nhất
...
input_images[0,7] = frame gần nhất
```

Giá trị pixel:

```text
float32 trong [0,1]
```

Backbone thực hiện ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### 2.4. Current depth

Ví dụ shape:

```python
input_depths.shape
# torch.Size([1, 224, 224, 1])
```

Depth biểu diễn khoảng cách theo mét.

Trong loader `vln_n1` đã kiểm chứng:

```text
raw uint16 → chia 10000 → depth mét
```

Sau đó:

- depth nhỏ hơn `0.1 m` trở thành `0`;
- depth lớn hơn `5 m` trở thành `0`;
- vùng invalid cũng trở thành `0`.

> Phép chia `10000` thuộc đường `vln_n1` đã kiểm chứng. Không áp dụng mù cho
> depth của dataset khác. Ví dụ depth PNG `vln_ce` có thể dùng đơn vị milimét
> và cần chia `1000`.

### 2.5. Ví dụ input hoàn chỉnh

Đây là ví dụ minh họa format, không phải script load checkpoint đầy đủ:

```python
import torch

B = 1

goal_point = torch.tensor(
    [[2.0, -0.5, -0.26]],
    dtype=torch.float32,
    device="cuda:0",
)

input_images = torch.rand(
    B, 8, 224, 224, 3,
    dtype=torch.float32,
    device="cuda:0",
)

input_depths = torch.rand(
    B, 224, 224, 1,
    dtype=torch.float32,
    device="cuda:0",
) * 5.0

assert goal_point.shape == (1,3)
assert input_images.shape == (1,8,224,224,3)
assert input_depths.shape == (1,224,224,1)
```

---

## 3. Output của NavDP standalone

Output đã kiểm chứng gồm hai nhóm trajectory:

| Output | Shape | Ý nghĩa |
|---|---:|---|
| Negative trajectories | `[K,24,3]` | Candidate có critic score thấp |
| Positive trajectories | `[K,24,3]` | Candidate có critic score cao |

Trong lần chạy real-data:

```text
negative trajectories: [2,24,3]
positive trajectories: [2,24,3]
```

`K=2` vì smoke test yêu cầu hai diffusion sample. API này trả tối đa khoảng
tám trajectory ở mỗi nhóm, tùy batch size và số sample.

### 3.1. Ý nghĩa `[K,24,3]`

```text
K  = số trajectory candidate
24 = số future waypoint của mỗi trajectory
3  = (x,y,yaw) của từng waypoint
```

Mỗi waypoint là pose tương đối trong robot-local frame:

```text
trajectory[k,t] = (x_t, y_t, yaw_t)
```

Trong đó:

- `x_t,y_t` là vị trí tích lũy so với điểm bắt đầu;
- `yaw_t` là hướng dự đoán tại waypoint;
- đây không phải pixel coordinate;
- đây không phải motor velocity;
- một waypoint không đồng nghĩa với một simulator action.

### 3.2. Ví dụ trajectory

Ví dụ rút gọn:

```python
trajectory = [
    [0.08, -0.01, -0.02],
    [0.17, -0.02, -0.03],
    [0.28, -0.05, -0.05],
    [0.42, -0.10, -0.08],
    # ...
    [1.92, -0.47, -0.25],
]
```

Diễn giải:

| Waypoint | Ý nghĩa |
|---|---|
| `[0.08,-0.01,-0.02]` | Cách start 8 cm phía trước, 1 cm bên phải |
| `[0.17,-0.02,-0.03]` | Cách start 17 cm phía trước, 2 cm bên phải |
| `[1.92,-0.47,-0.25]` | Gần goal: 1.92 m phía trước, 0.47 m bên phải, yaw khoảng -14.3° |

Đây là cumulative trajectory:

```text
start = (0,0,0)
waypoint 1 = pose tương đối so với start
waypoint 2 = pose tương đối so với start
...
waypoint 24 = pose cuối tương đối so với start
```

### 3.3. Positive/negative không phải dấu tọa độ

`positive trajectory` không có nghĩa tọa độ dương.  
`negative trajectory` không có nghĩa tọa độ âm.

Quy trình:

1. Diffusion policy sinh nhiều candidate.
2. Critic chấm mức phù hợp với goal và observation.
3. Candidate score cao được xếp vào positive set.
4. Candidate score thấp được xếp vào negative set.

---

## 4. Delta trajectory và cumulative trajectory

Model diffusion nội bộ có thể sinh action delta:

```text
(dx_1,dy_1,dyaw_1)
(dx_2,dy_2,dyaw_2)
...
```

Sau unnormalize, vị trí cumulative được dựng bằng tổng tích lũy:

```text
x_t = dx_1 + dx_2 + ... + dx_t
y_t = dy_1 + dy_2 + ... + dy_t
```

Ví dụ:

```text
delta XY:
  [0.10,-0.01]
  [0.12,-0.02]
  [0.11,-0.03]

cumulative XY:
  [0.10,-0.01]
  [0.22,-0.03]
  [0.33,-0.06]
```

Các output trajectory dùng để hiển thị/follow là đường cumulative.

---

## 5. Chuyển trajectory thành action

Trajectory là kế hoạch hình học liên tục. Environment Habitat thường cần action
rời rạc.

Action vocabulary:

| ID | Ý nghĩa |
|---:|---|
| `0` | STOP |
| `1` | Tiến `0.25 m` |
| `2` | Quay trái `15°` |
| `3` | Quay phải `15°` |
| `5` | Nhìn xuống `30°` |
| `-1` | Wait/no-op sentinel của wrapper |

`traj_to_actions()`:

1. Chia XY delta cho `4` để unnormalize.
2. Cumulative sum thành các candidate XY path.
3. Lấy mean path giữa các candidate.
4. Dùng follower với lookahead bốn waypoint.
5. Quay trái/phải để hướng về target.
6. Tiến từng bước `0.25 m`.
7. Dừng khi cách endpoint không quá `0.2 m`, hoặc bước tiếp theo làm robot xa
   endpoint hơn.

Ví dụ:

```python
trajectory = [
    [0.25, -0.02, -0.05],
    [0.48, -0.10, -0.15],
    [0.69, -0.23, -0.26],
    [0.86, -0.42, -0.40],
]
```

Có thể được follower chuyển thành:

```python
actions = [3,3,1,1]
```

Ý nghĩa:

```text
quay phải 15°
quay phải 15°
tiến 0.25 m
tiến 0.25 m
```

Một trajectory 24 waypoint không bắt buộc tạo đúng 24 action.

---

## 6. Input của S1 trong dual-system

Dual-system không gọi standalone point-goal API. S2 tạo learned latent plan và
truyền nó vào NavDP joint.

### 6.1. Tensor contract

| Input | Shape | Ý nghĩa |
|---|---:|---|
| `latent` | `[B,N,3584]` | Learned plan do custom Qwen2.5-VL/System 2 sinh |
| `rgbs` | `[B,2,224,224,3]` | RGB plan-frame và RGB current-frame |
| `depths` | `[B,2,224,224,1]` | Depth plan-frame và depth current-frame |

Hai frame có thứ tự:

```text
index 0 = frame lúc S2 sinh pixel goal/latent
index 1 = observation hiện tại
```

Ví dụ:

```python
rgbs.shape
# [1,2,224,224,3]

depths.shape
# [1,2,224,224,1]

latent.shape
# dự kiến [1,N,3584]
```

Code training dùng default `N=4`, nên shape dự kiến thường là:

```text
[1,4,3584]
```

Tuy nhiên `N` phải được in từ checkpoint InternVLA-N1 7B thật trước khi dùng
làm contract chính thức.

### 6.2. Ý nghĩa learned latent

Latent tổng hợp:

- instruction;
- các RGB history frame được S2 chọn;
- RGB hiện tại;
- text/pixel-goal output của S2;
- learned trajectory-query hidden states.

Có thể hiểu:

```text
latent = biểu diễn tensor của kế hoạch semantic “đi tới đâu”
```

Cặp RGB-D cung cấp:

```text
observation = thông tin “robot đang nhìn thấy gì và đã di chuyển tới đâu”
```

S1 kết hợp plan và observation để sinh local trajectory.

### 6.3. Lời gọi policy

```python
s1_output = policy.s1_step_latent(
    rgbs,
    depths,
    s2_output.output_latent,
)
```

Pixel goal không được truyền như tensor point-goal `[B,3]` trong lời gọi này.
Nó được dùng để xác định frame neo và tham gia quá trình tạo latent bên S2.

---

## 7. Output S1 trong dual-system

Raw joint model sinh nhiều trajectory candidate:

```text
[num_candidates,num_future_steps,3]
```

Default code hiện tại thường tương ứng:

```text
[32,32,3]
```

Shape chính xác vẫn phụ thuộc checkpoint/config 7B thật.

Policy chuyển raw trajectory thành action:

```python
S1Output(
    idx=[action_1,action_2,action_3,action_4]
)
```

S1 chỉ expose tối đa bốn action đầu của plan mới.

Ví dụ:

```python
S1Output(idx=[2,1,1,3])
```

Ý nghĩa:

```text
quay trái → tiến → tiến → quay phải
```

Agent:

1. trả action đầu ngay;
2. cache các action còn lại;
3. phát cached action ở các environment step tiếp theo;
4. khi cần, chạy lại S1 với cùng latent nhưng current RGB-D mới;
5. refresh S2 khi đạt budget `sys2_max_forward_step`.

---

## 8. So sánh nhanh

| | S1 standalone | S1 trong dual-system |
|---|---|---|
| Goal input | `(x,y,theta)` | Learned latent từ S2 |
| Goal shape | `[B,3]` | `[B,N,3584]` |
| RGB | `[B,8,224,224,3]` | `[B,2,224,224,3]` |
| Depth | `[B,224,224,1]` | `[B,2,224,224,1]` |
| Raw output | Ranked trajectories `[K,24,3]` | Candidate trajectories, thường `[32,32,3]` |
| Output agent dùng | Trajectory được critic xếp hạng | Tối đa 4 discrete action |
| Runtime status | ✅ Đã chạy thật | ⬜ Chờ checkpoint 7B |

---

## 9. Kết quả real-data đã kiểm chứng

Sample:

```text
InternData-N1-preview/
preview/vln_n1/traj_data/matterport3d_d435i/
1LXtFkjw3qL/trajectory_2
```

Kết quả:

| Trường | Giá trị |
|---|---|
| `goal_point` | `[1,3]`, `float32`, finite |
| RGB history | `[1,8,224,224,3]`, range `[0,0.82745]` |
| Current depth | `[1,224,224,1]`, range `[0,4.99821] m` |
| Negative trajectories | `[2,24,3]`, finite |
| Positive trajectories | `[2,24,3]`, finite |
| Latency | `1.5908 s` |
| Peak CUDA allocation | `575.5 MiB` |

Machine-readable evidence:

```text
MH_week2/logs/navdp_real_sample.json
```

Script tái lập:

```text
MH_week2/scripts/run_navdp_real_sample.py
```

## 10. Các lỗi format thường gặp

- Đưa RGB dạng `[B,T,3,H,W]` thay vì `[B,T,H,W,3]`.
- Đưa RGB `uint8 [0,255]` thay vì `float32 [0,1]`.
- Lặp depth thành tám frame trong standalone.
- Dùng depth raw nhưng không đổi đúng đơn vị.
- Hiểu `(x,y,theta)` là world coordinate thay vì robot-local.
- Hiểu trajectory là delta trong khi output đang được đọc như cumulative path.
- Hiểu mỗi waypoint là một action.
- Truyền pixel coordinate `[row,col]` vào point-goal `(x,y,theta)`.
- Coi positive/negative trajectory là dấu của tọa độ.
- Khẳng định latent `[1,4,3584]` trước khi đọc checkpoint 7B thật.

## 11. Tài liệu liên quan

- [System 1 I/O contract](io_system1.md)
- [Phân tích data → latent → S1 → action](system1_data_to_action_deep_dive.md)
- [Cấu trúc code dual-system](02_code_structure.md)
- [Data contract](03_data_contract.md)
