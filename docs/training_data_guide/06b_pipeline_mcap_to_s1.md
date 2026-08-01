# 06b — Pipeline: từ file `.mcap` → data train **System 1** (`vln_n1`)

> **File này để làm gì:** hướng dẫn biến log robot `.mcap` thành dataset mà `NavDP_Base_Datset` nạp
> được. Bản song song của [06_pipeline_mcap_to_s2](06_pipeline_mcap_to_s2.md), nhưng bài toán **khác
> hẳn**: không cần câu lệnh, đổi lại cần **quỹ đạo dày, đều** và một **bản đồ vật cản**.
>
> ⚠️ **Khác [06] một điểm về mức độ tin cậy:** tài liệu 06 đi kèm hai script đã chạy thật. File này
> **chưa có script kèm theo** — các đoạn code dưới đây là **bản triển khai tham chiếu** để bạn ghép
> thành `mcap2s1.py`. Bù lại, **mọi ràng buộc và mọi con số về data gốc đều là đo thật** trên scene
> `vln_n1/traij_data/3dfront_d435i/00154c06-…897` và đã được đối chiếu từng dòng với loader.
>
> Bộ tài liệu: [05_data_train_s1](05_data_train_s1.md) (hợp đồng dữ liệu) ·
> [03b_code_train_s1](03b_code_train_s1.md) (loader đọc gì) · [06](06_pipeline_mcap_to_s2.md) (bản S2)

---

## 0. Trước khi làm: S1 dễ hơn hay khó hơn S2?

### 0.1. Bảng so sánh hai pipeline

| | **→ data S2** ([06](06_pipeline_mcap_to_s2.md)) | **→ data S1** (file này) |
|---|---|---|
| Câu lệnh tiếng Anh | ✅ **bắt buộc** — người phải viết | ❌ **không cần một chữ nào** |
| Ảnh RGB | ✅ 2 góc (thẳng + cúi) | ✅ **1 góc** duy nhất |
| Depth | file phải tồn tại, **giá trị không quan trọng** | ✅ **giá trị được dùng thật** — vào thẳng encoder |
| Đơn vị depth | mét × **1000** | mét × **10000** |
| Intrinsics `K` | ✅ để chiếu pixel-goal | ✅ nhưng **chỉ ảnh hưởng 1 trong 3 kiểu đích** |
| Pose | ✅ rời rạc hoá thành `{0,1,2,3}` | ✅ **giữ nguyên liên tục**, đòi hỏi chính xác cao hơn |
| Bản đồ vật cản | ❌ | ✅ **cần**, nếu không critic chết ([05](05_data_train_s1.md) mục 5.3) |
| Nhãn phải chế biến | nhiều (sub-goal, chiếu pixel, rời rạc hoá) | **ít** — loader tự sinh gần hết |
| Chỗ dễ hỏng nhất | chọn sub-goal & phép chiếu | **mật độ / độ mượt của quỹ đạo** (mục 4.4) |

👉 **Kết luận:** S1 **ít việc chú thích hơn** nhưng **khắt khe hơn về chất lượng pose và bản đồ**.
Nếu robot của bạn có SLAM tốt (RTAB-Map, Cartographer, Nav2) thì S1 lại **dễ hơn** S2 — vì mọi thứ
cần đều đã có sẵn trong stack điều hướng.

### 0.2. Một `.mcap` "đủ dùng" cho S1 phải có 5 luồng

| Luồng | Sinh ra cái gì | Bắt buộc? |
|---|---|---|
| **RGB** | `videos/…/observation.images.rgb/*.jpg` | ✅ |
| **Depth** (đã căn theo RGB) | `videos/…/observation.images.depth/*.png` uint16 | ✅ **giá trị thật** |
| **`camera_info`** | `observation.camera_intrinsic` | ✅ |
| **Pose / TF / odom** | `action` (ma trận 4×4 mỗi frame) | ✅ **chất lượng cao** |
| **Bản đồ occupancy** (`/map`) hoặc LiDAR | `meta/pointcloud.ply` | ✅ về mặt chức năng |

Ba thứ **không cần**: câu lệnh, ảnh góc thứ hai, marker episode (cắt episode theo lượt chạy là đủ).

> 💡 **Depth phải "aligned to color".** Loader ghép ảnh RGB và ảnh depth **theo cùng chỉ số**, giả
> định chúng cùng khung nhìn. Với RealSense phải dùng
> `/camera/aligned_depth_to_color/image_raw`, **không** dùng `/camera/depth/image_rect_raw`.

---

## 1. Toàn cảnh đường ống

```
        log robot thật (ROS 2 → .mcap)
                   │
                   ▼   PHASE 0: khảo sát
        tools/mcap_inspect.py --mcap run_01.mcap --deep
                   │
                   ▼   mcap2s1.py  (bạn ghép từ mục 3→8)
   ┌──────────────────────────────────────────────────────────────────┐
   │ A. read_mcap       đọc rgb / depth / caminfo / pose / map        │
   │ B. sync_frames     đồng bộ thời gian + cắt episode               │
   │ C. resample        LẤY MẪU LẠI THEO QUÃNG ĐƯỜNG  ← khác S2 nhất  │
   │ D. make_labels     action 4×4 · camera_extrinsic · intrinsic     │
   │ E. write_images    .jpg + .png uint16 (×10000), tên ĐỆM 3 SỐ     │
   │ F. write_lerobot   parquet (16 số phẳng) + episodes_stats.jsonl  │
   │ G. write_pointcloud  occupancy 2D → điểm màu (0,0,128)           │
   │ H. self_check      mô phỏng loader + gọi loader thật             │
   └──────────────────────────────────────────────────────────────────┘
                   │
                   ▼
    <root_dir>/<group>/<scene>/{meta,data,videos}
                   │
                   ▼  trỏ il.root_dir → start_train.sh --model navdp
```

---

## 2. Phase 0 — Khảo sát file `.mcap`

Dùng công cụ đã có sẵn trong bộ tài liệu:

```bash
python tools/mcap_inspect.py --mcap run_01.mcap --deep
```

Điền cho xong bảng này **trước khi viết một dòng code nào**:

| Câu hỏi | Trả lời của bạn |
|---|---|
| Topic RGB, encoding, độ phân giải, tần số | |
| Topic depth — đã aligned chưa? encoding (`16UC1`/`32FC1`)? đơn vị? | |
| Topic `camera_info` — `K` có khác 0 không? | |
| Nguồn pose: `/odom`, `/tf`, hay `/amcl_pose`? Trôi (drift) bao nhiêu? | |
| **Chiều cao camera so với sàn** (m) | |
| **Góc cúi camera** (độ) — xem cảnh báo mục 4.3 | |
| Có `/map` (`nav_msgs/OccupancyGrid`) không? | |
| Robot có **xoay tại chỗ** không? (mục 4.5) | |

> 🚨 **Câu hỏi quan trọng nhất là "góc cúi".** Nếu camera bị chúc xuống, xem mục 4.3 trước khi làm
> tiếp — nó quyết định bạn có phải rectify ảnh hay không.

---

## 3. Giai đoạn A + B — Đọc và đồng bộ

Phần này **giống hệt** [06 giai đoạn A/B](06_pipeline_mcap_to_s2.md#giai-đoạn-a--read_mcap-đọc-log),
chỉ đổi danh sách topic. Nguyên tắc không đổi:

- **Luồng RGB là nhịp chính.** Với mỗi ảnh RGB tại `t`, tìm depth và pose **gần `t` nhất** bằng tìm
  nhị phân; lệch quá `--tol-ms` (mặc định 60 ms) → **bỏ frame**.
- **Cắt episode** theo lượt chạy (một lần bấm "go" = một episode). Không có tín hiệu nào thì cắt
  theo khoảng lặng dài giữa các đoạn di chuyển.

**Một khác biệt quan trọng so với S2:** ở S2 ta *lọc keyframe* để bỏ frame đứng yên. Ở S1 **đừng lọc
kiểu đó** — thay vào đó lấy mẫu lại theo quãng đường (giai đoạn C), vì S1 cần quỹ đạo **đều về không
gian**, không phải "đủ khác nhau".

```python
@dataclass
class Frame:
    t_ns: int
    rgb: bytes          # payload JPEG gốc, ghi thẳng không giải nén lại
    depth_m: np.ndarray # float32 (H,W) — đơn vị MÉT, đã chuẩn hoá từ encoding gốc
    x: float; y: float; psi: float    # pose 2D: vị trí + góc hướng (quy ước ROS, 0 = +X)
```

Chuẩn hoá depth về **mét** ngay tại đây, đừng để lẫn đơn vị:

```python
if encoding == "16UC1":   depth_m = raw.astype(np.float32) / 1000.0   # RealSense: mm
elif encoding == "32FC1": depth_m = raw.astype(np.float32)            # đã là mét
```

---

## 4. Giai đoạn C — **Lấy mẫu lại theo quãng đường** (bước quan trọng nhất)

### 4.1. Vì sao bắt buộc phải có bước này

Nhãn của S1 là **24 hiệu số vị trí, lấy cách nhau 4 frame, rồi nhân 4**
([03b](03b_code_train_s1.md) mục 6.5). `pred_digit = 4` và `predict_size = 24` **hard-code trong
loader**, nên **"một frame" phải tương ứng một quãng đường cố định** — nếu không, cùng một model sẽ
thấy hai thang đo khác nhau.

Số đo thật trên data gốc:

| Đại lượng | Giá trị đo được |
|---|---|
| Bước đi mỗi frame | trung vị **0.0386 m** (min 0.0285 – max 0.0411) |
| `fps` | 30 |
| Tốc độ tương ứng | ≈ 1.1 m/s |
| Cửa sổ nhãn | 24 × 4 = **96 frame** ≈ 3.2 s ≈ **3.5 m** |
| Biên độ nhãn đo thật | max **0.65 / 0.44 / 0.73** cho `(x, y, θ)` |

### 4.2. Hai ràng buộc cứng, suy ra bằng số học

`DDPMScheduler(clip_sample=True)` kẹp mẫu sinh ra về `[-1, 1]`
([navdp_policy:119](../../../code/internnav/model/basemodel/navdp/navdp_policy.py#L119)) → **nhãn
vượt `[-1,1]` là nhãn model không bao giờ sinh lại được.**

| Ràng buộc | Suy ra từ | Ngưỡng |
|---|---|---|
| **Quãng đường mỗi frame** | `4 × (4 frame × d) ≤ 1` | `d ≤ 0.0625 m` → **chọn `d = 0.037 m`** như data gốc |
| **Bán kính cua** | `4 × Δθ_4frame ≤ 1` → `Δθ ≤ 0.25 rad` trên cung `4d = 0.148 m` | **R ≥ 0.6 m** |

→ Robot cua gấp hơn bán kính 0.6 m sẽ tạo nhãn `θ` vượt ngưỡng. Xem cách xử lý ở mục 4.5.

### 4.3. 🚨 Góc cúi camera: phải bằng 0

Đo thật trên cả 78 frame của scene mẫu: cột 2 của `action` **luôn đúng bằng** `(0, 0, 1)` — tức
camera **không cúi, không nghiêng**, và độ cao `z = 0.35698 m` **không đổi**.

Đây không phải chi tiết vụn: hàm `process_pixel_goal`
([dataset:224-266](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L224)) nhân điểm đích
với `camera_extrinsic[:3,:3]` rồi chiếu bằng công thức pinhole **giả định ma trận đó đúng bằng dạng
chuẩn ở mục 5.2**. Nhét một góc cúi vào đó thì phép chiếu ra sai.

| Camera của bạn | Làm gì |
|---|---|
| Gắn ngang (pitch ≈ 0) | ✅ Không phải làm gì |
| Chúc xuống vài độ | Chấp nhận sai số nhỏ ở `pixel_goal`; hai kiểu đích kia (`point_goal`, `image_goal`) **không bị ảnh hưởng** |
| Chúc xuống nhiều (≥ 15°) | **Rectify ảnh về camera ảo nằm ngang** trước khi ghi, hoặc chấp nhận nhánh `pixel_goal` học sai |

### 4.4. Code lấy mẫu lại

```python
def resample_by_arclength(frames, step_m=0.037):
    """Nội suy lại chuỗi frame sao cho MỖI frame cách nhau đúng step_m mét."""
    xy  = np.array([[f.x, f.y] for f in frames])
    psi = np.unwrap(np.array([f.psi for f in frames]))       # gỡ nhảy ±π trước khi nội suy
    t   = np.array([f.t_ns for f in frames], dtype=np.float64)

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s   = np.concatenate([[0.0], np.cumsum(seg)])            # quãng đường tích luỹ
    if s[-1] < 4 * step_m:
        return []                                            # đoạn quá ngắn, bỏ

    grid = np.arange(0.0, s[-1], step_m)
    out = []
    for u in grid:
        j = int(np.searchsorted(s, u))                       # frame nguồn gần nhất → lấy ẢNH
        j = min(max(j, 0), len(frames) - 1)
        out.append(Sample(
            src   = j,                                       # ảnh dùng lại, KHÔNG nội suy ảnh
            x     = float(np.interp(u, s, xy[:, 0])),
            y     = float(np.interp(u, s, xy[:, 1])),
            psi   = float(np.interp(u, s, psi)),
            t_ns  = int(np.interp(u, s, t)),
        ))
    return out
```

> 📌 **Chỉ nội suy pose, không nội suy ảnh.** Ảnh lấy nguyên frame nguồn gần nhất. Vì `step_m` được
> chọn xấp xỉ bước đi thật của robot, độ lệch ảnh–pose nhỏ hơn một frame gốc.
>
> 📌 **Hệ quả về tần số log.** Muốn `0.037 m/frame` mà không phải bịa thêm frame, hãy log ảnh ở tần
> số sao cho `tốc độ / fps ≈ 0.037`: 1.1 m/s → **30 fps**; 0.5 m/s → **14 fps**; 0.3 m/s → **8 fps**.
> Log thưa hơn mức đó thì bước lấy mẫu lại sẽ **lặp lại cùng một ảnh nhiều lần** — vẫn chạy được,
> nhưng nhãn "đi tiếp" mà ảnh không đổi là tín hiệu nhiễu.

### 4.5. Xử lý đoạn **xoay tại chỗ**

`xyz_to_xyt` ([dataset:312](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L312)) tính `θ`
là **góc của vector di chuyển**. Robot đứng yên xoay → vector di chuyển ≈ 0 → `arctan2` của nhiễu →
**`θ` là rác**. Thêm nữa, lấy mẫu theo quãng đường sẽ **nuốt gọn** đoạn xoay tại chỗ thành một điểm,
tạo ra một cú nhảy hướng lớn giữa hai mẫu liền kề → vượt ngưỡng `Δθ ≤ 0.25 rad`.

Ba cách xử lý, theo thứ tự ưu tiên:

| Cách | Làm sao | Đánh đổi |
|---|---|---|
| **A. Lái theo cung** (khuyến nghị) | Cho robot vừa tiến vừa quay, bán kính ≥ 0.6 m | Phải đổi cách thu data |
| **B. Cắt episode tại chỗ xoay** | Phát hiện `|Δψ| > 20°` mà `Δp < 0.02 m` → kết thúc episode, mở episode mới sau khi xoay xong | Episode ngắn hơn; loại bỏ dữ liệu quay |
| **C. Giữ nguyên** | Không làm gì | Nhãn `θ` sai ở các đoạn rẽ — **không khuyến nghị** |

```python
def split_on_spin(frames, turn_deg=20.0, move_m=0.02):
    """Cắt chuỗi frame tại các đoạn xoay tại chỗ. Trả về danh sách episode."""
    eps, cur = [], [frames[0]]
    for a, b in zip(frames[:-1], frames[1:]):
        dp   = math.hypot(b.x - a.x, b.y - a.y)
        dpsi = abs((b.psi - a.psi + math.pi) % (2 * math.pi) - math.pi)
        if dpsi > math.radians(turn_deg) and dp < move_m:
            if len(cur) >= 4: eps.append(cur)
            cur = []
        cur.append(b)
    if len(cur) >= 4: eps.append(cur)
    return eps
```

> ⚠️ Episode **phải có ≥ 4 frame** sau khi lấy mẫu lại, nếu không `np.random.randint` trong
> `__getitem__` sẽ ném `ValueError` **giữa lúc train** ([03b](03b_code_train_s1.md) mục 5.4).
> Thực tế nên đặt ngưỡng cao hơn nhiều: **≥ 96 frame (3.5 m)** thì nhãn mới không bị đệm 0 ở đuôi.

---

## 5. Giai đoạn D — Sinh ba cột số

Đây là toàn bộ nội dung của parquet. Chỉ có **4 cột**, và 3 trong số đó là hình học.

### 5.1. `action` — ma trận 4×4 mỗi frame

Dựng thẳng từ **hướng đi**, không đi qua "yaw" để khỏi nhầm gốc góc:

```python
def action_matrix(x, y, psi, cam_h):
    """psi = góc hướng theo quy ước ROS (0 = trục +X của world). cam_h = độ cao camera (m)."""
    f = np.array([math.cos(psi), math.sin(psi), 0.0])   # TRƯỚC
    u = np.array([0.0, 0.0, 1.0])                       # LÊN
    r = np.cross(f, u)                                  # PHẢI  = f × u
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = r          # cột 0 = phải
    T[:3, 1] = u          # cột 1 = lên
    T[:3, 2] = -f         # cột 2 = LÙI   ← camera nhìn theo −z
    T[:3, 3] = (x, y, cam_h)
    return T
```

**Kiểm chứng:** đặt `psi = 100.56°` (hướng đi đo được ở frame 0 của scene mẫu) thì công thức cho ra
đúng ma trận trong parquet gốc:

```
cột 0 = ( 0.9831,  0.1833, 0)      ✓ khớp
cột 1 = ( 0,       0,      1)      ✓ khớp
cột 2 = ( 0.1833, -0.9831, 0)      ✓ khớp
```

> 🚨 **Cột 2 chỉ ra SAU.** Đây là chỗ dễ sai nhất và sai thì **không có lỗi nào báo ra** — model chỉ
> đơn giản học một quy ước lộn ngược. Chi tiết cách đo ra kết luận này:
> [03b](03b_code_train_s1.md) mục 6.1.

### 5.2. `observation.camera_extrinsic` — **hằng số**, không phải pose

Đây **không** phải "pose của frame đầu". Nó là **ma trận hiệu chuẩn lắp camera** mà `relative_pose`
dùng để gỡ bỏ phép xoay do lắp đặt ([03b](03b_code_train_s1.md) mục 6.2). Giá trị **bắt buộc**:

```python
E = np.eye(4, dtype=np.float32)
E[:3, :3] = [[1, 0,  0],
             [0, 0, -1],
             [0, 1,  0]]
E[:3, 3]  = (0.0, 0.0, cam_h)     # ← CHỈ độ cao camera có ý nghĩa
```

Tức chính là `action_matrix(0, 0, π/2, cam_h)`: "đứng ở gốc, nhìn theo trục +Y của world".
**Giống hệt nhau ở mọi frame, mọi episode.** Đo thật trên scene mẫu: khớp từng chữ số, với
`cam_h = 0.35698`.

> 🔑 `E[2,3]` (độ cao camera) **được dùng thật**: `process_pixel_goal` đặt điểm đích ở độ cao
> `0.8 × E[2,3]` so với camera để chiếu xuống ảnh
> ([dataset:233](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L233)). Ghi sai độ cao →
> chấm đích rơi sai chỗ trên ảnh.

### 5.3. `observation.camera_intrinsic` — 9 số phẳng

Lấy từ `camera_info.K`, ứng với **độ phân giải ảnh bạn ghi ra**. Nếu bạn resize ảnh trước khi lưu
thì phải **scale `K` theo cùng tỉ lệ**:

```python
K_scaled = K.copy()
K_scaled[0, :] *= (W_new / W_old)     # fx, cx
K_scaled[1, :] *= (H_new / H_old)     # fy, cy
```

Đo thật trên scene mẫu (`480×270`, camera D435i mô phỏng):

```
K = [[355.815,   0,     240],
     [  0,     351.687, 135],
     [  0,       0,       1]]        → cx = W/2, cy = H/2 đúng tâm ảnh
```

> 📌 Loader chỉ đọc **hàng đầu tiên** của cột này (`.tolist()[0]`) → `K` được coi là **không đổi
> suốt episode**. Ghi lặp lại cùng giá trị cho mọi hàng.

---

## 6. Giai đoạn E — Ghi ảnh

### 6.1. Cây thư mục phải tạo

```
<root_dir>/                                 ← trỏ bởi il.root_dir
└── <group>/                                ← BẮT BUỘC có tầng này (vd: myrobot_d435i)
    └── <scene_id>/
        ├── data/chunk-000/episode_000000.parquet
        ├── meta/episodes_stats.jsonl
        ├── meta/pointcloud.ply
        └── videos/chunk-000/
            ├── observation.images.rgb/   episode_000000_000.jpg
            └── observation.images.depth/ episode_000000_000.png
```

> 🚨 **Phải có đủ hai tầng `<group>/<scene>`.** Loader lặp `os.listdir(root)` rồi lặp tiếp
> `os.listdir(root/group)` ([dataset:75-81](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L75)).
> Đặt scene thẳng dưới `root_dir` → nó sẽ tưởng `data/`, `meta/` là tên scene → crash hoặc bỏ qua.

> 🚨 **Chỉ dùng một chunk.** `chunk_name = os.listdir(...)[0]`
> ([dataset:82](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L82)) chỉ lấy **một** thư
> mục chunk và `os.listdir` **không đảm bảo thứ tự**. Nhiều chunk = mất data im lặng.

### 6.2. Quy ước đặt tên — **bắt buộc đệm 0**

```
episode_{ep_index:06d}_{frame_index:03d}.jpg     ← ảnh RGB, frame_index đếm TỪ 0 TRONG EPISODE
episode_{ep_index:06d}_{frame_index:03d}.png     ← ảnh depth, cùng tên
```

Loader lấy ảnh bằng `sorted(os.listdir(rgb_dir))` rồi **cắt theo chỉ số**, nên thứ tự chữ cái phải
trùng thứ tự thời gian ([05](05_data_train_s1.md) mục 2.1).

> ⚠️ **3 chữ số chỉ đủ cho 1000 frame/episode.** Với `step_m = 0.037`, 1000 frame = 37 m. Episode
> dài hơn → dùng `:04d` **cho toàn bộ dataset** (nhất quán là được, không được trộn lẫn).

### 6.3. Ghi depth: nhân **10000**

```python
def save_depth_png(depth_m: np.ndarray, path: str):
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d, 0.0, 6.5535)               # 6.5535 m = trần của uint16 sau khi ×10000
    Image.fromarray((d * 10000.0).astype(np.uint16)).save(path)   # mode 'I;16'
```

Loader chia `10000`, rồi **đặt về 0** mọi giá trị `> 5 m` và `< 0.1 m`
([dataset:178-192](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L178)). Nghĩa là **dải
hữu ích là `[1000, 50000]`** trong ảnh uint16.

| Sai lầm | Hậu quả |
|---|---|
| Nhân 1000 (theo thói quen S2) | Mọi giá trị chia 10000 ra ≤ 0.65 m → **gần như toàn bộ bị đặt về 0** → depth trắng trơn |
| Lưu float32 PNG | `np.array(img, np.uint16)` diễn giải sai byte |
| Lưu depth chưa aligned | Depth lệch khỏi RGB → model học tương ứng sai |

Kiểm nhanh sau khi ghi: `min/max` của ảnh phải nằm quanh **1000–50000**, không phải 0–255 hay 0–6.

### 6.4. Ghi RGB

Nếu message đã là JPEG nén thì **ghi thẳng byte gốc** (nhanh hơn, không mất chất lượng). Nếu là
`sensor_msgs/Image` thô thì dựng numpy theo `encoding` (`rgb8`/`bgr8`) rồi `Image.save`.
Độ phân giải: **giữ nguyên bản gốc** — loader tự resize + đệm về 224×224
([dataset:164-176](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L164)).

---

## 7. Giai đoạn F — Parquet + `meta/`

### 7.1. Parquet: 16 số **phẳng**, không lồng

```python
import pyarrow as pa, pyarrow.parquet as pq

def write_parquet(path, K, E, actions):        # actions: list các ma trận 4×4
    n = len(actions)
    table = pa.table(
        {
            "index": pa.array(range(n), type=pa.int64()),
            "observation.camera_intrinsic": pa.array([K.reshape(-1).tolist()] * n,
                                                     type=pa.list_(pa.float32())),
            "observation.camera_extrinsic": pa.array([E.reshape(-1).tolist()] * n,
                                                     type=pa.list_(pa.float32())),
            "action": pa.array([A.reshape(-1).tolist() for A in actions],
                               type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, path)
```

> 🚨 **Phẳng, không lồng.** `vln_ce` lưu `pose` dạng `list<list<float>>` 4×4; `vln_n1` lưu **16 số
> một hàng**. Loader gọi `np.stack(frame).reshape(-1,4,4)`
> ([dataset:201](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L201)) — đưa dạng lồng vào
> sẽ ra shape sai. Bê nguyên code ghi parquet từ [06](06_pipeline_mcap_to_s2.md) sang là hỏng.

`vln_n1` **không có** 5 trường chuẩn LeRobot (`timestamp`, `frame_index`, `episode_index`,
`task_index`, `index` kiểu LeRobot) — đừng thêm vào cho "đúng chuẩn", loader không cần.

### 7.2. `episodes_stats.jsonl` — file **quyết định ảnh nào thuộc episode nào**

```python
def write_episodes_stats(path, episode_lengths):
    cursor, lines = 0, []
    for ep_i, n in enumerate(episode_lengths):
        lines.append(json.dumps({
            "episode_index": ep_i,
            "task_index":  {"min": ep_i, "max": ep_i, "count": 1},
            "image_index": {"min": cursor, "max": cursor + n - 1, "count": n},
        }))
        cursor += n                                   # ← TÍCH LUỸ, KHÔNG reset
    Path(path).write_text("\n".join(lines) + "\n")
```

🔑 **`image_index` là chỉ số TOÀN CỤC** trong `sorted(os.listdir(rgb_dir))` của cả scene, không phải
chỉ số trong episode. Đo thật trên scene mẫu:

```
{"episode_index": 0, "task_index": {...}, "image_index": {"min": 0,   "max": 77,  "count": 78}}
{"episode_index": 1, "task_index": {...}, "image_index": {"min": 78,  "max": 182, "count": 105}}
{"episode_index": 2, "task_index": {...}, "image_index": {"min": 183, "max": 305, "count": 123}}
```

Ba điều kiện phải đồng thời đúng:

1. `count` = số ảnh thật của episode = số hàng parquet;
2. các khoảng `[min, max]` **liền kề, không chồng, không hở**, phủ kín toàn bộ thư mục ảnh;
3. **thứ tự dòng** trong `episodes_stats.jsonl` khớp thứ tự `sorted()` của thư mục parquet — loader
   ghép bằng `data_paths[episode_idx]` với `episode_idx` là **số thứ tự dòng**
   ([dataset:108](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L108)), **không** đọc
   trường `episode_index`.

Số ảnh depth phải **bằng đúng** số ảnh RGB, cùng thứ tự sort.

### 7.3. Ba file `meta/` còn lại

`episodes.jsonl`, `tasks.jsonl`, `info.json` — **loader S1 không đọc**
([05](05_data_train_s1.md) mục 7.1). Ghi ra để dataset tự mô tả thì tốt, bỏ qua cũng không sao.

---

## 8. Giai đoạn G — `pointcloud.ply` từ bản đồ occupancy

### 8.1. Loader cần chính xác cái gì

```python
color_distance = np.abs(scene_color - np.array([0, 0, 0.5])).sum(axis=-1)
select_index   = np.where(color_distance < 0.05)[0]        # dataset:208-209
```

Open3D đọc màu `uchar` rồi chia 255, nên **`(0, 0, 128)` → `(0, 0, 0.50196)` → khoảng cách 0.002 →
được chọn** ✓. Sau đó **chỉ `x, y` được dùng** (`[:, 0:2]`, [dataset:473]) → **toạ độ z không ảnh
hưởng gì**.

Đo thật trên scene mẫu:

| Chỉ số | Giá trị |
|---|---|
| Tổng điểm | 88 750 (PLY nhị phân, `double x,y,z` + `uchar rgb`) |
| Điểm vật cản `(0,0,128)` | **27 276** |
| `z` của điểm vật cản | rải **ngẫu nhiên** trong `[−0.100, 0.100]` |
| Khoảng cách điểm gần nhất (xy) | trung vị **0.036 m** |
| Điểm nền `(102,102,102)` | 49 014 — **không được loader dùng** |

→ Kết luận: **thứ bạn cần chỉ là một bản đồ occupancy 2D độ phân giải ~3–5 cm**, đúc thành điểm.
Chọn 5 cm là hợp lý vì ngưỡng phạt của critic là khoảng cách **L1** 0.1 m.

### 8.2. Từ `nav_msgs/OccupancyGrid` → điểm vật cản

```python
def occupancy_to_obstacle_xy(grid_data, width, height, resolution, origin_xy, thresh=50):
    """grid_data: mảng int8 (-1 = chưa biết, 0..100 = xác suất chiếm chỗ)."""
    g = np.asarray(grid_data, dtype=np.int16).reshape(height, width)
    ys, xs = np.nonzero(g >= thresh)                       # -1 (chưa biết) tự động bị loại
    x = origin_xy[0] + (xs + 0.5) * resolution
    y = origin_xy[1] + (ys + 0.5) * resolution
    return np.stack([x, y], axis=1)
```

> ⚠️ Bản đồ phải **cùng hệ toạ độ với pose** đã ghi vào `action`. Nếu pose lấy từ `/odom` mà bản đồ ở
> khung `map`, phải áp phép biến đổi `map → odom` trước. Sai khung = critic phạt nhầm chỗ, **không
> có lỗi nào báo ra**.

### 8.3. Ghi file `.ply` (không cần Open3D)

```python
def write_pointcloud_ply(path, obstacle_xy, free_xy=None, slab=0.1, layers=3, seed=0):
    rng = np.random.RandomState(seed)
    pts, cols = [], []

    for _ in range(layers):                                 # đúc mỏng quanh mặt sàn
        z = rng.uniform(-slab, slab, size=len(obstacle_xy))
        pts.append(np.column_stack([obstacle_xy, z]))
        cols.append(np.tile(np.array([0, 0, 128], np.uint8), (len(obstacle_xy), 1)))

    if free_xy is not None:                                 # tuỳ chọn, chỉ để nhìn cho dễ
        pts.append(np.column_stack([free_xy, np.zeros(len(free_xy))]))
        cols.append(np.tile(np.array([102, 102, 102], np.uint8), (len(free_xy), 1)))

    P = np.concatenate(pts).astype(np.float64)
    C = np.concatenate(cols).astype(np.uint8)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(P)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    rec = np.empty(len(P), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                                           ("r", "u1"), ("g", "u1"), ("b", "u1")]))
    rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
    rec["r"], rec["g"], rec["b"] = C[:, 0], C[:, 1], C[:, 2]
    with open(path, "wb") as f:
        f.write(header); f.write(rec.tobytes())
```

Kiểm ngay sau khi ghi:

```python
import open3d as o3d, numpy as np
pcd = o3d.io.read_point_cloud(path)
c = np.asarray(pcd.colors)
n_obs = int((np.abs(c - np.array([0, 0, 0.5])).sum(-1) < 0.05).sum())
print("điểm vật cản loader sẽ thấy:", n_obs)      # PHẢI > 0
```

> 🚨 `n_obs == 0` là kiểu hỏng **im lặng và nguy hiểm nhất** của cả pipeline: train vẫn chạy, loss
> vẫn giảm, nhưng `critic` là hằng số `2.0` → model **không học được gì về né vật cản**
> ([05](05_data_train_s1.md) mục 5.3).

### 8.4. Nếu **không có** bản đồ

| Cách | Ưu | Nhược |
|---|---|---|
| Dựng occupancy từ **depth + pose** (chiếu điểm depth trong dải cao 0.1–1.5 m xuống lưới 2D, đếm) | không cần thêm cảm biến | nhiễu, chỉ phủ vùng camera đã thấy |
| Dùng **costmap của Nav2 / bản đồ LiDAR** đã có | chính xác nhất | phải có stack điều hướng |
| **Bỏ hẳn** `pointcloud.ply` | rẻ | ⚠️ Không được: `load_pointcloud` sẽ lỗi lúc đọc file. Muốn bỏ critic thì phải ghi file có điểm nhưng **không có** màu vật cản — và chấp nhận mất tín hiệu critic |

---

## 9. Giai đoạn H — Kiểm định

### 9.1. Tự kiểm (không cần dependency nặng)

```python
def self_check(scene_dir, cam_h):
    stats = [json.loads(l) for l in open(f"{scene_dir}/meta/episodes_stats.jsonl")]
    rgb   = sorted(os.listdir(f"{scene_dir}/videos/chunk-000/observation.images.rgb"))
    depth = sorted(os.listdir(f"{scene_dir}/videos/chunk-000/observation.images.depth"))
    parq  = sorted(os.listdir(f"{scene_dir}/data/chunk-000"))

    assert len(rgb) == len(depth),  "số ảnh rgb ≠ depth"
    assert len(stats) == len(parq), "số dòng episodes_stats ≠ số parquet"
    assert stats[0]["image_index"]["min"] == 0
    assert stats[-1]["image_index"]["max"] == len(rgb) - 1, "image_index không phủ kín thư mục ảnh"
    for a, b in zip(stats[:-1], stats[1:]):                 # liền kề, không hở
        assert b["image_index"]["min"] == a["image_index"]["max"] + 1

    for ep_i, (st, pq_name) in enumerate(zip(stats, parq)):
        df = pq.read_table(f"{scene_dir}/data/chunk-000/{pq_name}").to_pandas()
        n  = st["image_index"]["count"]
        assert len(df) == n, f"ep{ep_i}: {len(df)} hàng parquet ≠ {n} ảnh"
        assert n >= 4, f"ep{ep_i}: episode quá ngắn"
        A = np.array([np.array(a).reshape(4, 4) for a in df["action"]])
        assert np.allclose(A[:, 2, 3], cam_h, atol=1e-3), "độ cao camera không hằng số"
        assert np.allclose(A[:, :3, 1], [0, 0, 1], atol=1e-3), "cột 1 phải là (0,0,1)"
        d = np.linalg.norm(np.diff(A[:, :3, 3], axis=0), axis=1)
        assert d.max() <= 0.0625, f"ep{ep_i}: bước {d.max():.4f} m > 0.0625 → nhãn vượt [-1,1]"
```

### 9.2. Mô phỏng đúng phép tính nhãn của loader

Đây là kiểm tra **có giá trị nhất** — nó bắt được lỗi thang đo mà mọi kiểm tra khác bỏ sót:

```python
R_MOUNT_INV = np.linalg.inv(np.array([[1,0,0],[0,0,-1],[0,1,0]], float))

def label_amplitude(A, start, end):
    """Tính pred_actions đúng như loader, trả về biên độ lớn nhất."""
    Rb = A[start][:3, :3] @ R_MOUNT_INV
    H  = np.eye(4); H[:3, :3] = Rb; H[:3, 3] = A[start][:3, 3]
    Hi = np.linalg.inv(H)
    loc = []
    for i in range(start, end + 1):
        t = (Hi @ np.array([*A[i][:3, 3], 1.0]))[:3]
        loc.append([t[1], -t[0], t[2]])                    # đổi trục: x trước, y trái
    loc = np.array(loc)
    iv  = loc[1] - loc[0]
    xyt = np.array([[loc[i][0], loc[i][1],
                     np.arctan2(np.cross(iv[:2], (loc[i+1]-loc[i])[:2]),
                                np.dot(iv[:2],  (loc[i+1]-loc[i])[:2]))]
                    for i in range(len(loc) - 1)])
    idx = np.clip(np.arange(25) * 4, 0, len(loc) - 2)
    pa  = (xyt[idx][1:] - xyt[idx][:-1]) * 4.0             # (24, 3)
    return np.abs(pa).max(axis=0)

# chạy trên mọi episode:  amp = label_amplitude(A, 0, len(A)-1)
# ✅ đạt nếu amp < 1.0 ở cả 3 cột.  Data gốc đo được: [0.648, 0.441, 0.727]
```

### 9.3. Gọi **loader thật** — phán quyết cuối cùng

```python
from internnav.dataset.navdp_lerobot_dataset import NavDP_Base_Datset

ds = NavDP_Base_Datset(
    "./traj_data",              # root_dirs   (bên trong phải có <group>/<scene>/)
    "./navdp_cache.json",       # preload_path — thư mục cha phải TỒN TẠI
    8, 24, 4, 224, 1.0,         # memory_size, predict_size, batch_size, image_size, scene_scale
    pixel_channel=4, preload=False,
)
print("episode:", len(ds) // 50)                 # nhớ chia 50 (nhân bản, xem 03b mục 5.3)

pg, ig, tg, rgb, depth, act, aug, c_pred, c_aug, flag = ds[0]
print(rgb.shape, depth.shape, act.shape, tg.shape, ig.shape, pg.shape)
# kỳ vọng: (8,224,224,3) (224,224,1) (24,3) (224,224,4) (224,224,6) (3,)
print("critic:", float(c_pred), float(c_aug), " pixel nhìn thấy:", flag)
assert act.abs().max() <= 1.0, "nhãn vượt [-1,1] — xem mục 4.2"
assert float(c_pred) != 2.0 or float(c_aug) != 2.0, "critic là hằng số → pointcloud không có vật cản"
```

Cần `pip install open3d jsonlines`. Chạy `ds[0]` vài chục lần: kết quả **phải khác nhau** mỗi lần
(bốc cửa sổ ngẫu nhiên) — nếu giống hệt nhau là episode quá ngắn.

### 9.4. Nhìn bằng mắt

Vẽ `pixel_goal[:, :, 3]` (kênh mặt nạ) đè lên `pixel_goal[:, :, 0:3]`: chấm trắng **phải nằm trên
sàn, về phía robot sắp đi tới**. Code mẫu có sẵn ngay trong loader
([dataset:589-643](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L589) — khối
`if __name__ == "__main__"`), chỉ cần đổi đường dẫn.

---

## 10. Đăng ký & train

Khác S2, **không có `data_dict` nào phải sửa** — loader quét thẳng thư mục. Chỉ cần trỏ đường dẫn
trong [scripts/train/base_train/configs/navdp.py](../../../code/scripts/train/base_train/configs/navdp.py):

```python
il=IlCfg(
    root_dir='data/datasets/my_navdp/traj_data',      # ← thư mục chứa <group>/<scene>/
    dataset_navdp='data/datasets/my_navdp_cache.json',# ← thư mục cha phải tồn tại
    preload=False,                                    # lần đầu; các lần sau đặt True cho nhanh
    ckpt_to_load='checkpoints/navdp/navdp.ckpt',      # fine-tune; để '' = train từ đầu
    batch_size=32, epochs=5, lr=2e-5,
)
```

```bash
bash scripts/train/base_train/start_train.sh --name navdp_myrobot --model navdp
```

Cách chạy thử tối thiểu trên 1 GPU và các bẫy lúc train: [03b](03b_code_train_s1.md) mục 13.

---

## 11. Bảng rủi ro & cách phát hiện

| Rủi ro | Hậu quả | Phát hiện sớm bằng |
|---|---|---|
| Depth nhân 1000 thay vì 10000 | Depth gần như toàn 0 | in `min/max` ảnh: phải trong `[1000, 50000]` |
| Depth chưa aligned với RGB | Model học tương ứng sai, **không báo lỗi** | chồng depth lên RGB xem mép vật có trùng |
| Tên ảnh không đệm 0 | `sorted()` sai thứ tự → ảnh lệch episode | `self_check` §9.1 + nhìn bằng mắt |
| `image_index` tính theo từng episode | Ảnh lấy hẳn từ episode khác | assert "liền kề, phủ kín" ở §9.1 |
| Thứ tự dòng `episodes_stats` ≠ thứ tự parquet | Ghép quỹ đạo với ảnh của episode khác | so `count` với số hàng parquet từng episode |
| Bước đi > 0.0625 m/frame | Nhãn vượt `[-1,1]`, model không sinh lại được | §9.2 (`amp < 1.0`) |
| Cua gấp hơn R = 0.6 m / xoay tại chỗ | Cột `θ` vượt ngưỡng, hoặc `θ` là rác | §9.2 cột thứ 3 |
| Cột 2 của `action` chỉ ra trước (thiếu dấu −) | Model học quy ước lộn ngược | assert `dot(hướng đi, −cột2) ≈ +1` |
| `camera_extrinsic` ghi pose frame đầu | `relative_pose` gỡ sai phép xoay lắp đặt | assert bằng đúng ma trận §5.2 |
| Độ cao camera sai | `pixel_goal` chấm sai chỗ | nhìn bằng mắt §9.4 |
| `pointcloud.ply` không có màu `(0,0,128)` | `critic = 2.0` hằng số, **train vẫn chạy** | đếm điểm sau bộ lọc §8.3 |
| Bản đồ khác khung toạ độ với pose | Critic phạt nhầm chỗ | vẽ quỹ đạo đè lên bản đồ |
| Thiếu tầng `<group>/` | Loader hiểu nhầm `data/`, `meta/` là scene | `len(ds)` = 0 hoặc crash |
| Nhiều hơn 1 chunk | Mất data im lặng | đếm episode ở §9.3 |
| Episode < 4 frame | `ValueError` **giữa lúc train** | assert `n >= 4` ở §9.1 |
| Episode < 96 frame | Đuôi nhãn toàn 0 → model học dừng sớm | thống kê độ dài episode |
| Thư mục cha của `dataset_navdp` chưa có | Quét xong hết mới `FileNotFoundError` | `mkdir -p` trước |

---

## 12. Lộ trình theo phase

| Phase | Việc | Ước tính | Xong khi… |
|---|---|---|---|
| **0. Khảo sát** | `mcap_inspect.py`; điền bảng mục 2; đo chiều cao & góc cúi camera | 0.5 ngày | biết đủ 5 luồng có hay không |
| **1. Đọc + đồng bộ** | Ánh xạ topic; chuẩn hoá depth về mét; cắt episode | 1 ngày | in ra số frame mỗi episode hợp lý |
| **2. Lấy mẫu lại** | `resample_by_arclength` + `split_on_spin`; kiểm §9.2 | 1 ngày | `amp < 1.0` ở cả 3 cột, mọi episode |
| **3. Ghi ảnh + parquet** | Đúng tên, đúng dtype, đúng `episodes_stats` | 1 ngày | `self_check` §9.1 sạch |
| **4. Bản đồ vật cản** | Occupancy → `pointcloud.ply`; kiểm số điểm | 0.5–2 ngày | `n_obs > 0` và vẽ đè lên quỹ đạo thấy khớp |
| **5. Kiểm định** | Gọi loader thật §9.3 + nhìn bằng mắt §9.4 | 0.5 ngày | shape đúng, `critic ≠ 2.0` |
| **6. Train thử** | 1 GPU, vài chục step | 0.5 ngày | loss hữu hạn và giảm |
| **7. Mở rộng** | Nhiều lượt chạy → nhiều scene, cùng `<group>` | tuỳ data | đủ lượng để fine-tune |

---

## 13. Ghi chú kỹ thuật

- **Nhiều lượt chạy → nhiều scene.** Mỗi lượt là một `<scene_id>` riêng, **kèm bản đồ riêng**. Chỉ
  dùng chung `pointcloud.ply` khi hiện trường không đổi ([02](02_he_thong.md) mục 8.4).
- **Nhiều địa điểm → nhiều `<group>`.** Đặt tên theo `<nơi>_<model camera>` giống data gốc
  (`3dfront_d435i`) để sau này lọc cho dễ.
- **Đừng trộn `vln_ce` và `vln_n1`.** Hai bộ khác quy ước trục (OpenCV vs OpenGL), khác đơn vị depth
  (1000 vs 10000), khác cách đặt tên ảnh (có đệm 0 vs không). Bê code từ
  [06](06_pipeline_mcap_to_s2.md) sang phải sửa cả ba chỗ.
- **`fps` không được loader đọc**, nhưng nên ghi đúng vào `info.json` để người sau còn hiểu.
- **Nếu chỉ có RGB, không có depth:** chạy DepthAnythingV2 (chế độ metric) sinh depth rồi nhân
  10000. Khác S2 ở chỗ **giá trị depth ở đây được dùng thật**, nên depth ước lượng sẽ **hạ chất
  lượng model**, không phải chỗ giữ chỗ vô hại.
- **Trước khi tự tạo data, cân nhắc lại lời khuyên ở [05](05_data_train_s1.md) mục 9:** dùng
  checkpoint NavDP có sẵn và dồn công sức vào data System 2. Data S1 chỉ đáng làm khi robot của bạn
  có động lực học / hình dạng khác hẳn tập pretrain, hoặc bạn đã có sẵn stack SLAM xuất ra đủ 5 luồng.

---

*Quay lại mục lục: [00_README](00_README.md). Code phía train: [03b_code_train_s1](03b_code_train_s1.md).*
