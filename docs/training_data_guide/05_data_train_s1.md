# 05 — Cấu trúc data train **System 1** (`vln_n1`) — cái gì **bắt buộc**, cái gì **không**

> **File này để làm gì:** mô tả data mà `NavDP_Base_Datset` đọc để dạy **System 1** (bộ não "phản xạ"
> — vẽ đường đi cong né vật cản), và tách bạch **bắt buộc / không bắt buộc**.
>
> Số liệu **đo thật** trên scene `vln_n1/traij_data/3dfront_d435i/00154c06-2ee2-408a-9664-b8fd74742897`
> (có sẵn trong `InternNav/data/`), đối chiếu code
> [navdp_lerobot_dataset.py](../../../code/internnav/dataset/navdp_lerobot_dataset.py) và config
> [scripts/train/base_train/configs/navdp.py](../../../code/scripts/train/base_train/configs/navdp.py).
> Bộ tài liệu: [04_data_train_s2](04_data_train_s2.md) · [02_he_thong](02_he_thong.md)

---

## 1. System 1 học "trò chơi" gì?

Cho một **dãy ảnh robot vừa đi qua** + **ảnh depth** + **một điểm đích** → **vẽ ra đường đi cong,
mượt, tránh vật cản** cho vài mét phía trước.

Điều đặc biệt: **không ai phải chú thích bằng tay.** Máy tự suy ra đáp án từ hai nguồn:

1. **Quỹ đạo thật của chuyên gia** (dãy pose camera đã ghi) → chính là "đường đúng" để bắt chước.
2. **Bản đồ 3D vật cản** → để chấm điểm "đường này an toàn hay đâm tường" (gọi là **critic**).

→ *Imitation learning* + một "giám khảo" chấm an toàn.

---

## 2. Cây thư mục một scene

```
<root_dir>/<simulator>_<camera>/<scene_uuid>/
├── data/chunk-000/
│   ├── episode_000000.parquet          ← ✅ BẮT BUỘC — mỗi file 1 episode
│   └── … (đo thật: 32 episode)
├── meta/
│   ├── episodes_stats.jsonl            ← ✅ BẮT BUỘC (loader lấy image_index min/max ở đây)
│   ├── pointcloud.ply                  ← ✅ BẮT BUỘC — BẢN ĐỒ VẬT CẢN
│   ├── episodes.jsonl                  ← ⚪ không bắt buộc (loader S1 không đọc)
│   ├── tasks.jsonl                     ← ⚪ không bắt buộc
│   └── info.json                       ← ⚪ không bắt buộc
└── videos/chunk-000/
    ├── observation.images.rgb/         ← ✅ BẮT BUỘC  episode_000000_000.jpg  (4229 file)
    ├── observation.images.depth/       ← ✅ BẮT BUỘC  episode_000000_000.png  (4229 file)
    ├── observation.video.rgb/          ← ⚪ không bắt buộc  episode_000000.mp4 (33 file)
    └── observation.video.depth/        ← ⚪ không bắt buộc
```

- `<simulator>` ∈ `{3dfront, gibson, hm3d, hssd, matterport3d, replica}` — nơi render.
- `<camera>` ∈ `{d435i, zed}` — **model camera thật được mô phỏng**. Mỗi scene render 2 lần.
  → Nếu tự tạo data, camera của bạn nên **giống model thật** thì phân bố ảnh/độ sâu mới khớp.
- 📌 Bộ archive tải về chứa **cả hai dạng ảnh**: thư mục ảnh từng frame (`observation.images.*`) và
  video `.mp4` (`observation.video.*`). **Loader chỉ đọc dạng ảnh từng frame**
  ([dòng 89-97](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L89)) — các file `.mp4` là
  dư thừa, có thể xoá để tiết kiệm ổ đĩa.

### 2.1. 🚨 Tên file ảnh **phải đệm số 0**

Loader lấy danh sách ảnh bằng **`sorted(os.listdir(rgb_dir))`**
([dòng 92](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L92)) rồi **cắt theo chỉ số**
`image_index.min … max`. Vì `sorted()` sắp theo **thứ tự chữ cái**, tên không đệm số 0 sẽ cho thứ tự
sai: `_0, _1, _10, _11, _2, …`

| Bộ data | Kiểu đặt tên | Vì sao khác nhau |
|---|---|---|
| `vln_ce` (S2) | `episode_000000_12.jpg` — **không đệm** | Loader S2 **ghép đường dẫn bằng công thức**, không sort → không cần đệm |
| `vln_n1` (S1) | `episode_000000_000.jpg` — **đệm 3 chữ số** | Loader S1 **sort tên file** → **bắt buộc đệm** |

→ Sao chép nhầm quy ước đặt tên giữa hai hệ là một lỗi im lặng rất khó phát hiện.

---

## 3. Bảng số (`parquet`) — chỉ **4 cột**, và `action` là "linh hồn"

Đo thật (`episode_000000.parquet`, 78 hàng):

| Cột | Kiểu Arrow thật | Hình dạng sau khi reshape | Nghĩa |
|---|---|---|---|
| `index` | `int64` | scalar | thứ tự frame 0,1,2,… |
| `observation.camera_intrinsic` | `list<float>` **phẳng 9 phần tử** | → `(3,3)` | ma trận `K`, **không đổi suốt episode** |
| `observation.camera_extrinsic` | `list<float>` **phẳng 16 phần tử** | → `(4,4)` | pose **gốc** (frame xuất phát) của camera |
| `action` | `list<float>` **phẳng 16 phần tử** | → `(4,4)` mỗi frame | **Pose camera từng frame = quỹ đạo = ĐÁP ÁN của S1** |

> ⚠️ Chú ý: khác `vln_ce` (lưu `pose` dạng **lồng** `list<list<float32>>` 4×4), `vln_n1` lưu **phẳng**
> 16 số. Loader tự reshape ([dòng 198-201](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L198)).

> ⚠️ **Cùng tên "action" nhưng khác hẳn `vln_ce`:** ở đây là **ma trận 4×4 (đường đi liên tục)**, không
> phải số nguyên rời rạc. Xem [02](02_he_thong.md) mục 5.

> ⚠️ `vln_n1` **thiếu** 5 trường chuẩn LeRobot (`timestamp`, `frame_index`, `episode_index`, `index`
> LeRobot, `task_index`) → **LeRobot gốc không đọc được**, chỉ loader riêng của InternNav đọc được.

### 3.1. Số liệu thật (scene `3dfront_d435i`)

```
K = [[355.815,   0.000, 240.000],
     [  0.000, 351.687, 135.000],
     [  0.000,   0.000,   1.000]]        → ảnh 480×270, FOV ≈ 68°×42° = đúng spec RealSense D435i

camera_extrinsic (frame gốc) — vị trí (0, 0, 0.357)      → camera cao 35.7 cm
action[0] — vị trí (-4.724, -6.543, 0.357)
```

**Quy ước trục của `vln_n1` KHÁC `vln_ce`** (giải mã từ ma trận đo thật): cột 1 = "phải", cột 2 =
**"lên"**, cột 3 = "trước" → tức **x-phải, y-LÊN, z-trước**; trong khi `vln_ce` dùng **OpenCV**
(x-phải, y-**xuống**, z-trước). → **Đừng bao giờ bê nguyên pose từ hệ này sang hệ kia.**

---

## 4. Ảnh & depth — đơn vị **khác** System 2 (bẫy!)

| Loại | Định dạng đo thật | Kiểu số | Loader xử lý thế nào |
|---|---|---|---|
| RGB | `.jpg` 480×270 | uint8 | resize giữ tỉ lệ + đệm về **224×224**, chia 255 → `[0,1]` ([dòng 164-176](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L164)) |
| Depth | `.png` 480×270, mode `I;16` | uint16 (đo được 5173 … 65535) | **chia 10000** → mét, rồi **loại** giá trị `>5 m` và `<0.1 m` (đặt về 0) ([dòng 178-192](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L178)) |

> 🚨 **Bẫy hằng số depth:** System 1 chia **10000**, System 2 chia **1000**. Hai hệ **khác nhau 10
> lần**. Tạo data cho hệ nào thì theo đúng hằng số của hệ đó — sai thì model học sai mà **không có
> lỗi nào báo ra**.
>
> (Đo thật: giá trị 65535 là "bão hoà" — chỗ tia không chạm vật gì; sau khi chia 10000 = 6.55 m > 5 m
> nên bị loại về 0.)

---

## 5. `pointcloud.ply` — thứ khiến S1 khó tự làm ngoài đời

### 5.1. Loader làm gì với nó

[`process_obstacle_points`, dòng 204-213](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L204):

```python
scene_pcd = o3d.io.read_point_cloud(".../meta/pointcloud.ply")
color_distance = np.abs(scene_color - np.array([0, 0, 0.5])).sum(axis=-1)
select_index = np.where(color_distance < 0.05)[0]      # ← LỌC THEO MÀU
```

→ **Vật cản không phải suy từ hình học — nó được ĐÁNH DẤU BẰNG MÀU.** Điểm nào có màu ≈ `(0, 0, 0.5)`
(chuẩn hoá 0–1) tức **`(0, 0, 128)` dạng uint8** thì được coi là vật cản.

### 5.2. Đo thật trên scene mẫu

| Chỉ số | Giá trị đo được |
|---|---|
| Số điểm | **88 750** |
| Định dạng | PLY nhị phân, `double x,y,z` + `uchar red,green,blue` (do Open3D tạo) |
| Màu `[102,102,102]` (xám) | 49 014 điểm — **vùng đi được** |
| Màu `[0,0,128]` (xanh đậm) | **27 276 điểm — VẬT CẢN** (đúng bằng số điểm bộ lọc chọn ra) |
| Màu `[125,255,122]` (xanh lá) | 2 591 điểm — mốc/đường đi |
| Khoảng `z` | **−0.1 … 0.71 m** |

💡 **Phát hiện quan trọng:** khoảng `z` chỉ từ −0.1 đến 0.71 m → đây **không phải bản quét 3D đầy đủ
của căn phòng**, mà là một **lát cắt sát sàn** (bản đồ "đi được / không đi được" kiểu BEV, có độ dày).
→ Làm lại thứ này ngoài đời **dễ hơn nhiều** so với tưởng tượng ban đầu: bạn chỉ cần một **bản đồ
occupancy 2D** (thứ mà RTAB-Map / Cartographer/ Nav2 costmap vẫn xuất ra) rồi **rải điểm và tô màu
`(0,0,128)` cho ô vật cản**.

### 5.3. Critic được tính thế nào

[dòng 471-494](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L471):

```python
if trajectory_obstacle_points.shape[0] != 0:
    pred_distance = khoảng cách từ mỗi điểm quỹ đạo tới vật cản gần nhất
    pred_critic  = -5.0 * (pred_distance < 0.1).mean() + 0.5 * (tổng mức tăng khoảng cách)
else:
    pred_critic = 2.0        # ← KHÔNG có điểm vật cản → critic thành HẰNG SỐ
```

→ Nếu `pointcloud.ply` không có điểm màu vật cản, code **không crash** nhưng critic trở thành hằng số
→ **model không học được gì về né vật cản**. Đây là kiểu hỏng "im lặng" nguy hiểm nhất.

---

## 6. Loader tự sinh thêm những gì (bạn **không** cần tạo tay)

Mỗi lần lấy một mẫu, `__getitem__` ([dòng 416](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L416))
tự bịa ra từ 2 nguyên liệu gốc (**quỹ đạo** + **bản đồ**):

| Thứ sinh ra | Là gì |
|---|---|
| `memory_images` | 8 ảnh gần nhất robot vừa thấy (bộ nhớ ngắn hạn), `memory_size=8` |
| `point_goal` / `image_goal` / `pixel_goal` | đích biểu diễn theo 3 cách khác nhau |
| `pred_actions` | đoạn quỹ đạo tương lai cần dự đoán (`predict_size=24` bước, dạng x,y,θ) |
| `augment_actions` | bản quỹ đạo bị **xoay ngẫu nhiên ±60°** rồi làm mượt bằng cubic spline |
| `pred_critic` / `augment_critic` | điểm an toàn của hai quỹ đạo trên |

> 📌 Ngoài ra dataset **nhân bản danh sách lên 50 lần**
> ([dòng 128-137](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L128)) — vì mỗi lần gọi
> `__getitem__` nó **bốc ngẫu nhiên một đoạn khác** của cùng episode, nên cùng một episode dùng được
> nhiều lần mà vẫn ra mẫu khác nhau.

---

## 7. ⭐ BẢNG BẮT BUỘC / KHÔNG BẮT BUỘC

### 7.1. Mức FILE

| Thành phần | Bắt buộc? | Nếu thiếu |
|---|---|---|
| `data/chunk-XXX/episode_*.parquet` | ✅ | `FileNotFoundError` ([dòng 195-196](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L195)) |
| `meta/episodes_stats.jsonl` | ✅ | crash lúc `__init__` — loader cần `image_index.min/max` để biết episode dùng ảnh nào |
| `meta/pointcloud.ply` | ✅ **về mặt chức năng** | không crash nhưng **critic thành hằng số** → mất khả năng né vật cản (mục 5.3) |
| `videos/<chunk>/observation.images.rgb/*.jpg` | ✅ | ảnh lỗi bị thay bằng ảnh đen ([dòng 142-149](../../../code/internnav/dataset/navdp_lerobot_dataset.py#L142)) → train rác |
| `videos/<chunk>/observation.images.depth/*.png` | ✅ | tương tự (depth 0) |
| `meta/episodes.jsonl` · `meta/tasks.jsonl` · `meta/info.json` | ⚪ Không | loader S1 **không đọc** |
| `observation.video.*.mp4` | ⚪ Không | dư thừa, xoá được |
| **Câu lệnh ngôn ngữ** | ⚪ **Không cần** | S1 hoàn toàn không dùng ngôn ngữ |

### 7.2. Mức CỘT trong parquet

| Cột | Bắt buộc tồn tại | Giá trị phải đúng |
|---|---|---|
| `action` (4×4/frame) | ✅ | ✅ **Có** — đây là toàn bộ nhãn quỹ đạo |
| `observation.camera_extrinsic` | ✅ | ✅ **Có** — dùng làm hệ quy chiếu gốc trong `relative_pose` |
| `observation.camera_intrinsic` | ✅ | ✅ **Có** — dùng để chiếu `pixel_goal` |
| `index` | ✅ (có trong schema) | ⚪ không ảnh hưởng nhãn |

### 7.3. Mức GIÁ TRỊ

| Dữ liệu | Có thể làm ẩu? | Vì sao |
|---|---|---|
| Quỹ đạo (`action`) | ❌ **Không** | Là đáp án chính. Pose nhiễu → model học đi loạng choạng. |
| `pointcloud.ply` có màu vật cản | ⚠️ Có thể bỏ, nhưng **mất critic** | Xem 5.3. Bỏ = chỉ còn "bắt chước quỹ đạo", né vật cản kém. |
| Depth | ❌ Không | Vào thẳng input model mỗi frame. |
| Ảnh RGB | ❌ Không | Input chính. |
| `episodes_stats.jsonl` (ngoài `image_index`) | ✅ Có thể tối giản | Đo thật: file này **chỉ có 3 trường** `episode_index`, `task_index`, `image_index{min,max,count}`. |

---

## 8. Config train trỏ vào đâu

[scripts/train/base_train/configs/navdp.py](../../../code/scripts/train/base_train/configs/navdp.py):

```python
root_dir      = 'data/datasets/InternData-N1/vln_n1/traj_data'   # ← nơi S1 đọc data
image_size    = 224
memory_size   = 8      # nhớ 8 frame
predict_size  = 24     # dự đoán 24 bước tương lai
pixel_channel = 4
epochs = 1000 ; batch_size = 32 ; lr = 1e-4
```

Chạy bằng `bash scripts/train/base_train/start_train.sh --name <tên> --model navdp` (8 GPU,
qua `torchrun`).

> ⚠️ Loader S1 cần thêm thư viện **`open3d`** và **`jsonlines`** (không có trong môi trường tối
> thiểu). Loader S2 thì không cần.

---

## 9. Tóm tắt: để có data train S1, bạn cần

| Thành phần | Bắt buộc | Nguồn |
|---|---|---|
| Ảnh RGB dọc đường | ✅ | camera |
| Ảnh Depth uint16 (**/10000**) | ✅ | camera RGB-D hoặc DepthAnything |
| `camera_intrinsic` 3×3 | ✅ | calibrate / `camera_info` |
| `camera_extrinsic` 4×4 (frame gốc) | ✅ | pose xuất phát |
| Quỹ đạo `action` 4×4 mỗi frame | ✅ | pose theo thời gian (SLAM / simulator) |
| `pointcloud.ply` có **điểm màu `(0,0,128)`** | ✅ (nếu muốn critic hoạt động) | simulator / LiDAR / occupancy map dựng lại |
| Câu lệnh | ❌ | — |
| `episodes.jsonl`, `tasks.jsonl`, `info.json` | ❌ | — |

**Kết luận thẳng:** S1 **dễ nhất khi tạo bằng simulator** (bản đồ 3D có sẵn, chính xác, miễn phí).
Ngoài đời thật, nút thắt là bản đồ vật cản — nhưng như mục 5.2 cho thấy, thứ cần thực ra chỉ là một
**bản đồ occupancy sát sàn**, không phải bản quét 3D đầy đủ.

> 💡 **Lời khuyên cho người mới:** đừng tự train S1. Dùng checkpoint NavDP có sẵn (`navdp_pretrained`,
> nhẹ) và dồn toàn bộ công sức vào **data System 2** — phần vừa khả thi vừa quyết định "độ thông minh"
> của robot. Xem [08](08_phu_luc_thu_thap_data.md) mục 3.

---

Tiếp theo: bắt tay làm data S2 thật từ log robot → [06_pipeline_mcap_to_s2](06_pipeline_mcap_to_s2.md).
