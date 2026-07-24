# 04 — Data cho System 1: cấu trúc, kiểu dữ liệu, và vì sao nó khó thu

> **File này để làm gì:** mổ xẻ bộ data **`vln_n1`** dùng để train **System 1** (bộ não "phản xạ
> nhanh" — NavDP: vẽ đường đi cong né vật cản). Điểm mấu chốt bạn sẽ hiểu: System 1 **không cần con
> người chú thích**, nhưng lại cần **bản đồ 3D của căn phòng** — thứ khó kiếm nhất. Đối chiếu loader
> thật `internnav/dataset/navdp_lerobot_dataset.py` (class `NavDP_Base_Datset`) và config
> `scripts/train/base_train/configs/navdp.py`.
>
> Bộ tài liệu: [03_data_cho_system2](03_data_cho_system2.md) · [05_thu_thap_data](05_thu_thap_data.md) ·
> handbook gốc: [../handbook/03_data_contract.md](../handbook/03_data_contract.md)

---

## 1. Model System 1 học "trò chơi" gì?

Trò chơi của S1: **cho một dãy ảnh (robot vừa đi qua) + ảnh depth + một điểm đích → vẽ ra đường đi
cong, mượt, tránh vật cản, dài khoảng vài mét phía trước.**

Điều đặc biệt: **không ai phải ngồi chú thích "đường đúng" bằng tay.** Máy tính tự suy ra đáp án từ
hai nguồn:
1. **Đường đi thật của chuyên gia** (quỹ đạo camera đã ghi lại) → đó chính là "đường đúng" để bắt chước.
2. **Bản đồ 3D vật cản** của phòng → để chấm điểm "đường này an toàn hay đâm vào tường".

Đây gọi là *imitation learning* (học bắt chước) cộng thêm một "giám khảo" (critic) chấm an toàn.

---

## 2. Cấu trúc thư mục một scene

```
<simulator>_<camera>/<scene_uuid>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/
│   ├── observation.images.rgb/     ← ảnh màu robot thấy dọc đường
│   └── observation.images.depth/   ← ảnh độ sâu
└── meta/
    ├── episodes_stats.jsonl        ← mỗi episode dùng ảnh từ frame nào tới frame nào
    └── pointcloud.ply              ← 🔑 BẢN ĐỒ 3D VẬT CẢN CỦA PHÒNG (bắt buộc)
```

- **`<simulator>`** là nơi render: `3dfront, gibson, hm3d, hssd, matterport3d, replica`.
- **`<camera>`** là loại camera mô phỏng: `d435i` hoặc `zed`. Mỗi scene được render **2 lần** theo
  **model camera thật**. Điều này quan trọng: nếu tự tạo data, camera của bạn nên **giống model thật**
  thì phân bố ảnh/độ sâu mới khớp.

---

## 3. Bảng số (parquet) — chỉ 4 cột, nhưng cột `action` là "linh hồn"

| Cột | Kiểu dữ liệu | Nghĩa |
|---|---|---|
| `index` | số nguyên | thứ tự frame: 0, 1, 2, … |
| `observation.camera_intrinsic` | ma trận 3×3 | đặc tính ống kính (không đổi suốt episode). Ví dụ đo ở `d435i`: `fx=355.8, fy=351.7, cx=240, cy=135` → ảnh 480×270. |
| `observation.camera_extrinsic` | ma trận 4×4 | pose gốc (điểm bắt đầu) của camera. |
| `action` | ma trận **4×4 mỗi frame** | **Pose camera từng frame = quỹ đạo robot đã đi = ĐÁP ÁN của S1.** Đây không phải "nút bấm" như S2, mà là đường đi liên tục. |

> ⚠️ **Khác với S2:** ở đây `action` là **ma trận 4×4 (đường đi liên tục)**, không phải số nguyên
> rời rạc. Cùng tên "action" nhưng nghĩa hoàn toàn khác. (Xem lại mục 5 của
> [02_hai_he_thong](02_hai_he_thong.md).)

> ⚠️ `vln_n1` **thiếu** các trường chuẩn của LeRobot (`timestamp`, `frame_index`, `episode_index`…),
> nên **chỉ** loader riêng `navdp_lerobot_dataset.py` đọc được, LeRobot gốc thì không.

---

## 4. Ảnh và depth — đơn vị khác S2 (bẫy!)

| Loại | Định dạng | Kiểu số | Quy đổi trong loader |
|---|---|---|---|
| RGB | frame ảnh | uint8 | thu nhỏ về 224×224, chia 255 → [0,1] |
| Depth | ảnh | uint16 | **chia cho 10000** → mét, rồi cắt: giữ trong khoảng 0.1–5 mét |

> 🚨 **Bẫy quan trọng:** System 1 chia depth cho **10000**, còn System 2 chia cho **1000** (xem
> [03](03_data_cho_system2.md) mục 4). Hai hệ dùng **hằng số quy đổi khác nhau**. Khi tự tạo data,
> phải theo đúng loader của hệ mình đang nhắm tới, nếu không độ sâu sẽ sai 10 lần.

---

## 5. `pointcloud.ply` — thứ khiến System 1 khó tự làm ngoài đời

Đây là điểm cần hiểu kỹ nhất. Loader làm gì với bản đồ 3D:

1. Đọc `meta/pointcloud.ply` → được **đám mây điểm 3D có màu** của cả căn phòng.
2. Hàm `process_obstacle_points` **lọc ra các điểm là vật cản** (dựa trên màu đặc biệt trong data
   gốc).
3. Với mỗi đường đi ứng viên, tính **khoảng cách từ đường tới vật cản gần nhất** → ra điểm **critic**
   (giám khảo): đường mà lại gần/đâm vật cản thì bị điểm âm, đường thoáng thì điểm dương.

Nhờ critic này, S1 học **phân biệt đường an toàn vs đường va chạm**. **Không có bản đồ 3D → không có
critic → model né vật cản kém.**

> Vì vậy: trong simulator, bản đồ 3D có sẵn (miễn phí, chính xác). Ngoài đời thật, muốn có nó bạn cần
> **LiDAR 3D** hoặc dựng lại bản đồ từ nhiều ảnh depth (kỹ thuật *reconstruction / SLAM*) — đây là rào
> cản lớn nhất khi thu data S1 bằng thiết bị thường. Chi tiết ở [05_thu_thap_data](05_thu_thap_data.md).

---

## 6. Loader còn tự sinh thêm những gì (không cần bạn lo, nhưng nên biết)

Từ quỹ đạo (`action`) + bản đồ, mỗi lần lấy một mẫu train, loader tự tạo:
- **`memory_images`**: 8 ảnh gần nhất robot vừa thấy (bộ nhớ ngắn hạn).
- **`point_goal` / `image_goal` / `pixel_goal`**: đích biểu diễn theo 3 cách khác nhau.
- **`pred_actions`**: đoạn quỹ đạo tương lai cần dự đoán (dạng x, y, góc quay).
- **`augment_actions`**: bản quỹ đạo bị **xoay ngẫu nhiên ±60° rồi làm mượt** — kỹ thuật *data
  augmentation* (bịa thêm biến thể để model học đa dạng hơn).
- **`pred_critic` / `augment_critic`**: điểm an toàn của các đường trên.

Bạn không cần tạo tay những thứ này — chúng được **suy ra tự động** từ 2 nguyên liệu gốc: **quỹ đạo**
và **bản đồ 3D**.

---

## 7. Config train trỏ vào đâu (bằng chứng thật)

Trong `scripts/train/base_train/configs/navdp.py`:
```python
root_dir = 'data/datasets/InternData-N1/vln_n1/traj_data'   # ← đây là nơi S1 đọc data
image_size = 224
memory_size = 8        # nhớ 8 frame
predict_size = 24      # dự đoán 24 bước tương lai
```

---

## 8. Tóm tắt: để có data train cho System 1, bạn cần

| Thành phần | Bắt buộc? | Nguồn |
|---|---|---|
| Ảnh RGB dọc đường | ✅ | camera |
| Ảnh Depth (uint16) | ✅ | camera RGB-D hoặc DepthAnything |
| `camera_intrinsic` 3×3 | ✅ | calibrate / camera_info |
| Quỹ đạo camera (`action` 4×4 mỗi frame) | ✅ | pose theo thời gian (SLAM / simulator) |
| **`pointcloud.ply`** (bản đồ 3D vật cản) | ✅ | **simulator / LiDAR / reconstruction** ← khó nhất |
| Câu lệnh | ✖ (S1 không dùng lệnh) | — |

Kết luận thẳng: **System 1 dễ nhất khi tạo bằng simulator**, khó khi thu ngoài đời vì cần bản đồ 3D.
Cách làm cụ thể từng cấp thiết bị → [05_thu_thap_data](05_thu_thap_data.md).
