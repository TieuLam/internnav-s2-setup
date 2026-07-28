# 08 — Phụ lục: thu thập dữ liệu (simulator / RGB-D / điện thoại / ROS 2), lộ trình & bẫy

> **File này để làm gì:** trả lời **"tôi lấy từng thành phần dữ liệu ở đâu, bằng thiết bị gì?"**, rồi
> đề xuất **thứ tự việc cần làm** để không sa lầy, kèm danh sách **bẫy đã kiểm chứng trong code**.
>
> Bộ tài liệu: [04_data_train_s2](04_data_train_s2.md) · [05_data_train_s1](05_data_train_s1.md) ·
> [06_pipeline_mcap_to_s2](06_pipeline_mcap_to_s2.md)

---

## 1. Bảy "nguyên liệu" cần thu

| Nguyên liệu | System 2 cần? | System 1 cần? |
|---|---|---|
| Ảnh RGB | ✅ | ✅ |
| Ảnh Depth | ✅ (chỉ cần **tồn tại** nếu train S2 thuần) | ✅ (giá trị phải đúng) |
| Camera intrinsic `K` | ✅ (để chiếu pixel goal) | ✅ |
| Pose / quỹ đạo camera (4×4) | ✅ (giá trị chỉ cần đúng nếu train dual) | ✅ (giá trị phải đúng) |
| Bản đồ 3D vật cản (`pointcloud.ply`) | ❌ | ✅ |
| Câu lệnh (instruction) | ✅ (**người viết**) | ❌ |
| Pixel goal `[u,v]` | ✅ | ❌ (S1 tự chiếu) |

---

## 2. Mỗi nguyên liệu lấy ở đâu — theo 3 cấp thiết bị

| Nguyên liệu | (A) Simulator | (B) RGB-D chuyên dụng + robot | (C) Điện thoại / camera ROS 2 thường |
|---|---|---|---|
| **RGB** | render sẵn | RealSense D435i/D455, ZED 2 | camera điện thoại / topic `image_raw` |
| **Depth** | render sẵn (chuẩn) | có sẵn (uint16 mm) | ❗ không sẵn → **DepthAnythingV2**, hoặc ảnh giữ chỗ nếu chỉ train S2 |
| **Intrinsic `K`** | từ model camera mô phỏng | **calibrate** 1 lần bằng bảng cờ vua | điện thoại: ARKit/EXIF · ROS 2: topic `camera_info` |
| **Pose / quỹ đạo** | engine cho sẵn (chuẩn tuyệt đối) | odometry + IMU + LiDAR-SLAM | điện thoại: **ARKit/ARCore** · ROS 2: `tf` từ VIO/visual-SLAM (ORB-SLAM3, RTAB-Map) |
| **Bản đồ 3D `.ply`** | export từ mesh scene (miễn phí) | LiDAR 3D hoặc reconstruction RGB-D | ❗ khó — dựng bằng RTAB-Map rồi tự gán nhãn vật cản |
| **Câu lệnh** | người viết theo quỹ đạo | người viết | người viết |
| **Pixel goal** | chiếu đích 3D về ảnh | chiếu bằng `K` + pose | chiếu về ảnh, hoặc **chấm tay** trên frame |

---

## 3. Ba cấp thiết bị — chi tiết

### 3.1. Cấp A — Simulator (cách data gốc được tạo)

**Khuyến nghị nếu mục tiêu là train nghiêm túc.** "Chuyên dụng" ở đây là **phần mềm mô phỏng + nhà
quét 3D**, không phải thiết bị đắt tiền.

- **Công cụ:** Habitat-Sim hoặc Isaac Sim, load scene quét sẵn (Matterport3D, HM3D, Gibson, Replica,
  HSSD, 3D-Front).
- **Quy trình:** load nhà 3D → cho robot ảo đi theo đường tối ưu (hoặc tự lái) → mỗi bước ghi RGB,
  depth, pose 4×4, `K` → export mesh ra `pointcloud.ply` → đóng gói LeRobot.
- **Ưu:** pose/depth/bản đồ **chính xác tuyệt đối, miễn phí, quy mô lớn**; tạo được **cả S1 lẫn S2**.
- **Nhược:** ảnh mô phỏng khác ảnh đời thật (*sim-to-real gap*).

### 3.2. Cấp B — Camera RGB-D gắn robot (thu ngoài đời)

Khả thi cho **cả S1 và S2**, tốn công hạ tầng. Dùng khi muốn data **thật** để fine-tune.

- **Camera:** RealSense **D435i/D455** hoặc **ZED 2** — nên chọn đúng dòng mà `vln_n1` mô phỏng để
  phân bố ảnh khớp ([05](05_data_train_s1.md) mục 2).
- **Pose:** encoder bánh xe + IMU + LiDAR-SLAM (Cartographer, LIO-SAM).
- **Bản đồ vật cản:** LiDAR 3D, hoặc tích luỹ depth để dựng map rồi **tô màu `(0,0,128)` cho điểm vật
  cản** (đúng quy ước loader — [05](05_data_train_s1.md) mục 5).
- **Ghi log:** đồng bộ topic rồi ghi ra `.mcap` → chạy [06](06_pipeline_mcap_to_s2.md).

### 3.3. Cấp C — Thiết bị thường (điện thoại / camera robot)

**Kết luận ngắn: làm được data System 2, khó làm data System 1.**

#### S2 bằng thiết bị thường — **KHẢ THI** ✅

| Nguyên liệu | Cách làm |
|---|---|
| RGB | Quay video khi đi theo lộ trình, camera đặt ~60–125 cm, **chúc xuống** đủ thấy sàn |
| Depth | Chạy **DepthAnythingV2** (chế độ metric) xuất uint16 mm; hoặc ảnh giữ chỗ nếu chỉ train S2 |
| Pose 4×4 | Điện thoại: **ARKit/ARCore** · Robot ROS 2: **ORB-SLAM3 / RTAB-Map** → `tf` |
| `K` | Điện thoại: ARKit/EXIF · ROS 2: `camera_info` |
| Câu lệnh | **Bạn tự viết** cho mỗi lượt đi (tiếng Anh) |
| Action rời rạc | Suy từ chuyển động giữa 2 frame ([06](06_pipeline_mcap_to_s2.md) giai đoạn C.1) |
| Pixel goal | Chiếu waypoint 3D về ảnh ([06](06_pipeline_mcap_to_s2.md) giai đoạn C.3), hoặc **chấm tay** |

#### S1 bằng thiết bị thường — **KHÓ** ❗

Nút thắt là `pointcloud.ply` có nhãn vật cản. Ba lựa chọn:

| Phương án | Đánh giá |
|---|---|
| Bỏ critic, chỉ bắt chước quỹ đạo | Model đi được nhưng **né vật cản kém** — chỉ nên thử nghiệm |
| Dựng map bằng RTAB-Map rồi gán nhãn vật cản theo độ cao | Chất lượng hạn chế; nhớ rằng bản đồ gốc chỉ là **lát cắt sát sàn** `z ∈ [−0.1, 0.71] m` ([05](05_data_train_s1.md) mục 5.2) nên **occupancy map 2D là đủ** |
| **Không tự train S1** — dùng checkpoint NavDP có sẵn | ✅ **Khôn ngoan nhất cho người mới** |

---

## 4. Cần bao nhiêu camera khi quay thật?

> Câu hỏi hay gặp khi thấy một scene có **10 folder ảnh**: *"Vậy phải gắn 5 camera cùng lúc?"*

### 4.1. KHÔNG — 5 setting là "đặc sản của simulator"

Năm setting **không phải 5 camera chạy đồng thời**. Đó là **cùng một lần robot đi**, rồi simulator
**render lại 5 lần** giả vờ camera đặt ở độ cao/góc cúi khác. Trong máy tính, render thêm một góc gần
như **miễn phí**. Lúc train, mỗi cấu hình **chỉ dùng 1 (hoặc 1 cặp)** setting.

### 4.2. "1 góc" vs "2 góc"

| Setting | pitch_1 | pitch_2 | Số camera thật cần |
|---|---|---|---|
| `60cm_15_15` | 15° | 15° | **1** |
| `60cm_30_30` | 30° | 30° | **1** |
| `125cm_0_30` | 0° | 30° | 2 (hoặc 1 camera cúi được) |
| `125cm_0_45` | 0° | 45° | 2 |

→ Chọn cấu hình `pitch_1 == pitch_2` thì "ảnh thẳng" và "ảnh cúi" **là cùng một file** — loader thay
chuỗi ra đúng đường dẫn cũ ([04](04_data_train_s2.md) mục 2.1).

### 4.3. Trường hợp robot hình người (đầu ~125 cm, cổ cúi được)

Setting 125 cm dùng **hai góc thật sự khác nhau**, mô phỏng đúng hành vi người: *đi thì nhìn thẳng,
cần chấm đích thì liếc xuống sàn*.

⚠️ **Cái bẫy:** trong data gốc, cả luồng `0°` lẫn `30°` tồn tại **song song từng frame** (sim render
đồng thời). Một camera thật **không thể ở hai góc cùng lúc**.

| Phương án | Số camera | Cách làm | Đánh giá |
|---|---|---|---|
| **A. Hai camera gắn đầu** | 2 | 1 hướng thẳng (0°) + 1 nghiêng cố định (30°/45°), **chụp đồng thời** | ✅ **Khuyến nghị** — đúng cặp `125cm_0_30`, đồng bộ hoàn hảo |
| **B. Một camera biết cúi** | 1 | Đi nhìn thẳng; tới điểm chấm đích thì **dừng, cúi đầu chụp**, rồi đi tiếp | ⚠️ Được, nhưng phải chỉnh pipeline; đây đúng là cách robot chạy lúc inference |
| **C. Một camera, một góc cố định** | 1 | Dùng cấu hình `pitch_1 == pitch_2` | ✅ Đơn giản nhất · ❌ mất bước cúi nhìn sàn |

**Nguyên tắc vàng:** không quan trọng "bao nhiêu camera", mà là **cấu hình camera lúc quay data phải
khớp cấu hình lúc robot chạy thật** ([02](02_he_thong.md) mục 8.3).

---

## 5. Bảng độ khả thi

| | Simulator (A) | RGB-D + robot (B) | Điện thoại / ROS 2 (C) |
|---|---|---|---|
| Tạo data **System 2** | ✅ Dễ, chuẩn | ✅ Được | ✅ **Khả thi — nên bắt đầu ở đây** |
| Tạo data **System 1** | ✅ Dễ, chuẩn | ⚠️ Được nhưng công phu | ❗ Khó (thiếu bản đồ vật cản) |
| Chi phí | Thấp (chỉ cần GPU) | Cao (robot + cảm biến) | Rất thấp |
| Độ giống đời thật | Trung bình (sim-gap) | Cao | Cao |

---

## 6. Lộ trình đề xuất

> **Nguyên tắc vàng cho người mới:** đừng cố làm cả hai hệ cùng lúc bằng thiết bị thường.
> **Làm data System 2 trước** — khả thi nhất và cho kết quả nhìn thấy sớm nhất.

### Mức 0 — Hiểu, chưa tạo gì (1–2 ngày)
1. Đọc 01→05 của bộ tài liệu này.
2. Mở scene có sẵn `InternNav/data/vln_ce/traj_data/r2r/17DRP5sb8fy`, đọc parquet bằng `pandas`, in
   `df.dtypes` và vài hàng đầu → **đối chiếu bảng cột ở [04](04_data_train_s2.md)**.
3. Mục tiêu: "sờ" được dữ liệu thật, thấy đúng như mô tả.

### Mức 1 — Chạy thử pipeline trên dữ liệu mô phỏng (nửa ngày)
```bash
cd docs/training_data_guide/tools
python generate_s2_mcap.py --out demo_s2_robot.mcap
python mcap2s2.py --mcap demo_s2_robot.mcap --out ./traj_data --dataset-name myrobot --scene-id demo_scene
```
→ Hiểu trọn vòng đời dữ liệu **trước khi** đụng vào log thật. Chi tiết: [06](06_pipeline_mcap_to_s2.md).

### Mức 2 — Data S2 quy mô nhỏ từ thiết bị thật (1–2 tuần)
1. Quay 5–10 lượt đi ngắn (điện thoại có ARKit/ARCore, hoặc robot ROS 2 ghi `.mcap`).
2. Mỗi lượt viết 1 câu lệnh tiếng Anh.
3. Chạy `mcap2s2.py` với topic đã ánh xạ ([06](06_pipeline_mcap_to_s2.md) mục 7).
4. **Kiểm định bằng loader thật** ([06](06_pipeline_mcap_to_s2.md) mục 4).
5. Fine-tune từ checkpoint `InternVLA-N1-System2` ([03](03_code_train_s2.md) mục 7.1).

### Mức 3 — Quy mô lớn bằng simulator (nếu muốn train nghiêm túc)
Cài Habitat-Sim, load vài scene Matterport3D/HM3D, sinh hàng nghìn episode → data S2 **và** S1 chuẩn.

### Mức 4 — System 1 (chỉ khi đã vững)
Ưu tiên dùng checkpoint NavDP có sẵn. Nếu tự tạo: làm bằng simulator (có bản đồ 3D sẵn).

---

## 7. Checklist bẫy dữ liệu (tick trước khi tin kết quả)

- [ ] **Đơn vị depth khác nhau giữa 2 hệ:** S2 chia **1000**, S1 chia **10000**.
- [ ] **Depth phải là PNG uint16 milimét**, không phải float mét.
- [ ] **Pixel goal là `[u, v] = [cột, hàng]`.** Đầu ra model thường là `[hàng, cột]` → **phải đảo**;
      và pixel của model nằm trong ảnh 384×384 còn data là 640×480 → **phải scale**.
- [ ] **`action = -1` là frame khởi đầu** → loại khỏi thống kê độ chính xác.
- [ ] **"action" của S2 (số nguyên) ≠ "action" của S1 (ma trận 4×4)**.
- [ ] **Camera phải cúi** đủ thấy sàn — cấu hình nhìn thẳng cho `goal` toàn `-1`.
- [ ] **Tin `df.dtypes`, không tin `info.json`**.
- [ ] **Dùng cột `timestamp`**, không suy thời gian từ `fps`.
- [ ] **Mỗi scene là một dataset độc lập** — gộp nhiều scene phải tự quản lý index.
- [ ] **RGB `.jpg`, depth `.png`** (S2). Tên file S1 phải **đệm số 0**, S2 thì **không**.
- [ ] **Đường dẫn không chứa chữ `rgb` ở thư mục cha** (loader S2 dùng `.replace('rgb','depth')`).
- [ ] **dtype parquet đúng** — sai là dataset rỗng mà không có traceback.
- [ ] **S1 bắt buộc có `pointcloud.ply` với điểm màu `(0,0,128)`** — thiếu thì critic thành hằng số.
- [ ] **Đồng bộ thời gian trong cùng một lượt quay** (RGB ↔ depth ↔ pose) — sai kiểu này im lặng nhất.

---

## 8. Sai lầm tư duy hay gặp

| Nghĩ sai | Thực tế |
|---|---|
| "Có một dataset N1, cứ đổ hết vào train." | Có **3 bộ con** cho **3 model khác nhau**, không thay thế được nhau. |
| "Cứ có ảnh là train được." | Cần **pose** (camera ở đâu) + **`K`**; với S1 còn cần **bản đồ vật cản**. |
| "Điện thoại quay là xong data." | Điện thoại cho RGB (+pose qua ARKit). Depth phải *ước lượng*, pixel goal & câu lệnh phải *tạo thêm*. |
| "Train S1 và S2 giống nhau." | Hai pipeline, hai loader, hai loại label hoàn toàn khác. |
| "`info.json` nói sao thì đúng vậy." | `info.json` trong bộ N1 **hay sai** — luôn kiểm bằng dữ liệu thật. |
| "Phải có 5 camera vì thấy 5 setting." | 5 setting là simulator render dư. Robot thật nhiều nhất cần 2. |
| "Bản đồ 3D phải là bản quét đầy đủ căn phòng." | Đo thật: chỉ là **lát cắt sát sàn** `z ∈ [−0.1, 0.71] m` — occupancy map 2D là đủ. |
| "Phải train S2 từ đầu." | **Luôn là fine-tune** — từ Qwen2.5-VL hoặc từ checkpoint N1 công khai. |

---

*Quay lại mục lục: [00_README](00_README.md).*
