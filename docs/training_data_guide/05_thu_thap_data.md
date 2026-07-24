# 05 — Thu thập data: simulator, camera chuyên dụng, điện thoại, và ROS2

> **File này để làm gì:** trả lời câu hỏi cốt lõi — **"tôi lấy từng thành phần dữ liệu ở đâu, bằng
> thiết bị gì?"**. Chia làm 3 cấp: (A) simulator, (B) camera RGB-D chuyên dụng gắn robot, (C) thiết bị
> thường (camera điện thoại / camera robot qua ROS2). Với mỗi thành phần, chỉ rõ cách lấy và độ khó.
>
> Bộ tài liệu: [03_data_cho_system2](03_data_cho_system2.md) · [04_data_cho_system1](04_data_cho_system1.md) ·
> [06_lo_trinh_bat_dau](06_lo_trinh_bat_dau.md)

---

## 1. Nhắc lại: bạn cần thu những "nguyên liệu" nào

Gộp từ file 03 và 04, đây là toàn bộ nguyên liệu cần thu (các thứ khác máy tự suy ra):

| Nguyên liệu | System 2 cần? | System 1 cần? |
|---|---|---|
| Ảnh RGB | ✅ | ✅ |
| Ảnh Depth | ✅ | ✅ |
| Camera intrinsic (ống kính) | ✅ | ✅ |
| Pose / quỹ đạo camera (extrinsic 4×4) | ✅ | ✅ |
| Bản đồ 3D vật cản (`pointcloud.ply`) | ✖ | ✅ |
| Câu lệnh (instruction) | ✅ (người viết) | ✖ |
| Pixel goal `[u,v]` | ✅ | ✖ (S1 tự chiếu) |

Phần còn lại của file này giải thích **lấy 7 nguyên liệu đó ở đâu** theo từng cấp thiết bị.

---

## 2. Bảng tổng: mỗi nguyên liệu lấy từ đâu theo 3 cấp thiết bị

| Nguyên liệu | (A) Simulator | (B) RGB-D chuyên dụng + robot | (C) Điện thoại / camera ROS2 thường |
|---|---|---|---|
| **RGB** | render sẵn | RealSense D435i/D455, ZED 2 | camera điện thoại / camera robot (topic `image_raw`) |
| **Depth** | render sẵn (chuẩn) | có sẵn (uint16 mm) | ❗không sẵn → chạy **DepthAnythingV2** để đoán, hoặc mua camera RGB-D rẻ |
| **Intrinsic** | từ model camera mô phỏng | **calibrate** 1 lần bằng bảng cờ vua | điện thoại: đọc từ ARKit/EXIF; ROS2: topic `camera_info` |
| **Pose / quỹ đạo (extrinsic)** | engine cho sẵn (chuẩn tuyệt đối) | odometry + IMU + **LiDAR-SLAM** | điện thoại: **ARKit/ARCore**; ROS2: `tf` từ **VIO/visual-SLAM** (ORB-SLAM3, RTAB-Map) |
| **Bản đồ 3D (`.ply`)** | export từ mesh scene (miễn phí) | LiDAR 3D hoặc reconstruction RGB-D | ❗rất khó — phải dựng bằng RTAB-Map rồi tự gán nhãn vật cản |
| **Câu lệnh** | người viết theo quỹ đạo | người viết | người viết |
| **Pixel goal** | chiếu đích 3D về ảnh (biết sẵn đích) | chiếu đích về ảnh bằng intrinsic+pose | chiếu về ảnh, hoặc **chấm tay** trên frame |

Đọc theo cột: chọn cấp thiết bị của bạn rồi xem từng hàng cần làm gì.

---

## 3. Cấp A — Simulator (cách data gốc InternData-N1 được tạo)

**Đây là cách được khuyến nghị nếu mục tiêu là train nghiêm túc.** "Chuyên dụng" thực chất ở đây là
**phần mềm mô phỏng + nhà quét 3D**, không phải thiết bị đắt tiền.

- **Công cụ:** Habitat-Sim hoặc Isaac Sim, load scene quét sẵn (Matterport3D, HM3D, Gibson, Replica,
  HSSD, 3D-Front).
- **Quy trình:**
  1. Load một căn nhà 3D vào simulator.
  2. Cho robot ảo đi theo *đường tối ưu* (shortest-path) từ A tới B, hoặc bạn tự lái (teleop).
  3. Mỗi bước, ghi lại: ảnh RGB, ảnh depth, pose camera (4×4), intrinsic.
  4. Export mesh của scene ra `pointcloud.ply`.
  5. Đóng gói theo chuẩn LeRobotDataset đúng các cột như file 03/04.
- **Ưu điểm:** pose/depth/bản đồ **chính xác tuyệt đối, miễn phí, quy mô lớn**. Tạo được **cả S1 lẫn
  S2**.
- **Nhược điểm:** ảnh là đồ hoạ mô phỏng, khác ảnh đời thật một chút (gọi là *sim-to-real gap*).

---

## 4. Cấp B — Camera RGB-D chuyên dụng gắn robot (thu ngoài đời thật)

Khả thi cho **cả S1 và S2**, nhưng tốn công hạ tầng. Dùng khi bạn muốn data **thật** để fine-tune.

- **Camera:** Intel RealSense **D435i/D455** hoặc **ZED 2**. Nên chọn đúng dòng mà `vln_n1` mô phỏng
  (`d435i`/`zed`) để phân bố ảnh khớp nhất.
- **Lấy pose (extrinsic):** robot cần **encoder bánh xe + IMU + LiDAR-SLAM** (ví dụ Cartographer,
  LIO-SAM) để tính pose 4×4 mỗi frame một cách chính xác.
- **Lấy bản đồ 3D:** từ **LiDAR 3D**, hoặc tích luỹ nhiều ảnh depth để dựng map, rồi **gán nhãn vật
  cản** (ví dụ: điểm nào thấp/gần đường đi thì coi là chướng ngại).
- **Đường ống ROS2:** đồng bộ các topic theo thời gian rồi ghi ra file (xem mục 5.3).

---

## 5. Cấp C — Thiết bị thường: điện thoại / camera robot qua ROS2

Đây là phần trả lời trực tiếp câu hỏi của bạn. **Kết luận ngắn: làm được data System 2, khó làm data
System 1.**

### 5.1. System 2 bằng thiết bị thường — KHẢ THI ✅

Vì S2 về bản chất chỉ cần **RGB + câu lệnh + pixel goal + action rời rạc** — không cần bản đồ 3D.

| Nguyên liệu | Cách làm với điện thoại / camera thường |
|---|---|
| RGB (2 góc) | Quay video khi đi theo lộ trình. Camera đặt ở tầm ~60–125 cm. Cần góc nhìn thẳng + góc cúi → quay 2 lượt, hoặc gắn camera hơi chúc xuống. |
| Depth | Không có sẵn → chạy **DepthAnythingV2** (chế độ metric) trên từng frame RGB, xuất ảnh uint16 milimét. Chấp nhận sai số. |
| Pose 4×4 | **Điện thoại:** dùng **ARKit (iPhone)** / **ARCore (Android)** — chúng cho pose khá tốt sẵn. **Robot ROS2:** chạy **ORB-SLAM3** hoặc **RTAB-Map** → lấy pose từ topic `tf`. |
| Intrinsic | Điện thoại: đọc từ ARKit/EXIF. ROS2: topic `camera_info`. |
| Câu lệnh | **Bạn tự viết** cho mỗi lượt đi (tiếng Anh). |
| Action rời rạc | Suy từ chuyển động giữa 2 frame: đi thẳng → `1`; xoay trái/phải → `2`/`3`; đứng yên cuối → `0` (STOP). |
| Pixel goal `[u,v]` | Hai cách: (1) nếu biết toạ độ đích 3D + có pose + intrinsic → **chiếu** đích về ảnh (công thức trong loader S2); (2) **chấm tay** điểm đích trên frame bằng một công cụ đánh nhãn. |

### 5.2. System 1 bằng thiết bị thường — KHÓ ❗

Nút thắt là **`pointcloud.ply` có nhãn vật cản** (để tính critic). Không có LiDAR thì:
- **Phương án tối giản:** bỏ phần critic, chỉ train bắt chước quỹ đạo từ pose → model đi được nhưng
  **né vật cản kém**. Chỉ nên dùng thử nghiệm.
- **Phương án dựng map:** dùng **RTAB-Map** dựng bản đồ 3D từ chính ảnh depth (kể cả depth ước lượng),
  rồi gán nhãn vật cản bằng luật đơn giản (theo độ cao). Chất lượng hạn chế → chỉ nên fine-tune hoặc
  đánh giá định tính, **không** dùng để train từ đầu.
- **Phương án khôn ngoan:** **không tự train S1** — dùng checkpoint NavDP có sẵn (`navdp_pretrained`,
  nhẹ, không cần checkpoint 16.79 GB), chỉ tập trung công sức vào data S2.

### 5.3. Gợi ý đường ống ROS2 (khi dùng camera robot)

Nếu robot chạy ROS2, thu data bằng cách **đồng bộ các topic theo timestamp**:
- `image_raw` (RGB), `depth/image` (nếu có RGB-D), `tf` (pose), `camera_info` (intrinsic).
- Dùng `message_filters.ApproximateTimeSynchronizer` để gom các topic cùng thời điểm thành một frame.
- Ghi ra `rosbag`, sau đó viết script chuyển sang `.parquet` + thư mục ảnh đúng chuẩn LeRobot.

> Đây là node ROS2 mà bạn có thể nhờ tôi viết ở bước sau (xem [06_lo_trinh_bat_dau](06_lo_trinh_bat_dau.md)).

---

## 6. Bảng "độ khả thi" tổng kết

| | Simulator (A) | RGB-D + robot (B) | Điện thoại / ROS2 thường (C) |
|---|---|---|---|
| Tạo data **System 2** | ✅ Dễ, chuẩn | ✅ Được | ✅ **Khả thi** (nên bắt đầu ở đây) |
| Tạo data **System 1** | ✅ Dễ, chuẩn | ⚠️ Được nhưng công phu | ❗ Khó (thiếu bản đồ 3D) |
| Chi phí | Thấp (chỉ cần GPU) | Cao (robot + cảm biến) | Rất thấp |
| Độ giống đời thật | Trung bình (sim-gap) | Cao | Cao |

Bước tiếp: chọn lộ trình phù hợp và tránh các bẫy — [06_lo_trinh_bat_dau](06_lo_trinh_bat_dau.md).
