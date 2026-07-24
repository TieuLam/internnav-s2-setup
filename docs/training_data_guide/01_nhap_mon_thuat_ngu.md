# 01 — Nhập môn: từ điển thuật ngữ AI + Robot cho người mới

> **File này để làm gì:** giải thích **bằng lời thường** mọi thuật ngữ bạn sẽ gặp trong các file sau.
> Chưa cần nhớ hết — đọc lướt một lượt cho quen mặt, khi nào gặp lại thì tra ngược về đây.
>
> Bộ tài liệu: [00_README](00_README.md) · [02_hai_he_thong](02_hai_he_thong.md) ·
> [03_data_cho_system2](03_data_cho_system2.md) · [04_data_cho_system1](04_data_cho_system1.md)

---

## 1. Nhóm khái niệm AI / Machine Learning

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **Model (mô hình)** | Một "cái hộp" toán học có hàng tỷ con số bên trong (gọi là *tham số / weights*). Đưa đầu vào (ảnh, chữ) → nó trả đầu ra (điểm đích, đường đi). |
| **Train (huấn luyện)** | Quá trình cho model xem **rất nhiều ví dụ đúng** để nó tự chỉnh các con số bên trong sao cho đoán ngày càng giống đáp án. Giống dạy trẻ bằng cách cho xem hàng nghìn bài mẫu có sẵn lời giải. |
| **Dataset (bộ dữ liệu)** | Tập hợp các ví dụ dùng để train. Với ta: các đoạn robot đi trong nhà, kèm ảnh + đáp án. |
| **Label / Ground-Truth (GT) — nhãn / đáp án đúng** | "Lời giải" của mỗi ví dụ. Train tức là bắt model đoán, rồi so với label để sửa. Không có label thì không train được (kiểu này gọi là *học có giám sát — supervised learning*). |
| **Inference (suy luận)** | Lúc **dùng** model đã train xong để chạy thật (robot đang đi). Khác với train (lúc đang *dạy*). |
| **Imitation Learning (học bắt chước)** | Kiểu train mà model **bắt chước một chuyên gia**. Ở đây "chuyên gia" là đường đi tối ưu do máy tính tính sẵn trong simulator. Robot học đi giống đường mẫu đó. |
| **VLM (Vision-Language Model)** | Model **vừa nhìn ảnh vừa đọc chữ**. System 2 là một VLM: nó xem ảnh camera + đọc câu lệnh tiếng Anh. Nền tảng của nó là Qwen2.5-VL-7B (7 tỷ tham số). |
| **Diffusion Policy** | Một kiểu model chuyên **vẽ ra đường đi/hành động** bằng cách "khử nhiễu dần" (bắt đầu từ nhiễu ngẫu nhiên rồi tinh chỉnh thành quỹ đạo mượt). System 1 (NavDP) dùng kỹ thuật này. Bạn chưa cần hiểu sâu — chỉ cần biết nó **sinh ra đường đi liên tục**. |
| **Checkpoint** | File lưu lại toàn bộ con số của model tại một thời điểm (đuôi `.safetensors`). Train xong → lưu checkpoint để dùng lại. |
| **Epoch / Batch** | *Batch* = một nhóm ví dụ được đưa vào model cùng lúc. *Epoch* = một lượt học hết toàn bộ dataset. Train thường lặp nhiều epoch. |

---

## 2. Nhóm khái niệm Camera & Hình học 3D (phần khó nhất, nhưng quan trọng nhất)

Đây là nhóm bạn **bắt buộc** phải nắm để hiểu data, vì data điều hướng bản chất là **ảnh + thông tin
hình học về camera đang ở đâu**.

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **RGB** | Ảnh màu bình thường (Red-Green-Blue). Mỗi điểm ảnh (pixel) có 3 số màu. Đây là thứ camera điện thoại cho ra. |
| **Depth (ảnh độ sâu)** | Một ảnh đặc biệt: mỗi pixel không lưu màu, mà lưu **khoảng cách từ camera tới vật ở điểm đó** (vd 2.3 mét). Nhìn như ảnh xám: gần thì tối, xa thì sáng (hoặc ngược lại). Camera thường **không** có depth; camera chuyên dụng (RealSense, ZED) hoặc điện thoại đời mới (cảm biến LiDAR) mới có. |
| **RGB-D** | Camera cho **cả RGB lẫn Depth** cùng lúc. "D" là Depth. |
| **Pixel / toạ độ pixel `[u, v]`** | Một điểm trên ảnh, xác định bằng cột `u` và hàng `v`. Ví dụ ảnh 640×480 thì `u` chạy 0–639, `v` chạy 0–479. **Pixel goal** = "hãy đi về phía điểm `[u,v]` này trong ảnh". |
| **Pose (tư thế)** | **Camera (hoặc robot) đang ở đâu và quay mặt hướng nào** trong không gian. Gồm *vị trí* (x, y, z) + *hướng* (xoay quanh 3 trục). |
| **Ma trận 4×4 / SE(3)** | Cách toán học gói gọn "vị trí + hướng" thành **một bảng số 4 hàng 4 cột**. Nghe đáng sợ nhưng chỉ là "một pose được viết dưới dạng bảng số để máy tính tính toán". Mỗi khung hình (frame) có một ma trận này. Xâu chuỗi các ma trận qua nhiều frame = **quỹ đạo** (đường robot đã đi). |
| **Trajectory (quỹ đạo)** | Đường mà camera/robot đã đi, tức là **dãy pose theo thời gian**. Đây chính là "đáp án" quan trọng nhất khi dạy robot đi. |
| **Camera Intrinsic (nội tại)** | Bảng số 3×3 mô tả **đặc tính ống kính** của camera: tiêu cự (`fx, fy`) và tâm ảnh (`cx, cy`). Dùng để biết một điểm trong không gian 3D sẽ rơi vào pixel nào trên ảnh. Mỗi kiểu camera có intrinsic riêng; đo một lần bằng bước *calibrate*. |
| **Camera Extrinsic (ngoại tại)** | Chính là **pose** của camera dưới dạng ma trận 4×4 (camera đang ở đâu so với thế giới). "Extrinsic thay đổi mỗi frame" = camera di chuyển. |
| **Calibrate (hiệu chuẩn)** | Thao tác chụp một tấm bảng cờ vua để phần mềm tính ra `intrinsic` của camera. Làm một lần cho mỗi camera. |
| **Point Cloud (đám mây điểm)** | **Bản đồ 3D** của căn phòng, gồm hàng triệu điểm nhỏ trong không gian (mỗi điểm có toạ độ x,y,z và màu). Dùng để biết **chỗ nào là tường/bàn ghế (vật cản)**. Lấy từ mesh scan, LiDAR, hoặc dựng lại từ nhiều ảnh depth. File đuôi `.ply`. |
| **FOV (góc nhìn)** | Camera nhìn rộng bao nhiêu độ. D435i khoảng 68°×42°. |
| **Pitch (góc cúi)** | Camera chúc xuống bao nhiêu độ. `pitch = 0°` là nhìn thẳng; `pitch = 30°` là cúi xuống 30° (thấy sàn nhà — quan trọng để nhìn đường đi). |

> 🔑 **Nếu chỉ nhớ một câu:** data điều hướng = **ảnh** (RGB, đôi khi + Depth) **gắn với thông tin
> camera đang ở đâu** (pose / extrinsic) **và ống kính loại gì** (intrinsic). System 1 cần thêm
> **bản đồ vật cản 3D** (point cloud).

---

## 3. Nhóm khái niệm Robot & Môi trường

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **VLN (Vision-Language Navigation)** | "Điều hướng bằng thị giác + ngôn ngữ": robot **nghe câu lệnh bằng lời** rồi **nhìn** để tự đi tới đích. Đây là bài toán N1 giải. |
| **Simulator (trình mô phỏng)** | Phần mềm dựng lại căn nhà 3D trong máy tính (vd Habitat, Isaac Sim), cho robot ảo "đi" trong đó. Ưu điểm: mọi thông tin (pose, depth, bản đồ) đều **chính xác tuyệt đối và miễn phí**. Data gốc của N1 tạo bằng cách này. |
| **Scene scan (cảnh quét 3D)** | Mô hình 3D của nhà thật đã được quét sẵn (bộ Matterport3D, HM3D, Gibson…). Simulator load các scene này để render ảnh. |
| **ROS2 (Robot Operating System 2)** | Bộ khung phần mềm phổ biến để lập trình robot thật. Các bộ phận (camera, bánh xe, cảm biến) trao đổi dữ liệu qua các "kênh" gọi là **topic**. |
| **Topic / `tf` / `camera_info`** | *Topic* = kênh dữ liệu trong ROS2 (vd topic ảnh, topic depth). *`tf`* = kênh chuyên phát **pose** của robot theo thời gian. *`camera_info`* = kênh phát **intrinsic** của camera. |
| **SLAM / Odometry / VIO** | Các kỹ thuật giúp robot **tự biết nó đang ở đâu** (tính ra pose) khi di chuyển, dựa vào camera/bánh xe/cảm biến. Không có mấy cái này thì không lấy được extrinsic ngoài đời thật. |
| **DepthAnything** | Một model AI **đoán depth từ ảnh màu thường**. Dùng khi camera của bạn không có sẵn depth (vd camera điện thoại thường). Kết quả gần đúng, không chính xác bằng camera RGB-D thật. |

---

## 4. Nhóm khái niệm về định dạng file dữ liệu

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **LeRobotDataset (v2.1)** | Một **quy chuẩn cách sắp xếp thư mục** cho dữ liệu robot (do HuggingFace đề ra). Cả 3 bộ data của N1 đều gói theo chuẩn này. Mỗi *scene* (căn phòng) = một dataset riêng. |
| **Parquet** | Một định dạng file bảng (giống Excel nhưng nén và nhanh, đuôi `.parquet`). Lưu các con số của từng frame: action, pose, goal… Đọc bằng thư viện `pandas`. |
| **`meta/` (metadata)** | Thư mục chứa thông tin mô tả: câu lệnh (`tasks.jsonl`, `episodes.jsonl`), thống kê (`episodes_stats.jsonl`), và với System 1 là bản đồ 3D (`pointcloud.ply`). |
| **Episode** | **Một lượt robot đi từ điểm A tới điểm B** theo một câu lệnh. Một scene có nhiều episode. Mỗi episode gồm nhiều *frame* (khung hình liên tiếp). |
| **`info.json`** | File tự khai báo cấu trúc của dataset. ⚠️ Trong bộ data N1, file này **hay khai sai** — nguyên tắc là *tin dữ liệu thật (đọc bằng code), không tin `info.json`*. |

---

Xong phần thuật ngữ. Tiếp theo: vì sao N1 cần **hai** bộ não, và điều đó dẫn tới **hai** loại data —
[02_hai_he_thong](02_hai_he_thong.md).
