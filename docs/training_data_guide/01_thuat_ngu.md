# 01 — Từ điển thuật ngữ: ML · Camera & 3D · Robot/ROS · Định dạng dữ liệu

> **File này để làm gì:** giải thích **bằng lời thường** mọi thuật ngữ bạn sẽ gặp trong 7 file còn
> lại. Chưa cần nhớ hết — đọc lướt cho quen mặt, gặp lại thì tra ngược về đây.
>
> Mục lục bộ tài liệu: [00_README](00_README.md)

---

## 1. Nhóm Machine Learning cơ bản

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **Model (mô hình)** | Một "cái hộp" toán học chứa hàng tỷ con số (gọi là *tham số / weights*). Đưa đầu vào (ảnh, chữ) → trả đầu ra (điểm đích, đường đi). |
| **Policy (chính sách)** | Trong robot, người ta gọi model là "policy": quy tắc *"thấy tình huống X thì làm hành động Y"*. Trong repo bạn thấy `CMA_Policy`, `RDP_Policy`, `NavDP_Policy`, `InternVLAN1_Policy` — đó là các model khác nhau. |
| **Train (huấn luyện)** | Cho model xem **rất nhiều ví dụ có sẵn đáp án** để nó tự chỉnh các con số bên trong sao cho đoán ngày càng đúng. |
| **Dataset (bộ dữ liệu)** | Kho ví dụ để train. Ở đây: các lượt robot đi trong nhà + ảnh + đáp án. |
| **Label / Ground-Truth (GT)** | "Lời giải" của mỗi ví dụ. Không có label thì không train được (kiểu học này gọi là *supervised learning* — học có giám sát). |
| **Loss** | "Điểm phạt" đo model sai bao nhiêu. Train = tìm cách làm loss nhỏ dần. |
| **Epoch / Batch / Batch size** | *Batch* = một nhóm ví dụ đưa vào model cùng lúc. *Epoch* = một lượt học hết toàn bộ dataset. `batch_size=2` = mỗi bước xử lý 2 mẫu. |
| **Learning rate (lr)** | "Tốc độ học". Cao → học nhanh nhưng dễ loạn; thấp → chậm mà chắc. `2e-5` = 0.00002. |
| **Checkpoint** | File lưu toàn bộ con số của model tại một thời điểm (đuôi `.safetensors`). Train xong → lưu để dùng lại. |
| **Fine-tune (tinh chỉnh)** | Lấy model **đã học sẵn** rồi huấn luyện thêm cho hợp nhiệm vụ mới. Nhanh và hiệu quả hơn train từ đầu. |
| **From scratch (từ đầu)** | Khởi tạo model bằng số ngẫu nhiên rồi học lại toàn bộ. **InternVLA-N1 không dùng cách này** ([03](03_code_train_s2.md) mục 7). |
| **Freeze (đóng băng)** | Khoá một phần model lại, không cho nó học nữa (`requires_grad = False`). Dùng khi chỉ muốn dạy phần mới. |
| **Inference (suy luận)** | Lúc **dùng** model đã train xong để chạy thật. Khác với lúc *đang dạy*. |
| **Imitation Learning (học bắt chước)** | Model **bắt chước một chuyên gia**. Ở đây "chuyên gia" là đường đi tối ưu tính sẵn trong simulator. |
| **Data augmentation** | Bịa thêm biến thể của dữ liệu (đổi màu, độ nét, xoay…) để model học "chắc" hơn, đỡ phụ thuộc điều kiện cụ thể. |
| **VLM (Vision-Language Model)** | Model **vừa nhìn ảnh vừa đọc chữ**. System 2 là một VLM, nền tảng là **Qwen2.5-VL-7B** (7 tỷ tham số). |
| **Diffusion Policy** | Kiểu model **vẽ ra đường đi** bằng cách "khử nhiễu dần" (bắt đầu từ nhiễu ngẫu nhiên → tinh chỉnh thành quỹ đạo mượt). System 1 dùng kỹ thuật này. |
| **Token / Tokenizer** | Model ngôn ngữ không đọc chữ trực tiếp mà đọc **token** (mẩu chữ đã đánh số). *Tokenizer* là bộ chuyển chữ ↔ số. Ảnh cũng bị biến thành token (`<image_pad>`). |
| **Latent** | Một vector số "gói ý định" mà model truyền nội bộ. S2 gửi latent cho S1 thay vì gửi chữ. |

---

## 2. Nhóm huấn luyện quy mô lớn (nhiều GPU)

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **GPU** | Card đồ hoạ — phần cứng tính ma trận cực nhanh, thứ ML sống nhờ vào. Model 7B thường cần ≥ 24–40 GB VRAM mỗi GPU. |
| **Distributed training** | Chia việc cho nhiều GPU (thậm chí nhiều máy = *node*) chạy song song. |
| **`torchrun`** | Lệnh của PyTorch để khởi động nhiều tiến trình GPU cùng lúc. |
| **SLURM / `#SBATCH`** | Phần mềm xếp lịch job trên cụm siêu máy tính. Dòng `#SBATCH -N 8 --gres=gpu:8` = xin 8 máy × 8 GPU = **64 GPU**. |
| **DeepSpeed / ZeRO** | Kỹ thuật **chia nhỏ bộ nhớ** để nhét model khổng lồ vừa vào GPU. Stage 2 chia *gradient + optimizer*; Stage 3 chia thêm cả *trọng số*; `zero3_offload` đẩy bớt xuống **CPU RAM** (chậm hơn nhưng chạy được máy yếu). |
| **Gradient checkpointing** | Đánh đổi: tính lại một số phép thay vì nhớ kết quả → **tốn thời gian, tiết kiệm VRAM**. |
| **bf16** | Kiểu số thực 16-bit (bfloat16). Nhẹ hơn 32-bit, đủ chính xác cho train model lớn. |
| **Gradient accumulation** | Gộp nhiều batch nhỏ lại trước khi cập nhật trọng số → giả lập batch to trên GPU nhỏ. |
| **Collator** | Hàm gom nhiều mẫu lẻ thành một batch (đệm cho bằng độ dài, xếp chồng tensor). |

---

## 3. Nhóm Camera & Hình học 3D — **phần khó nhất, nhưng quan trọng nhất**

Data điều hướng bản chất là **ảnh + thông tin camera đang ở đâu**. Không nắm nhóm này thì không hiểu
được data.

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **RGB** | Ảnh màu bình thường (Red-Green-Blue), mỗi pixel 3 số màu. |
| **Depth (ảnh độ sâu)** | Ảnh mà mỗi pixel lưu **khoảng cách từ camera tới vật ở điểm đó**. Nhìn như ảnh xám. Camera thường **không** có; cần camera RGB-D (RealSense, ZED) hoặc ước lượng bằng AI. |
| **RGB-D** | Camera cho **cả RGB lẫn Depth** cùng lúc. |
| **Pixel / toạ độ `[u, v]`** | Một điểm trên ảnh: `u` = **cột**, `v` = **hàng**. Ảnh 640×480 → `u ∈ 0..639`, `v ∈ 0..479`. ⚠️ Nhiều thư viện dùng thứ tự ngược `[hàng, cột]` — **đây là nguồn lỗi kinh điển**. |
| **Pixel goal** | "Hãy đi về phía điểm `[u,v]` này **trên tấm ảnh**". Đây là **đáp án chính** mà System 2 phải học xuất ra. |
| **Waypoint** | Điểm đích trung gian trên đường đi (chưa phải đích cuối). Pixel goal = waypoint đã chiếu xuống ảnh. |
| **Pose (tư thế)** | Camera/robot **đang ở đâu và quay mặt hướng nào**: *vị trí* (x,y,z) + *hướng* (xoay quanh 3 trục). |
| **Ma trận 4×4 / SE(3)** | Cách gói "vị trí + hướng" thành **một bảng số 4×4**. 3×3 góc trên-trái = hướng xoay; cột thứ 4 = vị trí. |
| **Trajectory (quỹ đạo)** | Đường robot đã đi = **dãy pose theo thời gian**. Là "đáp án" quan trọng nhất khi dạy S1. |
| **Camera Intrinsic (nội tại) — ma trận `K`** | Bảng 3×3 mô tả **ống kính**: tiêu cự `fx, fy` và tâm ảnh `cx, cy`. Dùng để biết điểm 3D rơi vào pixel nào. Đo một lần bằng bước *calibrate*. |
| **Camera Extrinsic (ngoại tại)** | Chính là **pose camera** dưới dạng ma trận 4×4. "Extrinsic đổi mỗi frame" = camera đang di chuyển. |
| **Quy ước hệ camera (OpenCV)** | `x` = phải, `y` = **xuống**, `z` = trục quang (nhìn ra trước). Data `vln_ce` dùng đúng quy ước này ([04](04_data_train_s2.md) mục 5). |
| **Quy ước hệ robot** | `x` = trước, `y` = trái, `z` = lên. Khác hệ camera → phải có phép đổi trục giữa hai bên. |
| **Phép chiếu phối cảnh** | Công thức đưa điểm 3D về pixel: `u = fx·X/Z + cx`, `v = fy·Y/Z + cy` (với `X,Y,Z` trong hệ camera). Nếu `Z ≤ 0` → điểm ở **sau lưng** camera → không thấy. |
| **Calibrate (hiệu chuẩn)** | Chụp một bảng cờ vua để phần mềm tính ra `K`. Làm một lần cho mỗi camera. |
| **Point Cloud (đám mây điểm)** | **Bản đồ 3D** của căn phòng gồm hàng triệu điểm (x,y,z + màu). Dùng để biết chỗ nào là vật cản. File `.ply`. |
| **FOV (góc nhìn)** | Camera nhìn rộng bao nhiêu độ. RealSense D435i ≈ 69°×42° (kênh màu). |
| **Pitch (góc cúi)** | Camera chúc xuống bao nhiêu độ. `pitch = 0°` nhìn thẳng; `pitch = 30°` cúi 30° (thấy sàn — **quan trọng để chấm pixel goal**). |
| **Yaw** | Góc quay quanh trục đứng (robot xoay trái/phải). |

> 🔑 **Nếu chỉ nhớ một câu:** data điều hướng = **ảnh** + **camera đang ở đâu** (pose/extrinsic) +
> **ống kính loại gì** (intrinsic `K`). System 1 cần thêm **bản đồ vật cản 3D**.

---

## 4. Nhóm Robot & ROS 2 & MCAP

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **VLN (Vision-Language Navigation)** | "Điều hướng bằng thị giác + ngôn ngữ": robot **nghe lệnh bằng lời** rồi **nhìn** để tự đi tới đích. Đây là bài toán N1 giải. |
| **Episode** | **Một lượt robot đi từ A tới B** theo một câu lệnh. Gồm nhiều *frame* liên tiếp. |
| **Frame** | Một khung hình/khoảnh khắc. Một episode dài 46 frame = 46 hàng trong bảng số. |
| **Simulator** | Phần mềm dựng nhà 3D trong máy (Habitat, Isaac Sim) cho robot ảo "đi". Ưu điểm: pose/depth/bản đồ **chính xác tuyệt đối, miễn phí**. |
| **Scene scan** | Mô hình 3D nhà thật đã quét sẵn (Matterport3D, HM3D, Gibson…). Simulator load các scene này để render ảnh. |
| **ROS 2** | Bộ khung phần mềm phổ biến để lập trình robot thật. Các bộ phận trao đổi dữ liệu qua các "kênh" gọi là **topic**. |
| **Topic** | Kênh dữ liệu trong ROS 2, ví dụ `/camera/color/image_raw` (ảnh), `/odom` (vị trí). |
| **`tf`** | Topic chuyên phát **pose** của mọi bộ phận robot theo thời gian. |
| **`camera_info`** | Topic phát **intrinsic `K`** của camera. |
| **SLAM / Odometry / VIO** | Các kỹ thuật giúp robot **tự biết nó đang ở đâu** khi di chuyển (dựa vào camera/bánh xe/IMU). Không có chúng thì không có pose ngoài đời thật. |
| **`.mcap`** | **Định dạng "hộp đen"** lưu log nhiều luồng dữ liệu có mốc thời gian (chuẩn mới thay `.bag` của ROS). Bên trong gồm nhiều *channel* (topic), mỗi channel gắn một *schema* (mô tả hình dạng message), mỗi message có `log_time` tính bằng nanosecond. Mở được bằng **Foxglove Studio** hoặc thư viện `mcap` của Python. |
| **Schema (trong mcap)** | Bản mô tả "message này có những trường gì". Nhờ nó file **tự mô tả** — đọc được mà không cần cài ROS. |
| **Đồng bộ thời gian (time sync)** | Các luồng có tần số khác nhau (ảnh 10 Hz, pose 50 Hz) → phải **căn giờ** để ghép đúng ảnh với đúng pose. Ghép lệch → nhãn sai mà **không có thông báo lỗi**. |
| **DepthAnything(V2)** | Model AI **đoán depth từ ảnh màu thường**. Dùng khi camera không có depth. Kết quả gần đúng, kém camera RGB-D thật. |

---

## 5. Nhóm định dạng file dữ liệu

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **LeRobotDataset (v2.1)** | **Quy chuẩn sắp xếp thư mục** cho dữ liệu robot (do HuggingFace đề ra). Cả 3 bộ data của N1 đều gói theo chuẩn này. **Mỗi scene = một dataset độc lập**. |
| **Parquet** | Định dạng file bảng (giống Excel nhưng nén, nhanh — đuôi `.parquet`). Lưu con số của từng frame. Đọc bằng `pandas.read_parquet`. |
| **`chunk-XXX`** | Thư mục nhóm ≤ 1000 episode. `chunk = episode_index // 1000`. Thuần mẹo kỹ thuật để thư mục khỏi quá tải ([07](07_phu_luc_lerobot_format.md) mục 1). |
| **`meta/`** | Thư mục mô tả: `episodes.jsonl` (câu lệnh + độ dài), `tasks.jsonl` (bảng tra câu lệnh), `episodes_stats.jsonl` (thống kê), `info.json` (kê khai schema). |
| **`setting`** | Chuỗi `{chiều_cao}cm_{góc_cúi}deg` (vd `125cm_30deg`) — **định danh cấu hình camera** và là **hậu tố tên cột/thư mục** trong data S2. Công thức trong code: `f'{height}cm_{pitch_2}deg'` ([internvla_n1_lerobot_dataset.py:850](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L850)). |
| **`info.json`** | File tự khai báo cấu trúc dataset. ⚠️ Trong bộ N1 file này **hay khai sai** — nguyên tắc: *tin dữ liệu thật (`df.dtypes`), không tin `info.json`*. |
| **`<INSTRUCTION_SEP>`** | Chuỗi ngăn cách khi một episode có nhiều câu lệnh. Loader tách bằng `.split("<INSTRUCTION_SEP>")` ([dòng 770](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L770)). |

---

## 6. Bảng mã "action" của System 2 (hay tra nhất)

Định nghĩa tại [internvla_n1_lerobot_dataset.py:950](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L950):

| Giá trị | Ký hiệu model xuất ra | Nghĩa | Độ lớn thật (đo trên `vln_ce`) |
|---|---|---|---|
| `0` | `STOP` | Dừng — đã tới đích | — |
| `1` | `↑` | Tiến | **0.25 m** |
| `2` | `←` | Quay trái | **15°** |
| `3` | `→` | Quay phải | **15°** |
| `5` | `↓` | Cúi nhìn xuống (để thấy sàn rồi chấm đích) | — |
| `-1` | *(không xuất)* | **Frame khởi đầu** — chỉ là mốc, **loại khỏi thống kê độ chính xác** | — |

> Cách đo: chạy `get_trajectory_relative_to_frame()` trên `pose.60cm_30deg` của
> `r2r/17DRP5sb8fy/episode_000000` → chuỗi 3 lần action `3` cho ra yaw `-0.262 rad = -15°` mỗi bước,
> action `1` cho ra bước tiến `0.25 m`.

---

Xong phần thuật ngữ. Tiếp theo: bức tranh lớn của hệ thống → [02_he_thong](02_he_thong.md).
