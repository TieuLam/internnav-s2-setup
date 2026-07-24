# 01 — ZED đội đầu xuất ra cái gì?

> ⬜ **Trạng thái:** viết từ spec/tài liệu ZED SDK (chưa có camera trên tay). Khi nhận máy,
> việc đầu tiên là chạy đoạn "kiểm kê" ở mục 5 và sửa lại file này bằng số đo thật —
> đúng tinh thần handbook: *khai báo mâu thuẫn dữ liệu thì dữ liệu thắng*.

---

## 1. Đầu ra gốc: MỘT file `.svo2` cho mỗi lần bấm quay

ZED không ghi ra "ảnh + depth" rời rạc. Khi record, SDK ghi **file SVO2** — container riêng
của Stereolabs, bên trong có:

| Thành phần | Chi tiết |
|---|---|
| **Video stereo nén** | 2 luồng ảnh **trái + phải** đã đồng bộ từng frame (H.264/H.265 hoặc lossless) |
| **Dữ liệu IMU** | accelerometer + gyroscope ~400 Hz (ZED 2i thêm barometer, magnetometer, cảm biến nhiệt) |
| **Timestamp** | nanosecond cho từng frame ảnh và từng mẫu IMU |
| **Calibration nhúng** | intrinsic 2 mắt + baseline + distortion, theo từng resolution (calib factory của đúng serial máy) |

**Điểm quyết định toàn bộ pipeline:** SVO2 **replay được qua SDK y như camera live**. Nghĩa là
depth, pose, point cloud **không cần tính lúc quay** — về nhà mở file, tính lại với chế độ depth
tốt nhất (NEURAL), chạy đi chạy lại đến khi vừa ý. Ngược lại, thứ gì không nằm trong SVO2
(cân chỉnh sai resolution, quên bật ghi IMU) thì **mất vĩnh viễn**.

> Hệ quả thực dụng: buổi quay chỉ có MỘT nhiệm vụ kỹ thuật — ghi SVO2 đúng cấu hình.
> Mọi thứ phức tạp làm offline. Checklist quay: [04_recording_checklist.md](04_recording_checklist.md).

---

## 2. Thông tin derive được offline từ SVO2 (qua ZED SDK)

Replay SVO2 rồi gọi API, mỗi frame lấy được:

| Thông tin | API (sl.Mat / sl.Pose) | Dạng | Dùng cho |
|---|---|---|---|
| **RGB trái (rectified)** | `retrieve_image(VIEW.LEFT)` | uint8 H×W×4 (BGRA) | ảnh input của cả S1 lẫn S2 |
| RGB phải | `VIEW.RIGHT` | uint8 | thường không cần (giữ trong SVO2 phòng hờ) |
| **Depth** | `retrieve_measure(MEASURE.DEPTH)` | **float32, MÉT** (unit đặt được) | depth input S1/S2 — phải đổi đơn vị khi ghi ra, xem [02](02_pipeline_s1_vln_n1.md) mục 3 |
| Confidence map | `MEASURE.CONFIDENCE` | float32 | lọc pixel depth rác trước khi ghi |
| Point cloud | `MEASURE.XYZRGBA` | float32 H×W×4 | kiểm tra chất lượng, không bắt buộc cho 2 format đích |
| **Pose camera 6-DoF** | `get_position()` (Positional Tracking) | translation + quaternion → **ma trận 4×4** | nguồn của `camera_extrinsic`, `pose.{setting}` và **GT quỹ đạo của S1** |
| **Intrinsic** | `camera_information.camera_configuration.calibration_parameters.left_cam` | fx, fy, cx, cy + disto | cột `camera_intrinsic` + phép chiếu pixel goal |
| IMU raw/fused | `get_sensors_data()` | 400 Hz | tracking đã tự dùng; giữ raw để debug drift |

**Positional tracking = VIO (ảnh + IMU).** Bật được 2 tuỳ chọn quan trọng:
- `enable_area_memory` — loop closure, giảm drift khi đi vòng quanh văn phòng rồi quay lại;
- `enable_pose_smoothing` — làm mượt cú giật khi loop closure sửa pose.

⬜ Drift thực tế trong văn phòng (sàn bóng, tường kính) chưa đo — mục 4 của
[04_recording_checklist](04_recording_checklist.md) có bài test loop để đo con số này.

---

## 3. Những đặc thù của "đội đầu" ảnh hưởng trực tiếp đến data

| Đặc thù | Hệ quả | Đối sách |
|---|---|---|
| **Chiều cao camera ~155–175 cm** (tuỳ người đội) | Data gốc InternData-N1 quay ở **125 cm và 60 cm** ([handbook 03](../handbook/03_data_contract.md) mục 1.2) → lệch phân bố chiều cao so với data train gốc | Đo và ghi chiều cao thật của từng người quay; cân nhắc đội thấp/gắn ngực nếu muốn gần 125 cm. **Không được khai man con số này** — nó nằm trong tên setting |
| **Pitch dao động theo bước chân, gật đầu** | Data sim có pitch cố định tuyệt đối; data thật pitch ±5–10° quanh giá trị gắn | Ghi pitch danh nghĩa (đo bằng IMU lúc đứng yên); chấp nhận nhiễu, KHÔNG cố ổn định bằng gimbal (đổi phân bố motion) |
| **Xoay đầu nhanh → motion blur** (ZED 2i là rolling shutter) | Frame mờ → depth stereo sai, tracking trượt | Dặn người quay xoay người thay vì quắc đầu; lọc frame blur ở hậu kỳ |
| **Người quay là human, không phải robot** | Quỹ đạo mượt, không có bước rời rạc 25cm/15° như action space của S2 | Đây là vấn đề trung tâm của pipeline S2 — [03](03_pipeline_s2_vln_ce.md) mục 3 |
| Gương, vách kính văn phòng | Stereo depth "xuyên" qua kính, trả giá trị sai | Ghi chú vị trí kính trong metadata buổi quay; lọc bằng confidence map |

---

## 4. Chốt cấu hình ghi hình (đề xuất — chốt lại sau khi đo máy thật)

| Tham số | Đề xuất | Lý do |
|---|---|---|
| Resolution | **HD720 (1280×720)** | đủ downsample về 640×480 (S2) và ~480×270 (S1); HD2K chỉ 15 fps, VGA thì thiếu pixel |
| FPS | **30** | khớp nhịp sinh frame của cả 2 format; thừa thì subsample |
| Depth mode | quay: không cần bật · replay: **NEURAL** | depth tính offline (mục 1) |
| Depth unit khi replay | **METER** (mặc định) | tự quy đổi khi ghi ra từng format — 2 format đích dùng 2 đơn vị KHÁC NHAU (bẫy số 1 của toàn nhiệm vụ, xem [02](02_pipeline_s1_vln_n1.md) mục 3) |
| SVO2 compression | H.265 | buổi quay dài; lossless chỉ khi thẻ nhớ dư dả |
| Positional tracking | bật khi replay, `enable_area_memory=True` | pose là GT của S1 |

⬜ **Việc phải làm trước khi chốt:** tải 1 file `vln_n1/traj_data/*_zed/*.tar.gz` (~250 MB) và
đo intrinsic biến thể `_zed` — handbook mới đo `_d435i` (480×270, FOV 68°×42°). Biết ZED
mô phỏng của họ dùng resolution/FOV nào thì mình crop/scale data thật cho khớp phân bố.

---

## 5. Đoạn "kiểm kê" chạy ngay khi nhận camera

```python
import pyzed.sl as sl

zed = sl.Camera()
init = sl.InitParameters()                       # mặc định trước, đọc ra rồi mới chỉnh
err = zed.open(init)
assert err == sl.ERROR_CODE.SUCCESS, err

info = zed.get_camera_information()
print("Model        :", info.camera_model)        # ZED 2i? ZED X? — ghi vào metadata
print("Serial       :", info.serial_number)
print("Resolution   :", info.camera_configuration.resolution.width,
                        info.camera_configuration.resolution.height)
print("FPS          :", info.camera_configuration.fps)
calib = info.camera_configuration.calibration_parameters.left_cam
print("fx fy cx cy  :", calib.fx, calib.fy, calib.cx, calib.cy)
print("FOV h/v      :", calib.h_fov, calib.v_fov)
print("Baseline     :", info.camera_configuration.calibration_parameters.get_camera_baseline())
```

Kết quả in ra → dán vào file này thay cho các con số ⬜, và lưu thành
`metadata/camera_info.json` trong repo data.
