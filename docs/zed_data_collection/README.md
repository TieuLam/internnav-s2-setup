# Thu thập dữ liệu văn phòng bằng ZED camera đội đầu → data train S1/S2

> **Folder này để làm gì:** trả lời 3 câu hỏi cho nhiệm vụ thu data thật bằng ZED đội đầu:
> (1) camera xuất ra cái gì, (2) trong đó có thông tin gì, (3) biến data thô đó thành data
> train được System 1 (NavDP) và System 2 (Qwen-VL pixel-goal) như thế nào.
>
> Viết theo cùng nguyên tắc với [handbook](../handbook/03_data_contract.md): **mọi yêu cầu về
> format đích đều đối chiếu với code loader thật** (`internnav/dataset/*.py`, đọc 23/07/2026),
> không chép từ dataset card. Chỗ nào chưa kiểm chứng được đánh dấu ⬜.

## Mục lục

| File | Nội dung |
|---|---|
| [01_zed_output.md](01_zed_output.md) | ZED đội đầu ghi ra cái gì (SVO2) · từng loại thông tin bên trong · cái gì derive offline được |
| [02_pipeline_s1_vln_n1.md](02_pipeline_s1_vln_n1.md) | Raw ZED → format `vln_n1` để train **S1** — pipeline tự động 100%, làm trước |
| [03_pipeline_s2_vln_ce.md](03_pipeline_s2_vln_ce.md) | Raw ZED → format `vln_ce` để train **S2** — cần pseudo-label + viết instruction, khó hơn nhiều |
| [04_recording_checklist.md](04_recording_checklist.md) | Checklist trước/trong/sau buổi quay — quay sai thì hai pipeline trên vô nghĩa |

## Tóm tắt 30 giây

1. **Chỉ cần ghi file `.svo2`** khi đi quay. SVO2 chứa video stereo nén + IMU + calibration;
   depth, pose, point cloud đều **tính lại offline** từ nó bằng ZED SDK — quên bật depth lúc
   quay không sao, quên bật IMU/chọn sai resolution là hỏng.
2. **S1 dễ, S2 khó.** S1 (NavDP) học "đi tới goal tránh vật cản" — GT là chính quỹ đạo camera,
   lấy thẳng từ positional tracking của ZED, không cần người gán nhãn. Format đích `vln_n1`
   thậm chí có sẵn biến thể camera **`zed`** (mô phỏng) — data thật của mình khớp phân bố.
3. **S2 cần 3 thứ ZED không tự có:** action rời rạc `{1,2,3,5}` (phải discretize từ quỹ đạo),
   pixel goal `[u,v]` (phải chiếu waypoint tương lai vào ảnh), và **instruction tiếng Anh**
   (phải người viết/duyệt). Ngoài ra format `vln_ce` giả định render được **2 góc pitch cùng
   lúc** (ngang + cúi) — camera thật không làm được, phải chọn phương án thỏa hiệp
   (xem [03](03_pipeline_s2_vln_ce.md) mục 4).
4. **Ba con số phải chốt trước buổi quay đầu tiên:** chiều cao camera (cm), góc cúi (deg),
   resolution/fps. Chúng đóng đinh vào tên cột `{height}cm_{pitch}deg` của data S2 và phân bố
   FOV của data S1 — đổi giữa chừng là chia đôi dataset.

## Nguồn đối chiếu

- Format đích: [handbook 03_data_contract](../handbook/03_data_contract.md) (schema đo thật trên file tải về).
- Loader S1: `InternNav/code/internnav/dataset/navdp_lerobot_dataset.py` (cột parquet: dòng 198–202; xử lý depth: 178–192).
- Loader S2: `InternNav/code/internnav/dataset/internvla_n1_lerobot_dataset.py` (setting: 850; đường dẫn ảnh: 1015–1021) và `vlln_lerobot_dataset.py` (ý nghĩa pitch_1/pitch_2: 18–37, 325–329, 577).
- Thông số ZED: spec ZED SDK (⬜ chưa đo trên máy thật — khi nhận camera phải in `camera_information` ra đối chiếu).
