# 04 — Checklist buổi quay (quay sai thì hai pipeline vô nghĩa)

> Nguyên tắc: **SVO2 là thứ duy nhất không làm lại được.** Mọi lỗi hậu kỳ sửa được bằng code;
> lỗi lúc quay (sai resolution, quên đo chiều cao, tracking chết giữa chừng) là mất buổi quay.

---

## 1. Trước buổi quay (làm MỘT lần, chốt rồi không đổi)

- [ ] Chạy script kiểm kê camera ([01](01_zed_output.md) mục 5) → lưu `metadata/camera_info.json`.
- [ ] **Chốt 3 con số định danh setting** (không đổi giữa các buổi — đổi là chia đôi dataset):
  - [ ] Chiều cao camera khi đội (đo bằng thước, từng người quay): `______ cm`
  - [ ] Góc cúi pitch (đọc IMU khi người quay đứng thẳng nhìn tự nhiên): `______ °` — mục tiêu ~30°, **cấm ≈ 0°** ([03](03_pipeline_s2_vln_ce.md) mục 4)
  - [ ] Resolution/FPS: HD720@30 (hoặc số đã chốt sau khi đo biến thể `_zed` — [02](02_pipeline_s1_vln_n1.md) mục 4)
- [ ] Gắn camera **chắc** (mũ bảo hiểm/headstrap vặn vít, không dây thun lỏng) — pitch trôi giữa buổi là hỏng nhãn setting.
- [ ] Thẻ nhớ/SSD đủ chỗ: ước tính dung lượng ghi thử 1 phút SVO2 H.265 × tổng phút dự kiến × 1.5.
- [ ] Pin laptop/battery pack cho cả buổi; kiểm tra nhiệt (ZED throttle khi nóng).
- [ ] Kịch bản tuyến: liệt kê trước các hành trình (điểm đầu → điểm cuối có nghĩa: "cửa thang máy → phòng họp lớn"), mỗi tuyến đi 2 chiều = 2 episode.

## 2. Trong buổi quay (mỗi episode)

- [ ] Bấm record **trước khi đội**, đứng yên 3 giây sau khi đội (IMU init + lấy mốc pitch).
- [ ] Một episode = một hành trình liên tục, 30 giây – 2 phút. Dừng record giữa các episode
      (file SVO2 nhỏ, mất 1 episode không mất cả buổi).
- [ ] Đi tốc độ tự nhiên nhưng: **xoay người thay vì quắc đầu**, không vừa đi vừa nhìn điện thoại,
      không đứng nói chuyện giữa episode.
- [ ] Ngay sau mỗi episode: **nói mô tả tuyến bằng lời** (ghi âm 15 giây) — nguyên liệu viết
      instruction ([03](03_pipeline_s2_vln_ce.md) mục 6), để hôm sau không phải nhớ lại.
- [ ] Ghi sổ: mã episode · tuyến · người quay · bất thường (đám đông, cửa kính, đèn tắt).

## 3. Đa dạng hoá (giá trị dataset nằm ở đây)

- [ ] Nhiều khu chức năng: hành lang, khu bàn làm việc, pantry, phòng họp, sảnh thang máy.
- [ ] Nhiều điều kiện: sáng/chiều (ánh sáng cửa sổ đổi), đông người/vắng người.
- [ ] Nhiều kiểu tuyến: thẳng dài, nhiều khúc rẽ, qua cửa, vòng qua vật cản.
- [ ] Có cả tuyến "nhàm chán" (hành lang trống) — S1 cần chúng để học đi thẳng ổn định.

## 4. Sau buổi quay (trước khi xoá thẻ nhớ — bài test chất lượng)

- [ ] Replay từng SVO2: tracking `OK` xuyên suốt? Đoạn `SEARCHING/OFF` → ghi chú cắt bỏ.
- [ ] **Test drift:** với episode có quay về điểm xuất phát, đo khoảng cách pose đầu–cuối.
      ⬜ Chưa có ngưỡng đo thật; tạm lấy **< 1% chiều dài quỹ đạo** làm mức chấp nhận, đo vài
      buổi đầu rồi chốt số thật vào đây.
- [ ] Xem nhanh depth 5 frame ngẫu nhiên/episode: kính, gương có tạo vùng rác lớn không.
- [ ] Backup SVO2 + sổ ghi chú + file ghi âm về ≥ 2 nơi, đặt tên
      `YYYYMMDD_<khu>_<tuyến>_<người>.svo2`.

## 5. Bản đồ tài liệu — quay xong rồi làm gì

```
SVO2 đã backup
 ├─▶ pipeline S1 (tự động):   02_pipeline_s1_vln_n1.md   → data train NavDP
 └─▶ pipeline S2 (bán tự động): 03_pipeline_s2_vln_ce.md → data train S2 (cần viết instruction)
      hai việc ⬜ chặn đường, làm TRƯỚC khi sinh hàng loạt:
      1. đo file vln_n1 biến thể _zed thật (intrinsic + đơn vị depth + nhịp frame)  [02 mục 3–4]
      2. xác minh hệ trục pose bằng phép thử chiếu quỹ đạo                          [02 mục 5]
```
