# 06 — Lộ trình bắt đầu & các bẫy thường gặp

> **File này để làm gì:** cho bạn — người mới — một **thứ tự việc cần làm** để không sa lầy, cùng danh
> sách **bẫy dữ liệu** đã được kiểm chứng trong code. Đọc file này sau khi đã nắm 01–05.
>
> Bộ tài liệu: [00_README](00_README.md) · [05_thu_thap_data](05_thu_thap_data.md) ·
> handbook gốc: [../handbook/03_data_contract.md](../handbook/03_data_contract.md)

---

## 1. Nguyên tắc vàng cho người mới

> **Đừng cố làm cả hai hệ thống cùng lúc bằng thiết bị thường.** System 1 cần bản đồ 3D (khó);
> System 2 chỉ cần ảnh + lệnh + điểm đích (khả thi). Hãy **làm data System 2 trước** — đó là phần
> "thông minh" nhất và cho kết quả nhìn thấy được sớm nhất.

---

## 2. Lộ trình đề xuất (theo mức độ, làm dần)

### Mức 0 — Hiểu, chưa tạo gì (1–2 ngày)
1. Đọc xong bộ tài liệu này (01→05).
2. Tải **một scene nhỏ** của `vln_ce` (file nhỏ nhất repo chỉ ~16 MB) về xem tận mắt.
3. Mở file `.parquet` bằng `pandas`, in ra `df.dtypes` và vài hàng đầu → **đối chiếu với bảng cột ở
   file 03**. Mục tiêu: "sờ" được dữ liệu thật, thấy đúng như mô tả.

### Mức 1 — Tạo data System 2 quy mô nhỏ (khả thi với điện thoại)
1. Quay 5–10 lượt đi ngắn trong nhà/văn phòng bằng điện thoại (bật ARKit/ARCore để có pose).
2. Với mỗi lượt: viết 1 câu lệnh tiếng Anh mô tả.
3. Chạy DepthAnythingV2 để sinh depth; suy `action` từ chuyển động; chấm/chiếu `pixel goal`.
4. Đóng gói theo chuẩn LeRobotDataset giống `vln_ce`.
5. **Kiểm tra:** nạp thử bằng loader `NavPixelGoalDataset` — nếu chạy không lỗi là format đúng.

### Mức 2 — Tạo data quy mô lớn bằng simulator (nếu muốn train thật)
1. Cài Habitat-Sim, load vài scene Matterport3D/HM3D.
2. Sinh hàng nghìn episode tự động → data S2 **và** S1 chuẩn.
3. Đây là con đường để có đủ dữ liệu train nghiêm túc.

### Mức 3 — System 1 (chỉ khi đã vững)
- Ưu tiên dùng checkpoint NavDP có sẵn thay vì tự train.
- Nếu tự tạo: làm bằng simulator (có bản đồ 3D sẵn), không nên bằng thiết bị thường.

---

## 3. Checklist bẫy dữ liệu (kiểm trước khi tin kết quả)

Tổng hợp từ code loader thật và handbook — **tick từng ô** khi tự tạo data:

- [ ] **Đơn vị depth khác nhau giữa 2 hệ:** System 2 chia depth cho **1000**, System 1 chia cho
      **10000**. Tạo data cho hệ nào thì theo đúng hằng số hệ đó (file 03 mục 4, file 04 mục 4).
- [ ] **Pixel goal là `[u, v] = [cột, hàng]`.** Khi so với đầu ra model (thường là `[hàng, cột]`)
      phải **đảo lại**, và chú ý ảnh model dùng là 384×384 còn data là 640×480 → phải scale.
- [ ] **`action = -1` là frame khởi đầu** (chỉ của `vln_ce`) → **loại bỏ** khi tính độ chính xác.
- [ ] **"action" của S2 (số nguyên) khác "action" của S1 (ma trận 4×4)** → đừng nhầm hai thứ.
- [ ] **Tránh cấu hình camera `125cm_0deg` đơn** (nhìn thẳng, không thấy sàn → goal toàn `-1`).
- [ ] **Tin dữ liệu thật (`df.dtypes`), không tin `info.json`** — file khai báo hay sai trong bộ N1.
- [ ] **Dùng cột `timestamp` để tính thời gian**, không suy từ `fps` (giá trị `fps` hay là số mẫu sai).
- [ ] **Mỗi scene là một dataset độc lập** — khi gộp nhiều scene phải tự quản lý chỉ số (index).
- [ ] **RGB của `vln_ce` khi train là `.jpg`** (không phải `.png`), depth mới là `.png`. Đặt đúng đuôi.
- [ ] **System 1 bắt buộc có `pointcloud.ply`** — thiếu nó thì phần critic (né vật cản) không train được.

---

## 4. Sai lầm tư duy hay gặp (và cách sửa)

| Nghĩ sai | Thực tế |
|---|---|
| "Có một dataset N1, cứ đổ hết vào train." | Có **3 bộ con** cho **3 model khác nhau**, không thay thế được nhau. |
| "Cứ có ảnh là train được." | Điều hướng cần **pose** (camera ở đâu) và với S1 là **bản đồ 3D** — ảnh không thôi là chưa đủ. |
| "Điện thoại quay là xong data." | Điện thoại cho RGB (+pose qua ARKit). Depth phải *ước lượng*, pixel goal & câu lệnh phải *tạo thêm*. |
| "Train S1 và S2 giống nhau." | Hai pipeline, hai loader, hai loại label hoàn toàn khác. |
| "info.json nói sao thì đúng vậy." | `info.json` trong bộ N1 **hay sai** — luôn kiểm bằng dữ liệu thật. |

---

## 5. Bước tiếp theo — bạn có thể nhờ tôi làm gì

Khi sẵn sàng viết code, ba việc khả thi ngay (xếp theo thứ tự nên làm):

1. **Script sinh data System 2** từ một video RGB + câu lệnh (điện thoại/ROS2) → xuất đúng format
   `vln_ce` LeRobotDataset. *(Khuyến nghị bắt đầu ở đây.)*
2. **Node ROS2** thu đồng bộ RGB + depth + `tf` + `camera_info` → `.parquet`.
3. **Script sinh data System 1** từ Habitat/Isaac Sim → format `vln_n1` (kèm `pointcloud.ply`).

Cứ nói việc bạn muốn, tôi sẽ viết code kèm giải thích từng dòng theo đúng phong cách "giải thích chậm"
của bộ tài liệu này.

---

*Hết bộ tài liệu. Quay lại mục lục: [00_README](00_README.md).*
