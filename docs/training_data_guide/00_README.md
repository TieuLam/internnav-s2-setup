# 00 — README: Hướng dẫn tạo dữ liệu huấn luyện cho InternVLA-N1 (bản cho người mới)

> **Bộ tài liệu này để làm gì:** giải thích **từ đầu, cho người chưa biết gì về robot và AI**, rằng
> muốn *huấn luyện* (train) mô hình điều hướng InternVLA-N1 thì cần **loại dữ liệu nào**, dữ liệu đó
> **có cấu trúc ra sao**, và **thu thập ở đâu / bằng thiết bị gì** (từ simulator, camera chuyên dụng,
> cho tới camera điện thoại hoặc camera gắn trên robot dùng ROS2).
>
> Đây là bản "giải thích chậm" song song với bộ [handbook](../handbook/) (bản kỹ thuật cô đọng cho
> người đã quen việc). Mọi thứ trong tài liệu này đều **đối chiếu với code thật** trong repo
> (`internnav/dataset/*.py`, `scripts/train/...`), không phải chép từ mô tả trên mạng.

---

## Ai nên đọc và đọc theo thứ tự nào

Bạn **mới bắt đầu** → đọc lần lượt từ 01 đến 06. Mỗi file dựa trên file trước.

| File | Nội dung | Trả lời câu hỏi |
|---|---|---|
| [01_nhap_mon_thuat_ngu](01_nhap_mon_thuat_ngu.md) | Từ điển khái niệm AI + robot cơ bản | "Train, model, RGB-D, pose, point cloud… là gì?" |
| [02_hai_he_thong](02_hai_he_thong.md) | Vì sao N1 có **2 bộ não** và mỗi bộ ăn data khác nhau | "Tại sao không có *một* loại data?" |
| [03_data_cho_system2](03_data_cho_system2.md) | Data cho **System 2** (bộ não "suy nghĩ") | "S2 cần gì, cấu trúc thế nào?" |
| [04_data_cho_system1](04_data_cho_system1.md) | Data cho **System 1** (bộ não "phản xạ") | "S1 cần gì, cấu trúc thế nào?" |
| [05_thu_thap_data](05_thu_thap_data.md) | Thu thập bằng simulator / camera chuyên dụng / điện thoại / ROS2 | "Tôi lấy data này ở đâu, bằng thiết bị nào?" |
| [06_lo_trinh_bat_dau](06_lo_trinh_bat_dau.md) | Nên làm gì trước, làm gì sau; các bẫy thường gặp | "Bắt đầu từ đâu cho đỡ sa lầy?" |

---

## Tóm tắt 30 giây (đọc xong sẽ hiểu vì sao)

InternVLA-N1 là robot điều hướng **nghe lệnh bằng lời** (kiểu *"đi tới cửa nhà thờ rồi dừng giữa
các bậc thang"*) rồi tự đi tới đích. Nó có **2 bộ não**:

- **System 2** — bộ não *chậm mà khôn*: nhìn ảnh + đọc câu lệnh → chỉ ra **một điểm trên ảnh** cần
  đi tới (gọi là *pixel goal*).
- **System 1** — bộ não *nhanh mà bản năng*: nhận điểm đích đó → vẽ ra **đường đi cong liên tục**
  để né bàn ghế, tường.

Muốn *dạy* (train) 2 bộ não này, bạn cần **2 bộ dữ liệu khác nhau**:

| | System 2 | System 1 |
|---|---|---|
| Bộ dữ liệu gốc | `vln_ce` | `vln_n1` |
| Cốt lõi phải có | ảnh + câu lệnh + **điểm đích trên ảnh** | ảnh + **đường đi thật (quỹ đạo)** + **bản đồ 3D vật cản** |
| Cần người chú thích? | Có (viết câu lệnh) | Không (tự suy từ đường đi) |

Chi tiết từng thứ nằm ở các file sau. Bắt đầu với [01_nhap_mon_thuat_ngu](01_nhap_mon_thuat_ngu.md).

---

## Liên kết tới tài liệu kỹ thuật gốc (khi bạn đã vững)

- [../handbook/03_data_contract.md](../handbook/03_data_contract.md) — hợp đồng dữ liệu (bản đo thật).
- [../handbook/02_code_structure.md](../handbook/02_code_structure.md) — cấu trúc code, chữ ký hàm.
- Code loader thật: `internnav/dataset/internvla_n1_lerobot_dataset.py` (S2),
  `internnav/dataset/navdp_lerobot_dataset.py` (S1).
