# Bộ tài liệu: Dữ liệu & huấn luyện InternVLA-N1 (bản dành cho người mới học ML)

> **Bộ tài liệu này trả lời trọn vẹn 6 câu hỏi:**
> 1. Các thuật ngữ trong dự án nghĩa là gì? → [01](01_thuat_ngu.md)
> 2. Hệ thống InternVLA-N1 hoạt động ra sao, gồm những mảnh nào? → [02](02_he_thong.md)
> 3. Code huấn luyện **nhánh System 2** chạy thế nào, từng phần làm gì? → [03](03_code_train_s2.md)
> 4. Data train **System 2** có cấu trúc gì, **cái nào bắt buộc / không bắt buộc**? → [04](04_data_train_s2.md)
> 5. Data train **System 1** có cấu trúc gì, **cái nào bắt buộc / không bắt buộc**? → [05](05_data_train_s1.md)
> 6. Làm sao **sinh data S2 từ file `.mcap`** của robot? → [06](06_pipeline_mcap_to_s2.md)
>
> Mọi khẳng định đều **đối chiếu code thật** trong `InternNav/code/` (kèm `file:line` để tự kiểm
> chứng) và **đo thật** trên dữ liệu có sẵn tại `InternNav/data/vln_ce/traj_data/r2r/17DRP5sb8fy`.
> Chỗ nào là suy luận/đề xuất sẽ được nói rõ là suy luận/đề xuất.

---

## Mục lục

| File | Nội dung | Dành cho lúc bạn muốn… |
|---|---|---|
| [01_thuat_ngu](01_thuat_ngu.md) | Từ điển: ML, camera & hình học 3D, robot/ROS, định dạng file | …tra một từ lạ |
| [02_he_thong](02_he_thong.md) | Hai bộ não, 3 bộ data con, model nào–config nào–script nào | …hiểu bức tranh lớn |
| [03_code_train_s2](03_code_train_s2.md) | **Mổ xẻ code nhánh train S2**: trainer → argument → dataset → collator | …đọc/sửa code |
| [04_data_train_s2](04_data_train_s2.md) | Cấu trúc data S2 + **bảng bắt buộc/không bắt buộc** | …chuẩn bị data S2 |
| [05_data_train_s1](05_data_train_s1.md) | Cấu trúc data S1 + **bảng bắt buộc/không bắt buộc** | …chuẩn bị data S1 |
| [06_pipeline_mcap_to_s2](06_pipeline_mcap_to_s2.md) | **`.mcap` → data S2**, 6 giai đoạn, có code chạy được | …bắt tay làm data thật |
| [07_phu_luc_lerobot_format](07_phu_luc_lerobot_format.md) | Phụ lục: `chunk`, `parquet`, 4 file `meta/` — giải nghĩa từng trường | …hiểu file vật lý |
| [08_phu_luc_thu_thap_data](08_phu_luc_thu_thap_data.md) | Phụ lục: thu data bằng simulator / RGB-D / điện thoại, lộ trình, bẫy | …chọn thiết bị & lộ trình |

**Công cụ kèm theo** (thư mục [tools/](tools/)):

| Script | Việc nó làm | Đã chạy thử? |
|---|---|---|
| [tools/generate_s2_mcap.py](tools/generate_s2_mcap.py) | Sinh một file `.mcap` **chứa đủ dữ liệu để dựng data S2** (2 luồng RGB + depth + camera_info + pose + câu lệnh) | ✅ 78 frame, 3.8 MB |
| [tools/mcap2s2.py](tools/mcap2s2.py) | Chuyển `.mcap` → dataset LeRobot đúng chuẩn `vln_ce` | ✅ sinh 2 episode, **loader thật đọc được** |

---

## Tóm tắt 60 giây

InternVLA-N1 là mô hình điều hướng **nghe lệnh bằng lời** rồi tự đi tới đích. Nó có **hai bộ não**:

```
Câu lệnh: "đi dọc hành lang, rẽ phải ở cây cột xanh, dừng cạnh cột vàng"
        │
        ▼
┌──────────────────────────┐   nhìn ảnh + đọc lệnh
│  SYSTEM 2 — "nghĩ chậm"  │ → chấm MỘT ĐIỂM trên ảnh (pixel goal)
│  Qwen2.5-VL 7B (VLM)     │ → hoặc xuất ←/→ (xoay) hoặc STOP
└──────────────────────────┘ → kèm một "gói tín hiệu" (latent) cho S1
        │
        ▼
┌──────────────────────────┐   nhận điểm đích + ảnh + depth
│  SYSTEM 1 — "phản xạ"    │ → vẽ ĐƯỜNG ĐI cong né vật cản
│  NextDiT / NavDP         │ → ra lệnh bánh xe
└──────────────────────────┘
        │
        ▼   robot đi → chụp ảnh mới → lặp lại
```

Hai bộ não **học hai bài toán khác nhau** nên ăn **hai loại data khác nhau**:

| | **System 2** | **System 1** |
|---|---|---|
| Bộ data gốc | `vln_ce` | `vln_n1` |
| Nhãn cốt lõi | **pixel goal `[u,v]`** + action rời rạc | **quỹ đạo camera** (ma trận 4×4/frame) |
| Người phải chú thích? | ✅ Có — viết câu lệnh tiếng Anh | ❌ Không — tự suy từ quỹ đạo |
| Thứ khó kiếm nhất | câu lệnh + biết đích ở pixel nào | **bản đồ 3D vật cản** (`pointcloud.ply`) |
| Thu bằng thiết bị thường? | ✅ Khả thi | ❌ Khó |

👉 **Kết luận thực dụng:** nếu bạn có log robot (`.mcap`) thu ngoài đời thật, hãy **làm data System 2
trước** — nó khả thi, và là phần "thông minh" nhất của hệ. Đường đi cụ thể: [06](06_pipeline_mcap_to_s2.md).

---

## Ba sự thật hay bị hiểu nhầm nhất

1. **Không có "một dataset của InternVLA-N1".** Có **3 bộ con** (`vln_ce`, `vln_n1`, `vln_pe`) nuôi
   **3 model khác nhau**, mỗi bộ có một loader riêng và **không thay thế được cho nhau**
   ([02](02_he_thong.md) mục 3).
2. **Không bao giờ "train từ đầu".** Cả hai script train đều nạp trọng số có sẵn bằng
   `from_pretrained(...)` — luôn là **fine-tune** ([03](03_code_train_s2.md) mục 7).
3. **Chữ "action" có hai nghĩa.** Với S2 là **số nguyên** `{0,1,2,3,5}`; với S1 là **ma trận 4×4**
   (quỹ đạo liên tục). Cùng tên, khác hẳn nhau ([02](02_he_thong.md) mục 5).

---

## Liên kết tới tài liệu kỹ thuật khác trong repo

- [../handbook/03_data_contract.md](../handbook/03_data_contract.md) — hợp đồng dữ liệu (bản đo thật, cô đọng).
- [../handbook/02_code_structure.md](../handbook/02_code_structure.md) — cấu trúc code + chữ ký hàm phía inference.
- [../io_system2.md](../io_system2.md) — vào/ra của System 2 lúc chạy thật.
- Code loader thật: `InternNav/code/internnav/dataset/internvla_n1_lerobot_dataset.py` (S2),
  `.../navdp_lerobot_dataset.py` (S1).

*Bộ tài liệu mô tả code tại thời điểm phân tích (27/07/2026). Nếu code đổi, hãy kiểm lại các `file:line` được trích.*
