# Bộ tài liệu: Dữ liệu & huấn luyện InternVLA-N1 (bản dành cho người mới học ML)

> **Bộ tài liệu này trả lời trọn vẹn 9 câu hỏi:**
> 1. Các thuật ngữ trong dự án nghĩa là gì? → [01](01_thuat_ngu.md)
> 2. Hệ thống InternVLA-N1 hoạt động ra sao, gồm những mảnh nào? → [02](02_he_thong.md)
> 3. Code huấn luyện **nhánh System 2** chạy thế nào, từng phần làm gì? → [03](03_code_train_s2.md)
> 4. Data train **System 2** có cấu trúc gì, **cái nào bắt buộc / không bắt buộc**? → [04](04_data_train_s2.md)
> 5. Code huấn luyện **nhánh System 1** (NavDP) chạy thế nào? → [03b](03b_code_train_s1.md)
> 6. Data train **System 1** có cấu trúc gì, **cái nào bắt buộc / không bắt buộc**? → [05](05_data_train_s1.md)
> 7. Làm sao **sinh data S2 từ file `.mcap`** của robot? → [06](06_pipeline_mcap_to_s2.md)
> 8. Làm sao **sinh data S1 từ file `.mcap`** của robot? → [06b](06b_pipeline_mcap_to_s1.md)
> 9. Làm sao **sinh data S2 từ rosbag2 `.db3`** (log ROS 2 thật)? → [06c](06c_pipeline_db3_to_s2.md)
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
| [03_code_train_s2](03_code_train_s2.md) | **Mổ xẻ code nhánh train S2**: trainer → argument → dataset → collator | …đọc/sửa code S2 |
| [03b_code_train_s1](03b_code_train_s1.md) | **Mổ xẻ code nhánh train S1 (NavDP)**: config → model → dataset → 4 loss | …đọc/sửa code S1 |
| [04_data_train_s2](04_data_train_s2.md) | Cấu trúc data S2 + **bảng bắt buộc/không bắt buộc** | …chuẩn bị data S2 |
| [05_data_train_s1](05_data_train_s1.md) | Cấu trúc data S1 + **bảng bắt buộc/không bắt buộc** | …chuẩn bị data S1 |
| [06_pipeline_mcap_to_s2](06_pipeline_mcap_to_s2.md) | **`.mcap` → data S2**, 6 giai đoạn, có code chạy được | …bắt tay làm data S2 |
| [06b_pipeline_mcap_to_s1](06b_pipeline_mcap_to_s1.md) | **`.mcap` → data S1**, 8 giai đoạn + bản đồ vật cản | …bắt tay làm data S1 |
| [06c_pipeline_db3_to_s2](06c_pipeline_db3_to_s2.md) | **`.db3` (rosbag2) → data S2**: giải mã CDR, cây TF, depth từ PointCloud2 | …dùng **log ROS 2 thật** |
| [07_phu_luc_lerobot_format](07_phu_luc_lerobot_format.md) | Phụ lục: `chunk`, `parquet`, 4 file `meta/` — giải nghĩa từng trường | …hiểu file vật lý |
| [08_phu_luc_thu_thap_data](08_phu_luc_thu_thap_data.md) | Phụ lục: thu data bằng simulator / RGB-D / điện thoại, lộ trình, bẫy | …chọn thiết bị & lộ trình |
| [09_giai_thich_ham_mcap2s2](09_giai_thich_ham_mcap2s2.md) | Mổ **từng hàm** của `tools/mcap2s2.py`: vào/ra, vì sao, sửa ở đâu | …**sửa** script chứ không chỉ chạy |
| [10_phu_luc_raw_data_vln_ce](10_phu_luc_raw_data_vln_ce.md) | Phụ lục: `vln_ce/raw_data` là gì — "đề bài" JSON vs "bài giải" `traj_data`, ai đọc nó | …hiểu thư mục `raw_data` trên HF |

**Công cụ kèm theo** (thư mục [tools/](tools/)):

| Script | Việc nó làm | Đã chạy thử? |
|---|---|---|
| [tools/mcap_inspect.py](tools/mcap_inspect.py) | Khảo sát một file `.mcap` bất kỳ: liệt kê topic, schema, tần số, cây field | ✅ dùng ở phase 0 của cả hai pipeline |
| [tools/mcap2s2.py](tools/mcap2s2.py) | Chuyển `.mcap` → dataset LeRobot đúng chuẩn `vln_ce` | ✅ sinh 2 episode, **loader thật đọc được** |
| [tools/db32s2.py](tools/db32s2.py) | Chuyển **rosbag2 `.db3`** (log ROS 2 thật) → dataset `vln_ce`. Tự giải mã CDR, suy chiều cao/góc cúi camera từ cây TF, dựng depth từ `PointCloud2` — **không cần cài ROS** | ✅ chạy trên bag thật 572 MB, **loader thật đọc được** |

> 📌 Pipeline S1 ([06b](06b_pipeline_mcap_to_s1.md)) **chưa có script kèm theo** — tài liệu cung cấp
> bản triển khai tham chiếu để bạn ghép thành `mcap2s1.py`.

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

👉 **Kết luận thực dụng:** nếu bạn có log robot thu ngoài đời thật, hãy **làm data System 2 trước** —
nó khả thi, và là phần "thông minh" nhất của hệ. Đường đi cụ thể tuỳ định dạng log:
**`.db3` (rosbag2 — mặc định của `ros2 bag record`)** → [06c](06c_pipeline_db3_to_s2.md) ·
**`.mcap`** → [06](06_pipeline_mcap_to_s2.md).
Nếu robot của bạn **đã có sẵn stack SLAM/Nav2** (bản đồ occupancy + pose chính xác) thì data S1 lại
hoá ra dễ — đường đi: [06b](06b_pipeline_mcap_to_s1.md).

---

## Ba sự thật hay bị hiểu nhầm nhất

1. **Không có "một dataset của InternVLA-N1".** Có **3 bộ con** (`vln_ce`, `vln_n1`, `vln_pe`) nuôi
   **3 model khác nhau**, mỗi bộ có một loader riêng và **không thay thế được cho nhau**
   ([02](02_he_thong.md) mục 3).
2. **Nhánh S2 không bao giờ "train từ đầu".** Cả hai script `qwenvl_train` đều nạp trọng số có sẵn
   bằng `from_pretrained(...)` — luôn là **fine-tune** ([03](03_code_train_s2.md) mục 7).
   ⚠️ **Nhánh S1 thì ngược lại:** `ckpt_to_load=''` là mặc định → **khởi tạo ngẫu nhiên** mọi thứ trừ
   xương sống DepthAnythingV2 ([03b](03b_code_train_s1.md) mục 4.2).
3. **Chữ "action" có hai nghĩa.** Với S2 là **số nguyên** `{0,1,2,3,5}`; với S1 là **ma trận 4×4**
   (quỹ đạo liên tục). Cùng tên, khác hẳn nhau ([02](02_he_thong.md) mục 5).

---

## Liên kết tới tài liệu kỹ thuật khác trong repo

- [../handbook/03_data_contract.md](../handbook/03_data_contract.md) — hợp đồng dữ liệu (bản đo thật, cô đọng).
- [../handbook/02_code_structure.md](../handbook/02_code_structure.md) — cấu trúc code + chữ ký hàm phía inference.
- Code loader thật: `InternNav/code/internnav/dataset/internvla_n1_lerobot_dataset.py` (S2),
  `.../navdp_lerobot_dataset.py` (S1).

*Bộ tài liệu mô tả code tại thời điểm phân tích (27–28/07/2026). Nếu code đổi, hãy kiểm lại các `file:line` được trích.*
