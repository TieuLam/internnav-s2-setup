# 03 — Data cho System 2: cấu trúc, kiểu dữ liệu, ý nghĩa từng thứ

> **File này để làm gì:** mổ xẻ chi tiết bộ data **`vln_ce`** dùng để train **System 2** (bộ não
> "nghĩ chậm" — nhìn ảnh + đọc lệnh → chấm điểm đích). Giải thích **từng trường dữ liệu là gì, kiểu
> gì, và model học được điều gì từ nó**. Mọi thứ đối chiếu với loader thật
> `internnav/dataset/internvla_n1_lerobot_dataset.py` (class `NavPixelGoalDataset`).
>
> Bộ tài liệu: [02_hai_he_thong](02_hai_he_thong.md) · [04_data_cho_system1](04_data_cho_system1.md) ·
> [05_thu_thap_data](05_thu_thap_data.md) · handbook gốc: [../handbook/03_data_contract.md](../handbook/03_data_contract.md)

---

## 1. Model System 2 học "trò chơi" gì?

Hình dung một trò chơi: **đưa cho model một tấm ảnh camera và một câu lệnh, nó phải trả lời một
trong ba kiểu:**

1. **"Đi tới điểm này"** → chấm một pixel `[u, v]` trên ảnh (đây là *pixel goal*, câu trả lời quan
   trọng nhất). Trước khi chấm, nó xuất ký hiệu `↓` (cúi nhìn xuống để thấy sàn).
2. **"Quay đi đã"** → xuất chuỗi hành động quay: `←` (trái) hoặc `→` (phải), khi chưa thấy đích.
3. **"Xong rồi, dừng"** → xuất `STOP`.

Train tức là cho model xem **hàng chục nghìn tình huống** kèm đáp án đúng của ba kiểu trên, để nó học
cách nhìn cảnh + hiểu lệnh mà chọn đúng.

---

## 2. Cấu trúc thư mục một scene (căn phòng)

```
<scene>/
├── data/chunk-000/episode_000000.parquet        ← bảng số: action, pose, goal của từng frame
├── videos/chunk-000/
│   ├── observation.images.rgb.{H}cm_{pitch_1}deg/    ← ảnh màu NHÌN THẲNG   (.jpg)
│   │      episode_000000_0.jpg, episode_000000_1.jpg, ...
│   ├── observation.images.rgb.{H}cm_{pitch_2}deg/    ← ảnh màu NHÌN CÚI     (.jpg)
│   └── observation.images.depth.{H}cm_{pitch_2}deg/  ← ảnh độ sâu           (.png)
└── meta/
    ├── episodes.jsonl     ← câu lệnh của từng episode
    └── tasks.jsonl        ← danh sách câu lệnh
```

> **Vì sao có 2 góc ảnh (`pitch_1` và `pitch_2`)?** Loader train đọc **cặp** ảnh mỗi frame: một góc
> *nhìn thẳng* (để hiểu bối cảnh) và một góc *nhìn cúi* (để thấy sàn, chấm điểm đích chính xác). Ví dụ
> cấu hình `60cm_30_30` nghĩa là camera cao 60cm, cả hai góc cúi 30°; `125cm_0_30` là cao 125cm,
> góc thẳng 0° và góc cúi 30°.

**`{setting}`** ghép theo công thức `{chiều_cao}cm_{góc_cúi}deg`. Các cấu hình dùng để train (đọc từ
`data_dict` trong loader): `r2r/rxr/scalevln` × `{125cm_0_30, 125cm_0_45, 60cm_15_15, 60cm_30_30}`.

> ⚠️ **Tránh** cấu hình camera nhìn thẳng đơ `125cm_0deg` (không cúi): camera không thấy sàn → cột
> `goal` toàn `(-1,-1)` (không có đích) → vô dụng cho train.

---

## 3. Bảng số (parquet) — từng cột là gì

Đọc file `.parquet` bằng `pandas` sẽ ra một bảng, mỗi hàng = một frame. Các cột quan trọng:

| Cột | Kiểu dữ liệu | Nghĩa (giải thích cho người mới) |
|---|---|---|
| `action` | số nguyên `{0,1,2,3,5}`; `-1` = frame đầu | Nút bấm đúng tại frame đó: `1=tiến ↑`, `2=trái ←`, `3=phải →`, `5=cúi ↓`, `0=STOP`. `-1` chỉ đánh dấu frame khởi đầu (bỏ qua khi tính điểm). |
| `goal.{setting}` | 2 số nguyên `[u, v]` | **Điểm đích trên ảnh** (cột `u`, hàng `v`). `(-1,-1)` nghĩa là frame này chưa có đích để chấm. Đây là *label* quan trọng nhất. |
| `relative_goal_frame_id.{setting}` | số nguyên | Còn bao nhiêu frame nữa thì tới đích. Loader dùng số này để cắt đoạn train. `-1` = không có. |
| `pose.{setting}` | ma trận 4×4 (số thực) | Camera đang ở đâu/hướng nào tại frame đó. Dùng để **tính ra quỹ đạo GT** cho phần "traj token" của model (xem mục 5). |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | các số quản lý | Trường chuẩn của LeRobot: đánh số frame/episode. Luôn dùng `timestamp` để tính thời gian, **đừng** suy từ fps. |

---

## 4. Các file ảnh — kiểu và đơn vị (rất dễ sai)

| Loại ảnh | Định dạng | Kích thước | Kiểu số | Ghi chú quan trọng |
|---|---|---|---|---|
| RGB (2 góc) | `.jpg` | 640×480 | uint8 (0–255) | Model tự thu nhỏ về 384×384 bên trong — **bạn không cần resize trước**. |
| Depth | `.png` | 640×480 | **uint16, đơn vị milimét** | Loader S2 **chia cho 1000** để ra mét, rồi cắt (clip) ở 5 mét. |

> 🚨 **Bẫy đơn vị depth (nhớ kỹ):** depth lưu bằng **số nguyên milimét** (vd giá trị `2300` = 2.3 m).
> Loader System 2 quy đổi bằng phép **`/1000`**. (Loader System 1 lại dùng `/10000` — khác nhau! Xem
> [04_data_cho_system1](04_data_cho_system1.md).) Nếu bạn tự tạo data mà để sai đơn vị, model học sai
> hoàn toàn mà không báo lỗi.

---

## 5. Loader biến data thô thành "bài tập" ra sao (logic thật, giải thích chậm)

Class `NavPixelGoalDataset` duyệt mỗi episode và cắt thành **ba loại bài tập**, đúng với ba kiểu trả
lời ở mục 1:

1. **Bài "chấm đích" (pixel-goal sample):** frame nào có `goal ≠ (-1,-1)` → tạo mẫu bắt model xuất
   `↓` rồi cặp số `u v`. Đây là loại giá trị nhất.
2. **Bài "quay" (turn sample):** frame chưa có đích → mẫu bắt model xuất chuỗi `←`/`→`.
3. **Bài "dừng" (stop sample):** frame cuối episode → mẫu bắt model xuất `STOP`. (Loader nhân loại này
   lên 5 lần để model không "quên" học dừng — gọi là *cân bằng lớp*.)

Ngoài ra, khi bật chế độ `pixel_goal_only`, loader còn dựng thêm **nhãn quỹ đạo** từ cột `pose`:
- `traj_images` / `traj_depths`: dãy ảnh + depth của đoạn đường tới đích.
- `traj_poses`: quỹ đạo (x, y, góc quay) tương đối, đã được nội suy cho mượt.

Nghĩa là **System 2 không chỉ học chấm điểm — nó còn học một chút "hình dung đường đi"** (phần này
nối vào System 1). Vì vậy, để tạo data S2 *đầy đủ*, bạn cần cả **pose 4×4 mỗi frame**, không chỉ pixel
goal.

---

## 6. Câu lệnh (instruction) — lấy từ đâu

Nằm trong `meta/episodes.jsonl`. Mỗi episode có trường `tasks` chứa câu lệnh tiếng Anh tự nhiên,
ví dụ thật đo được:

> *"You are facing toward the entrance of the church, turn around and stop in between the steps..."*

Một episode có thể chứa nhiều câu lệnh, ngăn nhau bằng ký hiệu `<INSTRUCTION_SEP>`; loader lấy phần
đầu tiên. **Đây là phần bắt buộc con người phải viết** — máy không tự bịa ra được câu lệnh đúng.

---

## 7. Tóm tắt: để có MỘT ví dụ train cho System 2, bạn cần

| Thành phần | Bắt buộc? | Nguồn |
|---|---|---|
| Ảnh RGB (nhìn thẳng + nhìn cúi) | ✅ | camera |
| Ảnh Depth (uint16, mm) | ✅ (cho phần quỹ đạo) | camera RGB-D **hoặc** DepthAnything |
| Câu lệnh tiếng Anh | ✅ | con người viết |
| `action` rời rạc mỗi frame | ✅ | suy từ chuyển động (Δvị trí, Δgóc) |
| `goal [u,v]` (pixel đích) | ✅ | chiếu đích 3D về ảnh, hoặc chấm tay |
| `pose` 4×4 mỗi frame | ✅ | pose camera (SLAM / ARKit / simulator) |

Cách **thu thập** từng thành phần này bằng simulator / camera chuyên dụng / điện thoại / ROS2 →
[05_thu_thap_data](05_thu_thap_data.md).

Tiếp theo: bộ data khó hơn một bậc — [04_data_cho_system1](04_data_cho_system1.md).
