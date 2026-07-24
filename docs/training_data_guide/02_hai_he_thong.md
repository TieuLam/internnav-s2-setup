# 02 — Hai bộ não: vì sao InternVLA-N1 cần hai loại data khác nhau

> **File này để làm gì:** giải thích **ý tưởng cốt lõi** khiến toàn bộ phần data trở nên dễ hiểu:
> N1 không phải một model, mà là **hai model ghép lại** (dual-system), và **mỗi model ăn một loại
> dữ liệu riêng**. Hiểu điều này rồi thì file 03 và 04 chỉ là chi tiết.
>
> Bộ tài liệu: [01_nhap_mon_thuat_ngu](01_nhap_mon_thuat_ngu.md) ·
> [03_data_cho_system2](03_data_cho_system2.md) · [04_data_cho_system1](04_data_cho_system1.md) ·
> [05_thu_thap_data](05_thu_thap_data.md)

---

## 1. Ví von: bộ não người có "nghĩ chậm" và "phản xạ nhanh"

Ý tưởng lấy cảm hứng từ tâm lý học (khái niệm *"Thinking, Fast and Slow"*):

- **System 2 — nghĩ chậm, có lý trí.** Khi bạn đọc bản đồ, cân nhắc *"à, phải rẽ ở chỗ cái cây kia"*
  — đó là suy nghĩ có chủ đích, chậm. Trong robot: **nhìn ảnh + đọc câu lệnh → quyết định đích đến**.

- **System 1 — phản xạ nhanh, bản năng.** Khi đang đi mà tự né cái ghế trước mặt không cần suy nghĩ —
  đó là phản xạ. Trong robot: **từ đích đến → vẽ đường đi mượt né vật cản**, làm liên tục nhiều lần/giây.

Robot cần **cả hai**: System 2 để hiểu *đi đâu*, System 1 để lo *đi thế nào cho khỏi đâm*.

---

## 2. Hai bộ não làm việc cùng nhau như thế nào (luồng chạy thật)

```
   Câu lệnh: "đi tới cửa nhà thờ rồi dừng ở bậc thang"
                     │
                     ▼
        ┌─────────────────────────┐
        │   SYSTEM 2  (VLM)        │  nhìn ảnh + đọc lệnh
        │   "nghĩ chậm"           │  → chấm 1 ĐIỂM trên ảnh (pixel goal)
        └─────────────────────────┘  → kèm 1 "gói tín hiệu" (latent) cho S1
                     │
                     ▼
        ┌─────────────────────────┐
        │   SYSTEM 1  (NavDP)      │  nhận điểm đích + ảnh + depth
        │   "phản xạ nhanh"       │  → vẽ ĐƯỜNG ĐI cong né vật cản
        └─────────────────────────┘  → ra lệnh bánh xe: tiến/xoay
                     │
                     ▼
              Robot di chuyển → chụp ảnh mới → lặp lại
```

Điểm mấu chốt: **S2 và S1 nói hai "ngôn ngữ" khác nhau**, nên khi *dạy* chúng, ta cũng phải cho ăn
hai loại "bài tập" khác nhau.

---

## 3. Vì sao KHÔNG có "một loại data chung" (điểm dễ hiểu lầm nhất)

Người mới hay tưởng: "có một dataset của InternVLA-N1". **Sai.** Có **ba** bộ con trong dataset
`InternData-N1`, và chúng **không thay thế được cho nhau**:

| Bộ con | Nuôi model nào | Đặc điểm nhận dạng | Ta có dùng để train không? |
|---|---|---|---|
| **`vln_ce`** | **System 2** | có sẵn cả RGB + Depth, có cột `goal` (điểm đích trên ảnh) | ✅ dùng train S2 |
| **`vln_n1`** | **System 1** | mỗi frame là ma trận 4×4 (quỹ đạo), có bản đồ 3D `.ply` | ✅ dùng train S1 |
| **`vln_pe`** | baseline đối chứng (CMA/RDP) — **không** thuộc dual-system | pose robot + pose camera tách riêng | ✖ không dùng cho N1 |

> Bằng chứng trong code: mỗi bộ có **một "đầu đọc" (loader) riêng** trong `internnav/dataset/`.
> `internvla_n1_lerobot_dataset.py` chỉ đọc được `vln_ce`; `navdp_lerobot_dataset.py` chỉ đọc được
> `vln_n1`. Đưa nhầm bộ → loader tìm không thấy cột cần thiết và **báo lỗi** (hoặc tệ hơn: chạy sai
> âm thầm). Đây là lý do phải phân biệt rạch ròi ngay từ đầu.

---

## 4. Bảng so sánh hai loại data (học thuộc bảng này là đủ nền)

| | **Data cho System 2** (`vln_ce`) | **Data cho System 1** (`vln_n1`) |
|---|---|---|
| Model học điều gì | "Nhìn cảnh + nghe lệnh → **chấm điểm đích** trên ảnh, hoặc bảo rẽ/dừng" | "Có đích rồi → **vẽ đường đi cong** né vật cản" |
| Đầu vào (input) | ảnh RGB + câu lệnh (+ depth phụ) | ảnh RGB + depth + điểm đích |
| Đáp án (label/GT) | **pixel goal `[u,v]`** + action rời rạc (tiến/trái/phải/dừng) + quỹ đạo | **quỹ đạo camera** (dãy ma trận 4×4) + điểm "an toàn hay va chạm" (critic) |
| Cần con người chú thích? | **Có** — phải viết câu lệnh tiếng Anh cho mỗi lượt đi | **Không** — mọi đáp án tự suy ra từ đường đi + bản đồ 3D |
| Thứ khó kiếm nhất | câu lệnh chất lượng + biết điểm đích ở pixel nào | **bản đồ 3D vật cản** (`pointcloud.ply`) |
| Thu bằng thiết bị thường được không? | **Được** (khả thi) | **Khó** (cần bản đồ 3D) |

Ghi nhớ dòng cuối: **nếu bạn dùng camera điện thoại / camera robot thường, hãy ưu tiên làm data cho
System 2 trước** — nó khả thi và là phần "thông minh" nhất. Lý do chi tiết ở
[05_thu_thap_data](05_thu_thap_data.md) và [06_lo_trinh_bat_dau](06_lo_trinh_bat_dau.md).

---

## 5. Một cụm từ gây rối cần làm rõ: "action"

Cả hai hệ đều có thứ gọi là **action** (hành động), nhưng **nghĩa khác nhau** — đây là bẫy hay gặp:

- **Action của System 2** = **số nguyên rời rạc**, một trong vài lựa chọn: `1 = tiến`, `2 = quay trái`,
  `3 = quay phải`, `5 = cúi nhìn xuống`, `0 = STOP (dừng)`. Kiểu "chọn 1 trong 5 nút bấm".
- **Action của System 1** = **ma trận 4×4 / quỹ đạo liên tục** — không phải "nút bấm" mà là "đường đi
  cong trơn". Kiểu "vẽ nét bút liền".

Khi đọc code hay dataset, luôn tự hỏi *"action này của hệ nào?"* để khỏi nhầm.

---

## 6. FAQ: Hai hệ có cần chung một video / đồng bộ dữ liệu không?

> Câu hỏi rất hay gặp: *"Data của S1 và S2 có phải sinh ra từ cùng một video gốc không? Nếu tôi quay
> cùng một căn phòng ở các buổi khác nhau, mỗi buổi tạo data cho một hệ thì có dùng được không?"*

**Trả lời ngắn: KHÔNG cần chung video, KHÔNG cần đồng bộ giữa hai hệ. Cách quay khác buổi cho mỗi hệ
là dùng được.** Nhưng có một loại "đồng bộ" khác thì *bắt buộc*. Phân biệt rõ 4 ý sau:

### 6.1. Data gốc của N1 vốn đã không chung video (bằng chứng)
Hai hệ được tạo từ **nguồn hoàn toàn khác nhau** — khác cả simulator, khác cả căn phòng, khác cả model
camera:

| | Data S2 (`vln_ce`) | Data S1 (`vln_n1`) |
|---|---|---|
| Bộ cảnh | r2r, rxr, scalevln | 3dfront, gibson, hm3d, hssd, matterport3d, replica |
| Camera | cấu hình 60/125cm có góc cúi | mô phỏng D435i / ZED |

→ Ngay từ đầu nhóm tác giả **không** quay chung một video rồi tách. Kịch bản "cùng phòng, khác buổi"
của bạn thậm chí *khớp hơn* data gốc.

### 6.2. Vì sao được phép tách rời?
Vì **hai hệ học hai bài toán độc lập**, mỗi hệ có loader riêng, label riêng, loss riêng — chúng không
"nhìn thấy" nhau lúc train. Chúng chỉ gặp nhau **lúc chạy thật (inference)** qua một giao diện bàn giao
đã thiết kế sẵn (S2 đưa điểm đích + latent → S1 nhận). Giống hai người học riêng hai trường, miễn cùng
nói một "ngôn ngữ bàn giao" thì ghép lại làm việc được.

### 6.3. Cái ĐỒNG BỘ mà bạn KHÔNG cần vs cái BẮT BUỘC

| Loại đồng bộ | Cần? | Giải thích |
|---|---|---|
| Đồng bộ **giữa** data S1 và data S2 (frame t của hệ này ↔ frame t của hệ kia) | ❌ **Không** | Hai tập có thể quay khác buổi, khác lộ trình, khác phòng. Train vẫn chạy. |
| Đồng bộ **bên trong một video** (RGB + depth + pose + goal của cùng một frame phải cùng một khoảnh khắc) | ✅ **Bắt buộc** | Nếu depth lệch pha với RGB, hoặc pose không khớp thời điểm ảnh → label sai → model học rác **mà không báo lỗi**. Đây là lý do đường ống ROS2 phải dùng `ApproximateTimeSynchronizer`. |

### 6.4. Điều kiện tinh tế: phải NHẤT QUÁN QUY ƯỚC giữa hai tập
Không cần chung video, nhưng để lúc chạy thật S2 bàn giao được cho S1, hai bên nên **cùng quy ước cảm
biến/hình học**: chiều cao & góc cúi camera, model camera/intrinsic, hệ toạ độ (trục x/y/hướng), và
đơn vị depth (S2 `/1000`, S1 `/10000` — mỗi hệ giữ đúng hằng số của mình). Đừng buổi này để camera cao
125cm cúi 30°, buổi kia để cao 30cm nhìn thẳng — sự khác biệt đó mới phá hỏng việc ghép, chứ không phải
chuyện "khác video".

### 6.5. Bẫy riêng cho kịch bản "cùng phòng, khác buổi"
Nếu giữa hai buổi quay **bàn ghế bị xê dịch**: với *train* thì không sao (hai hệ học riêng). Chỉ có
vấn đề nếu bạn định dùng **chung một `pointcloud.ply`** (bản đồ vật cản của S1) cho cả hiện trường cũ
lẫn mới. Quy tắc: **mỗi lượt quay của S1 phải đi kèm bản đồ 3D của đúng hiện trường lúc đó**.

---

Đã hiểu bức tranh lớn. Giờ đi vào chi tiết từng bộ: bắt đầu với
[03_data_cho_system2](03_data_cho_system2.md) (dễ hình dung hơn), rồi
[04_data_cho_system1](04_data_cho_system1.md).
