# 03 — Raw ZED → format `vln_ce` để train System 2 (pixel-goal VLM)

> **Vì sao khó hơn S1:** `vln_ce` chứa 3 loại nhãn mà sensor không tự sinh ra được —
> action rời rạc, pixel goal, và instruction ngôn ngữ. Trong sim, cả 3 đều lấy từ
> oracle (shortest-path follower + template R2R); ngoài đời phải **pseudo-label từ quỹ đạo**
> và **người viết instruction**. Thêm một rào cản cấu trúc: format giả định mỗi thời điểm có
> ảnh ở NHIỀU góc pitch — camera thật chỉ có một (mục 4).
>
> Format đích: [handbook 03](../handbook/03_data_contract.md) mục 1 (21 cột, đo thật).
> Cách loader tiêu thụ: `internvla_n1_lerobot_dataset.py` (dòng 752–1021).

---

## 1. Nhắc lại format đích (những gì PHẢI sinh ra cho mỗi episode)

| Thành phần | Spec (đo thật từ handbook) | Nguồn trên pipeline ZED |
|---|---|---|
| RGB PNG | 640×480 uint8, mỗi frame, key `observation.images.rgb.{setting}` | RGB trái rectified, resize (chú ý giữ tỉ lệ FOV — mục 5.1) |
| Depth PNG | 640×480 **uint16 MILIMÉT**, clip 10000 | depth NEURAL (mét) × 1000, clip 10000 — **khác đơn vị với pipeline S1!** (xem [02](02_pipeline_s1_vln_n1.md) mục 3) |
| `pose.{setting}` | (4,4) float32/frame | positional tracking (sau khi chốt hệ trục — [02](02_pipeline_s1_vln_n1.md) mục 5) |
| `action` | int32 `{1↑,2←,3→,5↓}`, frame đầu `-1` | **pseudo-label** — mục 3 |
| `goal.{setting}` | `[u,v]` int32, `(-1,-1)` = không có | **pseudo-label** — mục 3 |
| `relative_goal_frame_id.{setting}` | int32, số frame còn lại tới goal | sinh cùng lúc với `goal` |
| `meta/tasks.jsonl` + `episodes.jsonl` | instruction tiếng Anh; `length` = số dòng parquet (loader có assert) | **người viết** — mục 6 |
| 5 trường LeRobot (`timestamp/frame_index/episode_index/index/task_index`) | bắt buộc | timestamp thật từ SVO2 (giây, float) — đừng suy từ fps (bài học PL-D2) |

---

## 2. Tư tưởng chung: "diễn lại" oracle của sim trên quỹ đạo người thật

Trong sim, một episode `vln_ce` sinh ra thế này: có path GT → follower đi từng bước rời rạc
(tiến 25 cm / xoay 15°) → mỗi frame chấm sẵn goal là điểm path phía trước chiếu vào ảnh.
Pipeline của mình đảo lại: **người đi tự nhiên trước, rồi hậu kỳ "rời rạc hoá" quỹ đạo đó**
thành chuỗi frame + action + goal như thể một follower đã đi.

```
pose liên tục 30fps ──(a) resample──▶ chuỗi keyframe "kiểu robot"
                     ──(b) gán action──▶ {1,2,3} giữa các keyframe, -1 ở đầu, (5 nếu dùng — mục 4)
                     ──(c) chọn waypoint──▶ điểm quỹ đạo tương lai làm goal
                     ──(d) chiếu──▶ [u,v] trên ảnh keyframe hiện tại, che khuất → (-1,-1)
```

## 3. Pseudo-label action + pixel goal (bước (a)–(d))

**(a) Resample thành keyframe.** Duyệt quỹ đạo, nhả keyframe mỗi khi tích luỹ đủ
**Δtiến ≥ 0.25 m** hoặc **Δyaw ≥ 15°** (đúng granularity action space của N1). Người đi bộ
vừa tiến vừa xoay → mỗi đoạn giữa 2 keyframe quy về loại trội: `|Δyaw| > ngưỡng` → `2/3`,
ngược lại → `1`. Frame đầu episode gán `-1` (đúng quan sát đo thật: chuỗi bắt đầu bằng `-1`).

**(b) Kiểm soát nhiễu:** quỹ đạo người có vi-dao-động → lọc pose (EMA/spline) TRƯỚC khi
resample, nếu không sẽ sinh chuỗi `2,3,2,3` rác ở đoạn đi thẳng. Đối chiếu "độ trông giống
thật": phân bố action của data mình so với đo thật của handbook
(`[-1,2,2,…,1,1,2,1,2,1]` — turn nhiều hơn forward là bình thường với RxR).

**(c) Chọn goal cho từng keyframe.** Quy tắc mô phỏng oracle: goal = điểm quỹ đạo **xa nhất
còn nhìn thấy** trong giới hạn `num_future_steps` (tham số có thật của loader, dòng 839) —
thường là điểm ngay trước khúc rẽ tiếp theo. `relative_goal_frame_id` = số keyframe từ đây
tới đó.

**(d) Chiếu vào ảnh.** `p_cam = T_cam_world · p_world`; `u = cx + fx·X/Z`, `v = cy + fy·Y/Z`
(sau khi đã chốt hệ trục — phép thử ở [02](02_pipeline_s1_vln_n1.md) mục 5 dùng chung).
Goal bị loại → `(-1,-1)` khi: ra ngoài khung; Z ≤ 0; hoặc **bị che**: so `Z` với depth thật
tại `[u,v]` (lệch > ~0.3 m nghĩa là có tường/vật chắn giữa — sim không cần bước này vì
oracle biết mesh, mình thay bằng depth). Loader xử lý được frame goal `(-1,-1)` (nhánh
turn_list, dòng 876–889) nên không cần ép mọi frame có goal.

**Sanity check bắt buộc** trước khi tin nhãn: vẽ goal lên 50 ảnh ngẫu nhiên — chấm phải nằm
trên sàn/lối đi phía trước, không lơ lửng trên tường. (Nhớ quy ước: data lưu `[u,v]` =
[cột, hàng] — bài học đảo trục ở handbook 03 mục 4.4.)

---

## 4. Vấn đề cấu trúc: cặp pitch (ngang, cúi) — camera thật không có

Bằng chứng từ code: loader S2 ăn data theo **cặp** `(pitch_1, pitch_2)` cùng height —
`vlln_lerobot_dataset.py:18–37` khai `pitch_1=0, pitch_2=30`; docstring dòng 577 gọi pitch_1
là *"Horizon camera pitch"*; và dòng 325–329 + `internvla_n1_lerobot_dataset.py:1015–1021`
cho thấy nó lấy **RGB của pitch_1 (nhìn ngang)** ghép với **ảnh look-down + goal/pose của
pitch_2 (cúi)**. Sim render 2 camera cùng lúc nên có cả hai; ZED đội đầu chỉ có MỘT pitch
vật lý.

| Phương án | Cách làm | Đánh giá |
|---|---|---|
| **A. Một pitch duy nhất, pitch_1 = pitch_2 (≈30°)** | gắn camera cúi ~30°, sinh 1 setting duy nhất, config train đặt hai tham số bằng nhau | **Đề xuất.** Đơn giản, mọi nhãn thật. Trả giá: nhánh action `5 (↓)`/look-down không có data — model mất kỹ năng nhìn xuống ⬜ cần chạy thử loader với pitch_1==pitch_2 xem có nhánh nào crash |
| B. Synth ảnh cúi từ point cloud | reproject ảnh+depth sang camera ảo cúi hơn | Đúng cấu trúc data gốc nhưng ảnh synth nhiều lỗ ở vùng chưa quan sát; đắt công. Để dành làm nâng cấp |
| C. Quay 2 lượt (ngang + cúi) cùng tuyến | 2 pass cùng hành lang | Pose không trùng khớp từng frame giữa 2 lượt → nhãn cặp sai lệch; không khuyến nghị |

Chốt phương án A thì **góc cúi phải cố định và đo được** (đọc IMU lúc đứng yên, ghi vào
metadata) vì nó nằm trong tên setting `{height}cm_{pitch}deg` — với người đội đầu cao ~165 cm,
setting của mình sẽ kiểu `165cm_30deg`, KHÔNG trùng 5 setting gốc. Loader nhận height/pitch
từ config (dòng 845–850) nên tên mới không sao, miễn khai đúng.

⚠️ Tuyệt đối tránh pitch ≈ 0° làm setting duy nhất: bài học đo thật `125cm_0deg` có goal
toàn `(-1,-1)` vì không thấy sàn (handbook 03 mục 1.2) — camera ngang tầm mắt người là
đúng cái bẫy đó.

---

## 5. Hai bẫy kỹ thuật còn lại

1. **Resize về 640×480 làm méo FOV:** ZED HD720 là 16:9 (1280×720), đích là 4:3 (640×480).
   Resize thẳng sẽ méo intrinsic → **crop ngang về 960×720 rồi scale 2/3** (fx, fy scale theo,
   cx, cy tính lại). Ghi intrinsic sau-crop vào metadata vì S2 eval dựng intrinsic từ
   width/height/hfov (handbook 03 mục 4.2).
2. **Loader tìm RGB đuôi `.jpg` trong khi file `vln_ce` tải về là `.png`** (dòng 1015 vs đo
   thật handbook mục 1.3). ⬜ Mâu thuẫn chưa giải quyết — có thể pipeline gốc convert trước
   khi train. Khi sinh data: cứ ghi PNG (chuẩn đo thật), nhưng **chạy thử loader trước khi
   sinh hàng loạt**; nếu nó đòi `.jpg` thì thêm bước convert/symlink, đừng sửa loader.

---

## 6. Instruction — phần không tự động hoá được (và quyết định data có giá trị không)

- Mỗi episode ≥ 1 câu **tiếng Anh** kiểu R2R/RxR: mô tả tuyến đường bằng landmark
  ("*Walk past the row of desks, turn left at the glass meeting room, stop next to the printer*").
  Nhiều câu cho 1 episode thì nối bằng `<INSTRUCTION_SEP>` (loader split đúng token này, dòng 770).
- Quy trình gợi ý: người quay **nói mô tả tuyến ngay sau khi quay** (ghi âm/ghi chú) → một người
  khác viết lại thành câu R2R-style → duyệt chéo bằng cách chỉ đọc câu + xem video, đi lại được
  đúng tuyến thì đạt.
- Có thể nhờ VLM (đưa video + quỹ đạo) sinh nháp rồi người sửa — nhưng người duyệt là bắt buộc:
  instruction sai làm hỏng đúng thứ S2 cần học (grounding ngôn ngữ ↔ không gian).

---

## 7. Định nghĩa "xong" của pipeline S2

- [ ] 1 scene qua `get_annotations_from_lerobot_data(data_path, setting)` không lỗi, đủ
      episodes/instructions/pixel_goals/poses.
- [ ] Overlay goal lên ảnh: ≥ 90% frame có goal trông hợp lý (chấm trên lối đi).
- [ ] Phân bố action không có mẫu `2,3,2,3` dày đặc ở đoạn đi thẳng (nhiễu chưa lọc hết).
- [ ] Chạy thử `scripts/train/train_system2.sh` (sửa config trỏ data mới, pitch_1==pitch_2)
      qua được vài chục step.
- [ ] Checklist bẫy dữ liệu của [handbook 03](../handbook/03_data_contract.md) mục 5 tick đủ
      trên data MÌNH SINH RA (nó viết cho data tải về nhưng áp nguyên cho data tự sinh).
