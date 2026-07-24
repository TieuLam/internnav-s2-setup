# 02 — Raw ZED → format `vln_n1` để train System 1 (NavDP)

> **Vì sao làm cái này trước:** pipeline tự động 100%, không cần người gán nhãn. GT của S1
> chính là quỹ đạo camera — thứ ZED positional tracking cho không. Thêm nữa, `vln_n1` có sẵn
> biến thể camera **`zed`** (mô phỏng) → data ZED thật là "người anh em thật" của data train gốc.
>
> Format đích đối chiếu từ **loader thật** `navdp_lerobot_dataset.py` (đường load data của
> `scripts/train/base_train/configs/navdp.py`, `root_dir='data/datasets/InternData-N1/vln_n1/traj_data'`).

---

## 1. Format đích — cái loader THẬT SỰ đọc

### 1.1. Cây thư mục (mỗi scene = 1 dataset, mỗi lần quay 1 khu vực văn phòng = 1 "scene")

```
<scene>/
├── data/chunk-000/episode_XXXXXX.parquet
├── videos/chunk-000/observation.images.rgb/    ← từng FILE ẢNH rời, tên sort được
├── videos/chunk-000/observation.images.depth/  ← từng file depth rời, uint16
└── meta/episodes.jsonl (+ info.json, tasks.jsonl cho đủ bộ LeRobot)
```

⚠️ Handbook mục 2.2 ghi video của `vln_n1` tải về là `.mp4`, nhưng loader
(`navdp_lerobot_dataset.py:89–97, 142–158`) **liệt kê thư mục và mở từng file bằng
`PIL.Image.open`** — tức là trước khi train phải có bước tách mp4 → frame, hoặc data
giải nén ra đã là ảnh rời. ⬜ Chưa giải nén file `vln_n1` thật để xem — **mình sinh data mới
thì ghi thẳng ảnh rời** (PNG), khớp cái loader đọc, khỏi phụ thuộc câu trả lời.

### 1.2. Parquet — đúng 4 cột (khớp handbook 03 mục 2.2, đối chiếu loader dòng 198–202)

| Cột | Shape | Lấy từ đâu trên ZED | Ghi chú từ code loader |
|---|---|---|---|
| `index` | int64 | 0,1,2,… | |
| `observation.camera_intrinsic` | (3,3) float32 | calib trái sau rectify, **scale theo resolution ghi ra** | loader chỉ đọc dòng đầu (`tolist()[0]`) — hằng số cả episode |
| `observation.camera_extrinsic` | (4,4) float32 | pose camera từ positional tracking | loader **cũng chỉ đọc dòng đầu**; đáng chú ý: nó dùng `camera_extrinsic[2,3]` (≈ **chiều cao camera**) để dựng pixel-goal augmentation (dòng 233) → điền đúng chiều cao thật (mét) vào phần tử này |
| `action` | (4,4) float32 mỗi frame | **pose camera TỪNG frame** = quỹ đạo SE(3) | loader stack thành `camera_trajectory (N,4,4)` (dòng 201) — đây chính là GT mà NavDP học |

**Diễn giải cho đúng:** trong `vln_n1`, "action" không phải lệnh điều khiển — nó là **pose camera
tại frame đó**. NavDP tự cắt cửa sổ `predict_size=24` pose tương lai làm nhãn quỹ đạo
(dòng 383–391). Nên pipeline của mình chỉ cần ghi đúng chuỗi pose là xong phần nhãn.

---

## 2. Pipeline 6 bước (tất cả offline, từ file SVO2)

```
SVO2 ──replay──▶ (RGB, depth-NEURAL, pose 4×4, timestamp) mỗi frame
      ──cắt episode──▶ mỗi đoạn đi liên tục, tracking OK
      ──subsample──▶ nhịp frame khớp data gốc (⬜ đo — xem mục 4)
      ──đổi hệ──▶ pose về convention của vln_n1 (⬜ PHẢI xác minh — mục 5)
      ──ghi ảnh──▶ rgb PNG uint8 · depth PNG uint16 (nhân 10000, mục 3)
      ──ghi parquet + meta──▶ 4 cột như trên, mỗi scene 1 dataset
```

Chi tiết từng bước đáng nói:

1. **Replay & lấy dữ liệu:** mở SVO2 với `DEPTH_MODE.NEURAL`, bật positional tracking
   (`enable_area_memory=True`). Với mỗi frame `grab()`: lấy RGB trái rectified, depth float
   mét, `get_position(sl.REFERENCE_FRAME.WORLD)` → 4×4.
2. **Cắt episode:** bỏ 2–3 s đầu (IMU đang init), cắt tại chỗ tracking báo
   `POSITIONAL_TRACKING_STATE != OK` hoặc chỗ người quay đứng lại quá lâu. Mỗi episode nên
   là một hành trình có nghĩa (bàn A → phòng họp B) — sau này tái dùng làm episode S2 luôn.
3. **Lọc frame hỏng:** frame blur (variance of Laplacian thấp) hoặc depth phủ < ~60% pixel
   hợp lệ → cắt episode tại đó thay vì để lỗ hổng giữa chuỗi.
4. **Subsample nhịp:** xem mục 4.
5. **Ghi ảnh + parquet + meta:** `meta/episodes.jsonl` cần `episode_index` và `length` khớp
   số dòng parquet; loader dùng danh sách file sort được → đặt tên zero-pad (`000000.png`).

---

## 3. 🚨 Bẫy lớn nhất: đơn vị depth — HAI format đích dùng HAI đơn vị khác nhau

| Nơi | Đơn vị lưu (uint16) | Bằng chứng |
|---|---|---|
| `vln_ce` (S2) | **milimét**, clip 10000 = 10 m | đo thật trên file (handbook 03 mục 1.3, PL-D4) |
| `vln_n1` (S1) | **giá trị/10000 = mét** (tức 0.1 mm/đơn vị) | loader `navdp_lerobot_dataset.py:179`: `depth = load_depth(...) / 10000.0`, sau đó giữ dải **[0.1 m, 5 m]**, ngoài dải set 0 (dòng 188–189) |

→ Khi ghi depth cho pipeline S1: `uint16_value = round(depth_met * 10000)`, tức trần biểu diễn
6.55 m — quá đủ vì loader vứt mọi thứ > 5 m. Chỗ nào ZED trả NaN/inf (kính, quá xa) ghi 0
(loader coi 0 = không hợp lệ vì < 0.1 m).

⬜ Suy từ code loader, chưa đối chiếu file `vln_n1` thật. Trước khi ghi hàng loạt: giải nén 1
episode `*_zed`, in `np.array(Image.open(depth)).max()` — nếu max ≈ 50000 (5 m × 10000) thì
đúng như trên; nếu ≈ 5000 thì họ lưu mm và dòng `/10000` có ẩn ý khác → sửa lại mục này.

---

## 4. Khớp phân bố với data train gốc — 2 số phải đo trước khi sinh hàng loạt

1. **Intrinsic/FOV biến thể `_zed`:** ⬜ chưa đo (handbook mới đo `_d435i`: 480×270,
   fx=355.81, FOV 68°×42°). Tải 1 file `vln_n1/traj_data/hm3d_zed/*.tar.gz`, đọc cột
   `observation.camera_intrinsic` → biết cần resize/crop ảnh ZED thật về resolution nào.
   **Không đoán** — handbook đã chứng minh camera trong `vln_n1` mô phỏng đúng model thật.
2. **Nhịp frame (khoảng cách giữa 2 frame liên tiếp):** NavDP không đọc timestamp, nên "1 frame"
   ngầm định một bước di chuyển cỡ nào đó. ⬜ Đo trên file thật: trung bình
   `‖t[i+1] − t[i]‖` của cột action → subsample data 30 fps của mình (người đi bộ ~1.2–1.4 m/s)
   về đúng cỡ bước đó.

---

## 5. ⬜ Hệ trục toạ độ — mục chưa chốt, cấm ghi hàng loạt trước khi xác minh

Manh mối từ code: `process_pixel_goal` (loader dòng 233–236) dựng điểm
`[-y, x, height*0.8]` rồi nhân `R = camera_extrinsic[0:3,0:3]` và chiếu bằng
`u = cx + (Xc/Zc)·fx`, `v = cy + (−Yc/Zc)·fy` — kiểu trục Habitat (x phải, y **lên**, z ra sau),
khác convention COMPUTER_VISION của ZED (y xuống, z ra trước).

**Cách xác minh bắt buộc (1 buổi):** load 1 episode `vln_n1` thật → lấy pose 2 frame liên tiếp,
chiếu điểm quỹ đạo tương lai lên ảnh RGB frame hiện tại theo đúng công thức loader → điểm phải
rơi trên lối đi phía trước trong ảnh. Sau đó áp cùng phép thử với data ZED của mình (đổi trục
bằng `sl.CoordinateSystem` khi mở camera hoặc nhân ma trận đổi trục hậu kỳ) đến khi hình chiếu
cũng đúng. Đưa phép thử này thành unit test của pipeline.

---

## 6. Định nghĩa "xong" của pipeline S1

- [ ] 1 scene văn phòng sinh đúng cây thư mục mục 1.1, `navdp_lerobot_dataset.py` load
      không sửa code, `__getitem__` trả sample không NaN.
- [ ] Depth sau loader nằm trong [0, 5] m (in min/max — cùng tinh thần checklist handbook 03 mục 5).
- [ ] Phép thử chiếu quỹ đạo (mục 5) pass trên cả data gốc lẫn data mình.
- [ ] Chạy `scripts/train/base_train/` overfit thử 1 scene vài trăm step — loss giảm là pipeline thông.
