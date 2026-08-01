# 06c — Pipeline: từ **rosbag2 `.db3`** → data train System 2

> **File này để làm gì:** hướng dẫn dùng [tools/db32s2.py](tools/db32s2.py) để biến **log ROS 2 thật**
> (`.db3`) thành dataset LeRobot mà `NavPixelGoalDataset` nạp được — **không cần cài ROS**.
>
> Mọi con số dưới đây **đo thật** trên bag
> [`vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s`](tools/vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s/)
> (572 MB, 10 s, 20 651 message, robot hình người 2 camera ZED).
>
> Bộ tài liệu: [06_pipeline_mcap_to_s2](06_pipeline_mcap_to_s2.md) (bản `.mcap`) ·
> [04_data_train_s2](04_data_train_s2.md) (hợp đồng dữ liệu) · [03_code_train_s2](03_code_train_s2.md)

---

## 0. Vì sao cần script riêng, không dùng lại `mcap2s2.py`?

`.mcap` trong tài liệu [06](06_pipeline_mcap_to_s2.md) là file "tự mô tả" với message mã hoá **JSON**.
Log robot thật ghi bằng `ros2 bag record` mặc định là **`.db3` = SQLite + message mã hoá CDR**, và dữ
liệu hình học **không** nằm gọn trong một topic mà rải theo **cây TF**. Năm khác biệt và cách xử lý:

| | `.mcap` (JSON) | `.db3` (ROS 2 thật) | `db32s2.py` xử lý |
|---|---|---|---|
| Vỏ chứa | file mcap | **SQLite**: `topics` + `messages(topic_id, timestamp, data)` | `sqlite3` chuẩn của Python |
| Mã hoá message | JSON dễ đọc | **CDR** nhị phân, có căn lề | `class CDR` tự viết (mục 2) |
| Pose robot | 1 topic `/robot/pose` | `/kiss/odometry` + **cây `/tf`** | `class TFBuffer` (mục 3) |
| Chiều cao & góc cúi camera | ghi trong metadata | **phải suy ra từ `/tf_static`** | tự tính, in ra để kiểm (mục 3) |
| Depth | có topic ảnh depth | thường **không có**, chỉ `PointCloud2` | chiếu cloud → ảnh depth (mục 4) |

> 💡 **Không cần cài ROS 2 / rclpy / cv_bridge / rosbags.** Chỉ cần
> `numpy · pillow · pyarrow · scipy`. Đây là chủ ý: bạn xử lý data trên máy Windows/laptop bất kỳ.

---

## 1. Bước 0 — LUÔN khảo sát bag trước

```bash
cd docs/training_data_guide/tools
python db32s2.py --bag vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s --inspect
```

Chế độ này in: danh sách topic · **topic được tự đoán** · cây TF · quỹ đạo · **hình học camera suy từ
TF**. Kết quả thật trên bag mẫu:

### 1.1. Topic có gì (20 topic, 20 651 message trong 10 s)

| Topic | Kiểu | Msg | Vai trò cho S2 |
|---|---|---|---|
| `/camera/head_front_zed_onboard/left/color/rect/image/compressed` | CompressedImage | 150 | **RGB `pitch_1`** (camera đầu) |
| `/camera/waist_front_zed_onboard/left/color/rect/image/compressed` | CompressedImage | 149 | **RGB `pitch_2`** (camera bụng, nhìn cúi) |
| `…/left/color/rect/camera_info` (×4) | CameraInfo | 149 | **ma trận `K`** |
| `/camera/waist_front_zed_onboard/point_cloud/cloud_registered` | PointCloud2 | 100 | **nguồn depth duy nhất** |
| `/kiss/odometry` | Odometry | 99 | **quỹ đạo** (`odom_lidar → base_link`) |
| `/tf` · `/tf_static` | TFMessage | 2266 · 6 | **hình học lắp camera** |
| `/ekf/imu/data`, `/joint_states`, `/whole_body/motor_state`, `…/right/…` | | 14 799 | ⚪ không dùng cho S2 |

Script **tự đoán** đúng cả 6 topic cần thiết bằng cách lọc theo *kiểu message* rồi ưu tiên từ khoá
(`waist/chest/lower` → camera cúi; `head/front` → camera thẳng; chỉ giữ ống kính `/left/`).

### 1.2. Quỹ đạo

```
/kiss/odometry: odom_lidar → base_link
99 mẫu / 9.8 s (~10.1 Hz) · đường đi 4.58 m · yaw −96.1° → −96.1°
```

→ Robot **đi thẳng 4.58 m**, không rẽ. Điều này ảnh hưởng tới cách chọn sub-goal (mục 5, giai đoạn C).

---

## 2. Giải mã CDR — thay thế cho ROS

> **📦 CDR là gì?** *Common Data Representation* — cách ROS 2 xếp một message thành chuỗi byte.
> Không tự mô tả (không có tên field trong file), nên muốn đọc phải **biết trước cấu trúc message**.

`class CDR` trong script cài đúng **3 luật**:

1. **4 byte đầu** là *encapsulation header* `[0x00, endian, 0x00, 0x00]`; `endian` lẻ = little-endian.
2. **Căn lề**: mỗi số phải nằm ở offset **chia hết cho kích thước của nó**, tính từ **sau** 4 byte
   header. (`float64` phải ở offset chia hết cho 8 → chèn byte đệm nếu cần.)
   ⚠️ Đây là lỗi phổ biến nhất khi tự viết bộ giải mã: sai một byte đệm là **mọi số sau đó lệch hết**.
3. `string` = `uint32` độ dài (**kể cả ký tự NUL cuối**) + bytes. `sequence<T>` = `uint32` số phần tử
   + các phần tử.

Năm kiểu message được cài: `CameraInfo`, `CompressedImage`, `Odometry`, `TFMessage`, `PointCloud2`.
Muốn thêm kiểu khác → viết một hàm `msg_xxx(buf)` đọc các field **đúng thứ tự khai báo trong `.msg`**.

**Bằng chứng giải mã đúng** (nếu sai căn lề thì các số này sẽ vô nghĩa):

```
head_front  1920×1080  fx=fy=734.014  cx=969.061  cy=562.846   model=rational_polynomial  D=[0…0]
waist_front  960× 600  fx=fy=366.838  cx=482.257  cy=303.495
CompressedImage.format = "bgra8; jpeg compressed bgr8"   payload bắt đầu bằng ff d8 ff = JPEG ✅
```

> 🎨 **Chuyện màu BGR/RGB.** Chuỗi `format` ghi `bgr8` khiến ta tưởng phải đảo kênh R↔B. **Không.**
> `cv::imencode` nhận Mat BGR và ghi ra JPEG **đúng màu RGB**. Kiểm chứng bằng mắt: ảnh giải mã cho gỗ
> dán ra **màu gỗ**, còn bản đảo kênh cho gỗ ra **màu xanh xám** → bản không đảo mới đúng.
> Nếu bag của bạn do nguồn khác ghi và màu bị sai, dùng cờ `--bgr-swap`.

---

## 3. Cây TF — nơi giấu "camera cao bao nhiêu, cúi bao nhiêu độ"

> **📦 TF là gì?** Trong ROS, "camera ở đâu trên robot" **không** là một trường dữ liệu nào cả — nó là
> **tích của một chuỗi biến đổi cha→con**. Biến đổi cố định ở `/tf_static`, biến đổi động (khớp cổ,
> chân) ở `/tf` theo thời gian.

`TFBuffer.chain_to_root()` nhân dồn từ frame camera lên gốc cây; `lookup(target, source, t)` cho ra
`T_target←source` tại thời điểm `t` (biến đổi tĩnh thắng; biến đổi động lấy mẫu **gần `t` nhất**).

### 3.1. Bẫy: frame `_optical` **không có** trong cây TF

`frame_id` của ảnh là `waist_front_zed_onboard_left_camera_frame_optical`, nhưng `/tf_static` chỉ phát
`base_link → waist_front_zed_onboard_left_camera_frame` (**khung thân**). Đây là chuyện thường gặp với
driver ZED. `resolve_camera()` xử lý: bỏ hậu tố `_optical`, tra khung thân, rồi **tự nhân thêm phép
xoay chuẩn thân→quang học**:

```
p_optical = R_OPT_BODY · p_body,   R_OPT_BODY = [[0,-1,0],[0,0,-1],[1,0,0]]
      (thân: x trước, y trái, z lên)  →  (quang học: x phải, y xuống, z trục quang)
```

Phép xoay này **được xác nhận bằng chính bag**: với camera đầu, `/tf_static` có sẵn cặp
`head_…_optical → head_…_camera_frame` và ma trận xoay của nó đúng bằng `R_OPT_BODY` (sai số < 0.001).

### 3.2. Suy chiều cao so với **sàn**

`base_link` của robot hình người nằm ở **hông (pelvis)** — TF không nói nó cách sàn bao nhiêu.
`derive_base_height()` suy bằng cách tìm frame có tên chứa `ankle/foot/sole/toe`, lấy `z` thấp nhất so
với base, rồi trừ thêm `--foot-offset` (dày bàn chân, mặc định 4 cm):

```
left_ankle_roll_link  z = −0.787 m so với base_link
→ base_link cao ≈ 0.787 + 0.04 = 0.827 m so với sàn
```

### 3.3. Kết quả: hình học hai camera (suy hoàn toàn từ TF)

| | camera **đầu** (`pitch_1`) | camera **bụng** (`pitch_2`) |
|---|---|---|
| Ảnh | 1920×1080 | 960×600 |
| `fx = fy` | 734.0 | 366.8 |
| `cx, cy` | 969.1 · 562.8 | 482.3 · 303.5 |
| z so với `base_link` | +0.778 m | +0.151 m |
| **Cao so với sàn** | **1.605 m** | **0.978 m** |
| **Góc cúi** | **20.6°** | **30.0°** |

→ `setting = 98cm_30deg` (nhắc lại: `setting` luôn dùng **`pitch_2`** — camera nhìn cúi).

### 3.4. ⭐ Bằng chứng đẹp nhất: camera bụng **trùng khít** quy ước của `vln_ce`

Ma trận xoay của camera bụng tính từ TF, so với ma trận `pose.60cm_30deg` **đo thật** trong data gốc:

```
   tính từ TF của bag này          đo thật trong vln_ce (r2r/17DRP5sb8fy)
   [[ 0.   -0.5    0.866]          [[-0.   -0.5    0.866]
    [-1.    0.     0.   ]     ==    [-1.    0.    -0.   ]
    [ 0.   -0.866 -0.5  ]]          [ 0.   -0.866 -0.5  ]]
```

**Giống hệt.** Nghĩa là camera bụng của robot này **đúng vai "camera nhìn cúi 30°"** mà System 2 cần —
không phải chỉnh trục, không phải xoay bù gì cả. Đây cũng là lý do script chọn nó làm `pitch_2`.

Kiểm tra thêm hướng các trục (ở thời điểm giữa bag):

| Trục quang học | Mong đợi | Camera bụng | Camera đầu |
|---|---|---|---|
| `x` (phải) | `(0,−1,0)` = phải của robot | `(0, −1, 0)` ✅ | `(0.049, −0.999, −0.020)` ✅ |
| `y` (xuống) | thành phần z < 0 | `(−0.5, 0, −0.866)` ✅ | `(−0.352, 0.002, −0.936)` ✅ |
| `z` (trục quang) | hướng trước & hơi xuống | `(0.866, 0, −0.5)` ✅ | `(0.935, 0.053, −0.351)` ✅ |

→ Không camera nào bị lắp ngược.

---

## 4. Depth từ `PointCloud2` — vì log **không có** ảnh depth

Loader S2 **bắt buộc file depth tồn tại** ([04](04_data_train_s2.md) mục 6.3), nhưng bag này không có
topic ảnh depth. Thứ có sẵn là point cloud của ZED:

```
/camera/waist_front_zed_onboard/point_cloud/cloud_registered
448×256 điểm có tổ chức · field (x, y, z, rgb) float32 · point_step 16 · 1.84 MB/message @10 Hz
94 431 / 114 688 điểm hợp lệ · khoảng x (hướng trước) 0.80 … 5.00 m   ← ZED cắt ở 5 m
```

`cloud_to_depth_mm()` làm 3 bước:

1. Đọc `x, y, z` (hệ **thân** camera).
2. Xoay sang hệ **quang học** bằng `R_OPT_BODY` (mục 3.1).
3. Chiếu bằng `K` (đã hiệu chỉnh theo crop+resize), và với mỗi pixel **giữ điểm gần nhất**
   (`np.minimum.at`) → đúng quy tắc che khuất.
4. `--depth-fill` (bật sẵn): lấp lỗ bằng pixel hợp lệ **gần nhất** (`distance_transform_edt`), vì
   448×256 điểm không phủ kín 640×480.

Cuối cùng: `uint16`, **đơn vị milimét**, clip 10 000 — đúng hằng số của `vln_ce` (loader S2 chia 1000).

**Kiểm chứng depth đúng** (frame 6): giá trị 0.87 … 5.00 m, và trung bình theo hàng ảnh giảm dần đúng
theo hình học của camera cao 0.98 m cúi 30°:

| Vùng ảnh | Độ sâu trung bình |
|---|---|
| hàng 0–100 (xa, gần đường chân trời) | **4.41 m** |
| hàng 200–300 (giữa ảnh) | **1.89 m** |
| hàng 400–479 (sát chân robot) | **0.98 m** |

Đặt cạnh ảnh RGB, vật thể tối (gần) trong ảnh depth nằm **đúng vị trí** của màn hình máy tính bên trái
và thùng carton bên phải trong ảnh màu.

> Không có point cloud? Dùng `--depth-source zeros` → ghi ảnh uint16 toàn 0. **Đủ để train S2 thuần**
> (loader không dùng giá trị depth khi `pixel_goal_only=False`) nhưng **không dùng được cho train dual**.

---

## 5. Sáu giai đoạn (giống `mcap2s2.py`) và kết quả thật

```bash
python db32s2.py --bag vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s \
    --out ./traj_data --dataset-name vinbot --scene-id lab_run01 \
    --min-move 0.15 --subgoal-dist 1.0 \
    --instruction "Walk straight along the white walkway past the workbenches and stop at the end of the aisle."
```

### A. `read_mcap` → `Bag.open` + `index()` — đọc **hai lượt** để tiết kiệm RAM

Bag nặng 572 MB, ảnh + point cloud chiếm gần hết. Nếu nạp hết vào RAM là ~600 MB.
Cách làm: **lượt 1** chỉ lấy `(timestamp, file, rowid)` của topic nặng (`index()`, không tải blob) và
giải mã trọn các topic nhẹ (`odom`, `tf`, `camera_info`); **lượt 2** chỉ `fetch()` đúng những blob của
keyframe đã chọn. → RAM chỉ vài chục MB, cả pipeline chạy **5 giây**.

`Bag` cũng ghép **nhiều file `.db3`** của cùng một bag (`..._0.db3`, `..._1.db3`…) thành một dòng thời gian.

### B. `sync_frames` — đồng bộ + chọn keyframe

Nhịp chính = **luồng ảnh cúi (`pitch_2`)**, vì nhãn `goal` gắn vào ảnh đó. Với mỗi ảnh tại `t`: tìm
ảnh thẳng / odometry gần nhất; lệch quá `--tol-ms` (60 ms) → **bỏ frame** (thà mất frame còn hơn ghép
pose của thời điểm khác — sai kiểu này **không báo lỗi** lúc train). Point cloud được nới ngưỡng lên
120 ms vì nó chỉ 10 Hz.

Rồi **lọc keyframe**: bỏ frame robot gần như đứng yên so với frame trước.

```
ảnh cúi trong bag : 149
keyframe giữ lại  : 26
bỏ do lệch giờ    : 1
bỏ do đứng yên    : 122   (ngưỡng 0.15 m / 10°)
bước giữa keyframe: trung bình 0.177 m   (data gốc R2R = 0.25 m)
```

> 📌 Vì sao phải lọc? Ảnh 15 Hz mà robot đi ~0.03 m/frame → 149 frame gần trùng nhau. Data gốc R2R có
> **bước 0.25 m**; giữ khoảng bước tương tự giúp khái niệm "waypoint kế tiếp" của model khớp với lúc
> train gốc. Mặc định `--min-move 0.25`; bag 10 s này quá ngắn nên ví dụ dùng 0.15 để có nhiều mẫu hơn.

### C. `make_labels` — sinh nhãn

**`action`**: từ Δyaw giữa 2 keyframe → `2` (trái) / `3` (phải) / `1` (tiến); `action[0] = -1`.
Bag mẫu đi thẳng nên ra `start×1 ↑ tiến×25`.

**`pose.{setting}`**: mặc định `--pose-mode tf` → **pose thật**
`T_world←camera = T_world←base(t) · T_base←camera(t)`, với gốc world **dịch lên `+0.827 m` để sàn thành
`z = 0`**. (`--pose-mode synth` thì dựng lại theo khuôn lý tưởng của `vln_ce` từ `(x, y, yaw)` +
chiều cao/góc cúi danh nghĩa — dùng khi muốn khớp tuyệt đối giả định của loader.)

**`goal` + `relative_goal_frame_id`**: chiếu **vị trí chân robot ở frame tương lai `g`, trên sàn
(z = 0)** vào ảnh hiện tại. Chi tiết công thức: [06](06_pipeline_mcap_to_s2.md) giai đoạn C.3.

🔑 **Khác biệt quan trọng so với `mcap2s2.py` — cách chọn sub-goal.** Bản `.mcap` chỉ dùng luật "điểm
rẽ". Bag thật này **đi thẳng 4.58 m không rẽ lần nào** → luật đó chỉ cho **một** sub-goal ở cuối, `k`
lên tới hàng trăm frame (cửa sổ ảnh khổng lồ, lệch hẳn data gốc nơi `k` chỉ cỡ 4–25). Nên
`find_subgoal_frames()` ở đây cộng thêm **luật thứ hai**:

| Luật | Ý nghĩa |
|---|---|
| 1. Điểm rẽ | frame kết thúc một đoạn đi thẳng — *"đi thẳng tới chỗ rẽ"* |
| 2. **Mỗi `--subgoal-dist` mét đường đi** | phòng trường hợp đi thẳng dài (data gốc R2R có viewpoint mỗi ~1–2 m → mặc định 1.5 m) |
| 3. Frame cuối episode | luôn có |

Cộng thêm `--max-goal-frames` (mặc định 30) chặn trên cho `relative_goal_frame_id`.

**Nhãn thật sinh ra (episode 0, `--subgoal-dist 1.0`):**

```
action : [-1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, …]
rel_id : [ 6, 5, 4, 3, 2,-1, 6, 5, 4, 3, 2,-1, 4, 3, 2,-1,-1]
goal   : [[307,243],[304,272],[310,308],[318,364],[324,432],[-1,-1],
          [318,249],[313,277],[307,309],[313,360],[316,446],[-1,-1],[315,320],…]
```

Đọc bảng: `u ≈ 310–325` (giữa ảnh, vì robot đi thẳng về phía đích) · `v` **tăng dần** 243 → 432 (càng
gần, điểm đích càng trôi xuống dưới ảnh) · `-1` xuất hiện khi waypoint đã quá gần, rơi khỏi mép dưới →
đúng mẫu hình của data gốc. **Vẽ chấm đỏ lên ảnh: nó nằm đúng trên lối đi trắng phía trước robot.**

### D. `write_images` — ghi ảnh + hiệu chỉnh `K`

Ảnh nguồn 960×600 (tỉ lệ 1.60) khác đích 640×480 (1.333). `--out-fit`:

| Chế độ | Cách làm | Đánh đổi |
|---|---|---|
| **`crop`** (mặc định) | cắt giữa còn 800×600 rồi resize → **không méo** | mất góc nhìn ngang: hfov 105.7° → **95.4°** (gần data gốc ~90° hơn) |
| `stretch` | resize thẳng, giữ toàn bộ góc nhìn | ảnh bị méo (fx=244.6 ≠ fy=293.5) |

🚨 **`K` phải đi theo đúng phép cắt + resize**, nếu không `goal` sẽ lệch:
```
cắt   : cx −= x0 ,  cy −= y0
resize: fx,cx ×(out_w/src_w) ,  fy,cy ×(out_h/src_h)
```
Kết quả với `crop`: `fx = fy = 293.5 · cx = 321.8 · cy = 242.8` (fx = fy → đúng, không méo).

### E. `write_parquet` + `write_meta`

dtype khớp bản gốc (`action` int32 · `pose` `list<list<float32>>` · `goal` `fixed_size_list<int32>[2]`
· `rel_id` int32) — lý do: [04](04_data_train_s2.md) mục 3.3.

📌 Cột `timestamp` ghi **thời gian thật lấy từ bag** (`0, 0.536, 1.139, 1.676, …` giây), **không phải
`i/fps`** — vì keyframe đã lọc thưa nên khoảng cách giữa các frame **không đều**. Đây chính là lý do
tài liệu luôn dặn *"dùng cột `timestamp`, đừng suy từ `fps`"*.

### F. `self_check`

```
mẫu pixel_goal : 4    ← loại quan trọng nhất
mẫu turn       : 0    (bag không có cú rẽ nào)
mẫu stop       : 1
bị bỏ (k < 3)  : 0
file ảnh thiếu : 0
→ train S2  (pixel_goal_only=False): 9 mẫu
→ train dual (pixel_goal_only=True): 4 mẫu
```

---

## 6. Kiểm định bằng **chính loader thật**

Đừng tin script của mình — để code của dự án phán quyết ([06](06_pipeline_mcap_to_s2.md) mục 4 giải
thích mẹo giả lập `decord`/`torchcodec`):

```python
L.data_dict['vinbot_98cm_21_30'] = {"data_path": "traj_data/vinbot",
                                    "height": 98, "pitch_1": 21, "pitch_2": 30}
print(len(L.NavPixelGoalDataset(tokenizer=None, data_args=a)))
```

**Kết quả thật: `9` (S2) và `4` (dual) — trùng khớp giai đoạn F** → dataset đúng hợp đồng dữ liệu.

Round-trip quy ước hệ toạ độ, chạy `get_trajectory_relative_to_frame(poses, camera_deg=30)`:

```
[[0.    0.    -0.  ]      ← frame gốc đúng ở (0,0,0)
 [0.185 0.005 -0.019]     ← tiến 0.185 m, lệch ngang 5 mm
 [0.357 0.017  0.001]
 [0.523 0.022  0.023]
 [0.709 0.020  0.043] …]  ← x tăng đều, y ≈ 0 → đi thẳng ✅
```

(Chiều cao camera đọc lại từ parquet: `0.966 m` — dao động nhẹ quanh 0.978 m vì robot hình người
nhấp nhô khi bước. Đó là **dữ liệu thật**, không phải lỗi.)

---

## 7. Áp dụng cho bag khác — cần chỉnh gì

### 7.1. Nếu tự đoán topic sai

```bash
python db32s2.py --bag <bag> \
  --rgb-down-topic  /camera/.../image/compressed  --caminfo-down-topic  /camera/.../camera_info \
  --rgb-front-topic /camera/.../image/compressed  --caminfo-front-topic /camera/.../camera_info \
  --depth-topic /camera/.../points  --odom-topic /odom
```

### 7.2. Bảng tình huống

| Tình huống | Cờ cần dùng |
|---|---|
| Robot chỉ có **1 camera** | `--single-camera` → dùng camera cúi cho cả `pitch_1` và `pitch_2` (loader vẫn mở 2 đường dẫn nên script ghi cùng nội dung ra cả hai) |
| TF không có khớp chân (robot bánh xe) | `derive_base_height` trả 0 → **khai tay** `--height-cm 60` |
| Muốn ép đúng cấu hình data gốc | `--height-cm 60 --pitch1 30 --pitch2 30` + `--pose-mode synth` |
| Bag dài nhiều phút | `--split-sec 60` để cắt thành nhiều episode |
| Mỗi episode một câu lệnh khác | `--instruction-file ins.json` với `{"0": "…", "1": "…"}` |
| Ảnh là `sensor_msgs/Image` thô (không nén) | ⚠️ **cần viết thêm** một hàm `msg_image()` đọc `encoding` + `data` rồi dựng numpy (script hiện chỉ xử lý `CompressedImage`) |
| Pose chỉ có trong `/tf` (không có topic Odometry) | ⚠️ **cần sửa**: thay `bag.read_all(odom)` bằng `tf.lookup(world_frame, base_frame, t)` |
| Bag nén (`compression_mode: file`) | ⚠️ giải nén trước (`ros2 bag decompress`) — script đọc SQLite thô |

### 7.3. Câu lệnh ngôn ngữ — thứ **bắt buộc** và **không có trong bag**

Log robot không chứa "câu lệnh điều hướng". Không có nó thì mẫu train **vô nghĩa** (S2 học từ ngôn
ngữ), nên script **dừng hẳn** với thông báo hướng dẫn nếu thiếu. Hãy viết câu lệnh tiếng Anh mô tả
đúng lộ trình robot đã đi, ví dụ với bag mẫu:

> *"Walk straight along the white walkway past the workbenches and stop at the end of the aisle."*

---

## 8. Bảng rủi ro

| Rủi ro | Hậu quả | Cách phát hiện |
|---|---|---|
| Sai căn lề CDR | mọi số vô nghĩa | `--inspect`: `fx/cx/cy` phải hợp lý, `format` phải đọc được, JPEG phải bắt đầu `ff d8 ff` |
| Frame `_optical` không có trong TF | crash hoặc pose sai | script in `(+xoay thân→quang học)` để bạn biết nó đã tự bù |
| `base_link` ở hông, không phải sàn | `height_cm` sai vài chục cm | `--inspect` in chiều cao suy được; đối chiếu bằng thước |
| Camera `pitch_2 = 0°` | `goal` toàn `-1`, 0 mẫu pixel_goal | script cảnh báo ngay + giai đoạn F báo `⚠️ KHÔNG có mẫu pixel_goal` |
| `K` không hiệu chỉnh theo crop/resize | `goal` lệch chỗ | vẽ chấm `goal` lên ảnh xem có nằm trên lối đi |
| Đảo kênh màu sai | ảnh sai màu, model học kém | xem thử một ảnh: vật gỗ phải ra màu gỗ |
| Point cloud lệch giờ với ảnh | depth lệch pha | script đếm frame `depth = 0`; giảm ngưỡng nếu cần |
| Bag đi thẳng, không rẽ | chỉ 1 sub-goal, `k` khổng lồ | giảm `--subgoal-dist`; giai đoạn C in số sub-goal |
| Đường dẫn output chứa chữ `rgb` | loader dựng đường dẫn depth sai | script cảnh báo |

---

## 9. Hạn chế của bag mẫu (nói rõ để không kỳ vọng sai)

| Hạn chế | Ảnh hưởng |
|---|---|
| Chỉ **10 s / 4.58 m / đi thẳng** | Ra **4 mẫu pixel_goal** — đủ để *chứng minh pipeline đúng*, **không đủ để train**. Cần hàng nghìn mẫu → hàng chục–trăm bag có rẽ, có đích đa dạng. |
| **Không có cú rẽ nào** | Không sinh được mẫu *turn* → model không học được lệnh `←/→` từ bag này. |
| Không có câu lệnh trong log | Phải viết tay cho từng lượt đi. |
| ZED cắt depth ở **5 m** | Vùng xa đều = 5000 mm. Với train S2 thuần thì không sao. |
| Hai camera ở **hai độ cao khác nhau** (1.60 m và 0.98 m) | Data gốc `vln_ce` dùng cặp `pitch_1/pitch_2` **cùng độ cao**. Cặp lệch độ cao vẫn train được (loader không kiểm tra), nhưng lệch phân bố so với data gốc → nếu ưu tiên khớp tuyệt đối, dùng `--single-camera` (chỉ camera bụng). |

---

## 10. Chạy trọn vẹn (copy-paste)

```bash
cd InternNav/internnav-s2-setup/docs/training_data_guide/tools
pip install numpy pillow pyarrow scipy

# khảo sát
python db32s2.py --bag vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s --inspect

# sinh data
python db32s2.py --bag vslam_nav_test_bag_2026_07_29-13_59_45_10s_20s \
    --out ./traj_data --dataset-name vinbot --scene-id lab_run01 \
    --min-move 0.15 --subgoal-dist 1.0 \
    --instruction "Walk straight along the white walkway past the workbenches and stop at the end of the aisle."
```

Sau đó đăng ký vào `data_dict` + trỏ `--vln_dataset_use` như [06](06_pipeline_mcap_to_s2.md) mục 5
(script in sẵn hai dòng cần dán).

---

*Quay lại mục lục: [00_README](00_README.md).*
