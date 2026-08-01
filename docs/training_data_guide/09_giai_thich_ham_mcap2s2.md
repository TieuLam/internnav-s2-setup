# 09 — Giải thích **từng hàm** trong `tools/mcap2s2.py`

> **File này để làm gì:** [06](06_pipeline_mcap_to_s2.md) mô tả *pipeline* theo 6 giai đoạn. File này
> đi xuống một tầng nữa — **mổ từng hàm một**: nó nhận gì, trả gì, làm gì trong từng dòng, **vì sao**
> phải làm thế, và **phải sửa chỗ nào** khi bạn thay bằng `.mcap` của robot thật.
>
> Dành cho lúc bạn cần *sửa* script chứ không chỉ *chạy* script.
>
> Đọc trước: [01_thuat_ngu](01_thuat_ngu.md) (nếu gặp từ lạ) ·
> [04_data_train_s2](04_data_train_s2.md) (hợp đồng dữ liệu mà script này phải tuân theo).

---

## 0. Bản đồ toàn bộ hàm

Script có **19 hàm/property** + **4 cấu trúc dữ liệu**. Bảng này là mục lục — click vào dòng để nhảy
tới phần giải thích.

| # | Hàm | Dòng | Giai đoạn | Một câu |
|---|---|---|---|---|
| — | `RawStreams` | [84](tools/mcap2s2.py#L84) | A | túi đựng dữ liệu thô đọc từ mcap |
| 1 | [`read_mcap`](#1-read_mcap--đọc-mcap-một-lượt) | [95](tools/mcap2s2.py#L95) | A | đọc mcap → 6 list `(thời gian, payload)` |
| — | `Frame` / `EpisodeData` | [138](tools/mcap2s2.py#L138) / [149](tools/mcap2s2.py#L149) | B | một "khoảnh khắc" đã ghép đủ ảnh+pose / một lượt đi |
| 2 | [`_nearest`](#2-_nearest--tìm-message-gần-thời-điểm-t-nhất) | [155](tools/mcap2s2.py#L155) | B | tìm nhị phân message gần `t` nhất |
| 3 | [`_quat_to_yaw`](#3-_quat_to_yaw--quaternion--góc-quay) | [168](tools/mcap2s2.py#L168) | B | quaternion 4 số → 1 góc yaw |
| 4 | [`_payload_bytes`](#4-_payload_bytes--lấy-byte-ảnh-ra-khỏi-json) | [174](tools/mcap2s2.py#L174) | B | base64 → bytes ảnh |
| 5 | [`sync_frames`](#5-sync_frames--đồng-bộ--cắt-episode--lọc-keyframe) | [179](tools/mcap2s2.py#L179) | B | ghép 4 luồng theo thời gian, cắt episode, bỏ frame đứng yên |
| 6 | [`camera_pose_from_base`](#6-camera_pose_from_base--vị-trí-robot--ma-trận-camera-4×4) | [262](tools/mcap2s2.py#L262) | C | `(x,y,yaw)` → ma trận 4×4 camera→world |
| 7 | [`project_to_pixel`](#7-project_to_pixel--điểm-3d--toạ-độ-pixel) | [276](tools/mcap2s2.py#L276) | C | điểm 3D → `(u,v)` trên ảnh |
| 8 | [`discretize_actions`](#8-discretize_actions--quỹ-đạo-liên-tục--nút-bấm) | [296](tools/mcap2s2.py#L296) | C | quỹ đạo → chuỗi `{-1,1,2,3}` |
| 9 | [`find_subgoal_frames`](#9-find_subgoal_frames--chọn-đích-trung-gian) | [314](tools/mcap2s2.py#L314) | C | chọn các frame làm "đích trung gian" |
| 10 | [`make_labels`](#10-make_labels--ráp-4-cột-nhãn) | [334](tools/mcap2s2.py#L334) | C | gọi 4 hàm trên → 4 cột nhãn |
| 11 | [`write_images`](#11-write_images--ghi-3-luồng-ảnh) | [364](tools/mcap2s2.py#L364) | D | ghi ảnh đúng tên thư mục loader mong đợi |
| 12 | [`_save_jpeg`](#12-_save_jpeg--ghi-ảnh-màu) | [387](tools/mcap2s2.py#L387) | D | ghi jpg (giữ nguyên byte nếu đã là jpeg) |
| 13 | [`_save_depth_png`](#13-_save_depth_png--ghi-ảnh-độ-sâu--có-chốt-chặn) | [396](tools/mcap2s2.py#L396) | D | ghi png uint16, **chặn sai đơn vị** |
| 14 | [`write_parquet`](#14-write_parquet--ghi-bảng-số) | [411](tools/mcap2s2.py#L411) | E | ghi bảng số 1 episode, **đúng dtype** |
| 15 | [`write_meta`](#15-write_meta--ghi-4-file-mô-tả) | [444](tools/mcap2s2.py#L444) | E | ghi 4 file `meta/` |
| 16 | [`self_check`](#16-self_check--đóng-vai-loader-để-tự-chấm-điểm) | [522](tools/mcap2s2.py#L522) | F | đóng vai loader, đếm số mẫu train |
| 17 | [`Config` + 4 property](#17-config--4-property--nơi-sinh-ra-chuỗi-setting) | [593](tools/mcap2s2.py#L593) | — | gom tham số, sinh chuỗi `setting` |
| 18 | [`main`](#18-main--nhạc-trưởng) | [620](tools/mcap2s2.py#L620) | — | nhạc trưởng: đọc cờ, gọi A→F |
| 19 | [`_hist`](#19-_hist--đếm-số-lần-mỗi-action) | [718](tools/mcap2s2.py#L718) | — | in thống kê action cho dễ nhìn |

Sơ đồ ai gọi ai:

```
main
 ├── read_mcap                                        ← A
 ├── sync_frames ──┬── _nearest                       ← B
 │                 ├── _quat_to_yaw
 │                 └── _payload_bytes
 ├── (vòng lặp mỗi episode)
 │    ├── make_labels ──┬── discretize_actions        ← C
 │    │                 ├── find_subgoal_frames
 │    │                 ├── camera_pose_from_base
 │    │                 └── project_to_pixel
 │    ├── write_images ─┬── _save_jpeg                ← D
 │    │                 └── _save_depth_png
 │    ├── write_parquet                               ← E
 │    └── _hist  (chỉ để in ra màn hình)
 ├── write_meta                                       ← E
 └── self_check                                       ← F
```

---

## 0.5. Bốn cấu trúc dữ liệu (đọc trước, các hàm đều xoay quanh chúng)

### `RawStreams` — [dòng 84-92](tools/mcap2s2.py#L84)

```python
@dataclass
class RawStreams:
    rgb_front: List[Tuple[int, dict]]   # [(thời_gian_ns, payload_json), ...]
    rgb_down:  List[Tuple[int, dict]]
    depth:     List[Tuple[int, dict]]
    pose:      List[Tuple[int, dict]]
    caminfo:   List[Tuple[int, dict]]
    episodes:  List[Tuple[int, dict]]
    metadata:  Dict[str, str]
```

> 🧠 **`@dataclass` là gì?** Một cách viết tắt của Python để tạo class chỉ dùng để *đựng dữ liệu*.
> Bạn khai báo tên trường + kiểu, Python tự sinh `__init__`. Lợi ích thật sự: gõ `raw.rgb_front`
> thì IDE gợi ý được và gõ sai tên sẽ báo lỗi ngay — trong khi `raw["rgb_front"]` (dict) thì gõ sai
> chỉ vỡ lúc chạy.

Đây là **dữ liệu thô**: mỗi luồng là một danh sách các cặp *(dấu thời gian nano-giây, nội dung
message)*, **đã sắp xếp tăng dần theo thời gian** — điều kiện bắt buộc để `_nearest` tìm nhị phân được.

### `Frame` — [dòng 138-146](tools/mcap2s2.py#L138)

```python
@dataclass
class Frame:
    t_ns: int          # thời điểm (lấy theo ảnh RGB nhìn thẳng)
    rgb_front: bytes   # byte ảnh jpeg nhìn thẳng
    rgb_down:  bytes   # byte ảnh jpeg nhìn cúi
    depth_png: bytes   # byte ảnh png độ sâu
    x: float           # vị trí robot trên sàn (mét)
    y: float
    yaw: float         # hướng robot (radian)
```

Một `Frame` = **một khoảnh khắc đã ghép đủ** ảnh + độ sâu + vị trí. Đây là "đơn vị nguyên tử" của
toàn bộ phần sau: 1 `Frame` → 1 hàng parquet + 3 file ảnh.

### `EpisodeData` — [dòng 149-152](tools/mcap2s2.py#L149)

```python
@dataclass
class EpisodeData:
    instruction: str        # câu lệnh tiếng Anh
    frames: List[Frame]     # chuỗi frame của một lượt đi
```

Một **episode** = một lượt đi trọn vẹn ứng với một câu lệnh. `EpisodeData` ghép hai thứ đó lại.

### `Config` — [dòng 593](tools/mcap2s2.py#L593) → xem [mục 17](#17-config--4-property--nơi-sinh-ra-chuỗi-setting)

---

# GIAI ĐOẠN A — ĐỌC MCAP

## 1. `read_mcap` — đọc mcap một lượt

📍 [tools/mcap2s2.py:95-132](tools/mcap2s2.py#L95)

```python
def read_mcap(path: str, topics: Dict[str, str]) -> RawStreams:
```

| | |
|---|---|
| **Nhận** | `path` = đường dẫn file `.mcap`; `topics` = dict `{tên_nội_bộ: tên_topic_thật}` |
| **Trả** | một `RawStreams` |
| **Việc chính** | quét file **đúng một lượt**, phân loại message vào 6 giỏ, sắp xếp theo thời gian |

### Từng bước

**Bước 1 — dựng "giỏ" và bảng tra ngược** ([103-104](tools/mcap2s2.py#L103))

```python
buckets = {k: [] for k in topics}          # {"rgb_front": [], "rgb_down": [], ...}
wanted  = {v: k for k, v in topics.items()}  # {"/camera/front/image_raw": "rgb_front", ...}
```

`topics` ánh xạ *tên nội bộ → tên topic*; `wanted` là **bản lật ngược** *tên topic → tên nội bộ*.
Cần bản lật ngược vì khi đọc message ta chỉ biết `channel.topic`, phải tra ngược ra "đây là luồng gì".

> 💡 Tra bằng dict là **O(1)** — nếu duyệt `for k, v in topics.items(): if v == channel.topic` thì với
> log thật hàng trăm nghìn message sẽ chậm hơn hẳn. Đây là kiểu tối ưu "rẻ tiền mà đáng làm".

**Bước 2 — quét file** ([107-110](tools/mcap2s2.py#L107))

```python
with open(path, "rb") as f:
    reader = make_reader(f)
    for _, channel, message in reader.iter_messages(topics=list(wanted.keys())):
        buckets[wanted[channel.topic]].append((message.log_time, json.loads(message.data)))
```

- `iter_messages(topics=[...])` — chỉ trả về message của các topic ta quan tâm, thư viện tự bỏ qua
  phần còn lại → **không tốn công giải mã** những topic thừa (log robot thật có thể có 50 topic).
- `message.log_time` — dấu thời gian **nano-giây** (`1 giây = 1_000_000_000 ns`, hằng số `NS` ở
  [dòng 77](tools/mcap2s2.py#L77)).
- `json.loads(message.data)` — **đây là chỗ phụ thuộc định dạng**. File demo dùng schema JSON của
  Foxglove nên `message.data` là chuỗi JSON. ⚠️ Log ROS 2 thật mã hoá **CDR (nhị phân)** → dòng này sẽ
  vỡ. Cách sửa: `pip install mcap-ros2-support` rồi thay bằng decoder của nó
  ([06 mục 7.2](06_pipeline_mcap_to_s2.md)).

**Bước 3 — đọc metadata** ([111-114](tools/mcap2s2.py#L111))

```python
f.seek(0)
for record in reader.iter_metadata():
    if record.name == "s2_profile":
        metadata = dict(record.metadata)
```

`f.seek(0)` = **tua file về đầu**. Cần thiết vì vòng lặp trên đã đọc tới cuối file; muốn quét lại loại
record khác thì phải quay lại đầu.

`s2_profile` là một "tờ khai" **tuỳ chọn** mà bên ghi log có thể nhét sẵn vào file: chiều cao camera,
hai góc cúi, kích thước ảnh, đơn vị depth, fps. Nếu có, `main` **không phải đoán** các thông số này.
Log robot thật hầu như không bao giờ có → bạn phải truyền tay bằng `--height-cm --pitch1 --pitch2
--fps`. (Bản `.db3` giải bài này theo cách khác: **suy thẳng từ cây TF** — xem
[06c](06c_pipeline_db3_to_s2.md) mục 3.)

**Bước 4 — sắp xếp** ([116-117](tools/mcap2s2.py#L116))

```python
for v in buckets.values():
    v.sort(key=lambda m: m[0])
```

`m[0]` là dấu thời gian. Sắp xếp **là điều kiện sống còn** cho `_nearest` — tìm nhị phân trên mảng
chưa sắp xếp cho ra kết quả sai một cách âm thầm.

> 📌 mcap *thường* đã lưu message theo thứ tự thời gian, nhưng khi có nhiều kênh ghi song song hoặc
> file bị nối lại thì không chắc. Sort lại là bảo hiểm rẻ tiền.

**Bước 5 — in thống kê** ([119-123](tools/mcap2s2.py#L119)) — để bạn phát hiện ngay "topic X có 0
message" thay vì để lỗi lòi ra ở tận giai đoạn D.

---

# GIAI ĐOẠN B — ĐỒNG BỘ THỜI GIAN

## 2. `_nearest` — tìm message gần thời điểm `t` nhất

📍 [tools/mcap2s2.py:155-165](tools/mcap2s2.py#L155)

```python
def _nearest(stream, times: np.ndarray, t: int) -> Optional[Tuple[int, dict]]:
    if not stream:
        return None
    i = int(np.searchsorted(times, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(stream)]
    return min((stream[j] for j in cands), key=lambda m: abs(m[0] - t))
```

> 🧠 **Vì sao cần hàm này?** Camera chụp 10 Hz, cảm biến vị trí phát 50 Hz — **không bao giờ trùng
> đúng mốc thời gian**. Muốn biết "lúc chụp tấm ảnh này robot đang ở đâu" thì phải đi tìm bản tin
> vị trí *gần nhất*.

**`np.searchsorted(times, t)`** trả về **vị trí chèn**: chỉ số `i` sao cho chèn `t` vào đó thì mảng
vẫn tăng dần. Nó dùng **tìm nhị phân** — mảng 1 triệu phần tử chỉ mất ~20 phép so sánh thay vì
1 triệu.

Ví dụ `times = [100, 200, 300, 400]`, `t = 260`:

```
searchsorted → i = 2          (chèn 260 vào giữa 200 và 300)
ứng viên     → j ∈ {1, 2} → times[1]=200 (lệch 60), times[2]=300 (lệch 40)
min          → chọn 300  ✅
```

Phải xét **cả hai** ứng viên `i-1` và `i` vì `searchsorted` luôn trả về phần tử *bên phải*, mà phần
tử bên trái có thể gần hơn (như ví dụ ngược lại: `t = 210` → phải chọn 200).

Điều kiện `0 <= j < len(stream)` chặn hai biên: `t` nhỏ hơn mọi mốc (`i=0` → `i-1 = -1`, mà Python
hiểu `-1` là *phần tử cuối* → sẽ chọn nhầm message cuối file!) hoặc `t` lớn hơn mọi mốc
(`i = len` → vượt biên).

⚠️ **Tham số `times` truyền từ ngoài vào chứ không tính bên trong.** Nếu viết
`times = np.array([m[0] for m in stream])` ngay trong hàm thì mỗi lần gọi phải duyệt lại cả mảng →
với `N` ảnh × `M` bản tin pose thì chi phí thành **O(N×M)** thay vì `O(N·log M)`. Với log thật, đây
là khác biệt giữa "chạy 5 giây" và "chạy 20 phút". Xem chỗ tính sẵn ở
[dòng 187-189](tools/mcap2s2.py#L187).

## 3. `_quat_to_yaw` — quaternion → góc quay

📍 [tools/mcap2s2.py:168-171](tools/mcap2s2.py#L168)

```python
def _quat_to_yaw(q: dict) -> float:
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
```

> 🧠 **Quaternion là gì?** Là cách biểu diễn *hướng xoay trong không gian 3D bằng 4 số*. ROS luôn
> dùng quaternion (thay vì 3 góc roll/pitch/yaw) vì nó không bị **gimbal lock** — hiện tượng mất một
> bậc tự do khi hai trục xoay trùng nhau. Cái giá là con người **không đọc hiểu trực tiếp được**
> 4 số đó.

Robot chạy trên mặt sàn nên chỉ cần **một góc duy nhất: yaw** (quay quanh trục thẳng đứng z). Công
thức trên là dòng "trích yaw" chuẩn trong bảng chuyển quaternion → Euler.

| Ký hiệu | Ý nghĩa |
|---|---|
| `yaw = 0` | robot hướng theo trục **+x** của thế giới |
| `yaw = +π/2` | quay **trái** 90° (theo quy tắc bàn tay phải, nhìn từ trên xuống) |
| `atan2(a, b)` | arctang "thông minh": tự xác định đúng góc phần tư, trả về khoảng `(-π, π]` |

Dùng `atan2` chứ **không** dùng `atan(a/b)` — `atan` không phân biệt được hướng `135°` với `-45°`.

## 4. `_payload_bytes` — lấy byte ảnh ra khỏi JSON

📍 [tools/mcap2s2.py:174-176](tools/mcap2s2.py#L174)

```python
def _payload_bytes(payload: dict) -> bytes:
    return base64.b64decode(payload["data"])
```

JSON là **định dạng văn bản** — không nhét trực tiếp byte nhị phân của một tấm ảnh vào được.
Giải pháp chuẩn: **base64** — mã hoá mỗi 3 byte nhị phân thành 4 ký tự chữ-số an toàn. Hàm này làm
việc ngược lại: chuỗi base64 → byte ảnh JPEG/PNG gốc.

⚠️ Cái giá của base64: **phình ~33%**. Đó là lý do file mcap demo 78 frame đã 3.8 MB. Log robot thật
(mã hoá nhị phân) không có vấn đề này.

## 5. `sync_frames` — đồng bộ + cắt episode + lọc keyframe

📍 [tools/mcap2s2.py:179-256](tools/mcap2s2.py#L179) — **hàm dài nhất giai đoạn B, làm 3 việc**

```python
def sync_frames(raw, tol_ms, min_move, min_turn_deg) -> List[EpisodeData]:
```

| | |
|---|---|
| **Nhận** | `raw` (dữ liệu thô); `tol_ms` ngưỡng lệch giờ; `min_move` (m) và `min_turn_deg` (độ) để lọc frame đứng yên |
| **Trả** | danh sách `EpisodeData` — mỗi phần tử là một lượt đi đã ghép đủ dữ liệu |

### Việc 1 — Tìm ranh giới episode ([191-203](tools/mcap2s2.py#L191))

```python
for t, p in raw.episodes:
    if p.get("event") == "start":
        open_ep = (t, p.get("instruction", ""))
    elif p.get("event") == "end" and open_ep is not None:
        spans.append((open_ep[0], t, open_ep[1]))
        open_ep = None
```

Đây là mẫu **"máy trạng thái mở–đóng"**: gặp `start` thì ghi nhớ (mở ngoặc), gặp `end` thì chốt lại
thành một khoảng `(t_bắt_đầu, t_kết_thúc, câu_lệnh)` (đóng ngoặc). Điều kiện `open_ep is not None`
chống trường hợp log bị cắt cụt, có `end` mà thiếu `start`.

Không có marker nào → coi **cả file là 1 episode** ([200-203](tools/mcap2s2.py#L200)) — đúng tình
huống log robot thật, vốn không biết khái niệm "episode".

### Việc 2 — Đồng bộ 4 luồng ([211-226](tools/mcap2s2.py#L211))

> **Nguyên tắc vàng: chọn MỘT luồng làm nhịp chính.** Ở đây là **RGB nhìn thẳng**. Với mỗi ảnh tại
> thời điểm `t`, đi tìm ảnh cúi / depth / pose gần `t` nhất. Nếu bạn không chọn nhịp chính mà cố
> "ghép cặp mọi thứ với mọi thứ", số frame sinh ra sẽ phụ thuộc luồng nào phát dày nhất — vô nghĩa.

```python
if not (ep_start - tol_ns <= t <= ep_end + tol_ns):
    continue                                    # ảnh nằm ngoài episode → bỏ
m_down  = _nearest(raw.rgb_down, t_down, t)
m_depth = _nearest(raw.depth,    t_depth, t)
m_pose  = _nearest(raw.pose,     t_pose,  t)
if m_down is None or m_depth is None or m_pose is None:
    dropped_sync += 1; continue                 # luồng rỗng
if max(abs(m_down[0]-t), abs(m_depth[0]-t), abs(m_pose[0]-t)) > tol_ns:
    dropped_sync += 1; continue                 # lệch quá ngưỡng → BỎ frame
```

Dòng `max(...) > tol_ns` là **chốt chặn quan trọng nhất của cả script**:

> ⚠️ Ghép nhầm ảnh của thời điểm này với pose của thời điểm khác sẽ tạo ra **nhãn sai mà không có
> bất kỳ thông báo lỗi nào** — model vẫn train được, loss vẫn giảm, chỉ là học sai. Đây là kiểu hỏng
> nguy hiểm nhất trong làm data. **Thà mất frame còn hơn ghép sai.**

`tol_ms` mặc định 60 ms ([dòng 636](tools/mcap2s2.py#L636)). Robot đi 0.5 m/s thì 60 ms ≈ 3 cm sai
số vị trí — chấp nhận được. Robot chạy nhanh hơn → **giảm** ngưỡng này.

### Việc 3 — Lọc keyframe ([229-235](tools/mcap2s2.py#L229))

```python
if frames:
    prev = frames[-1]
    d = math.hypot(pos["x"] - prev.x, pos["y"] - prev.y)
    dyaw = abs(math.atan2(math.sin(yaw - prev.yaw), math.cos(yaw - prev.yaw)))
    if d < min_move and math.degrees(dyaw) < min_turn_deg:
        dropped_static += 1
        continue
```

Log robot thật đầy những đoạn robot **đứng im** (chờ lệnh, chờ người tránh đường) — hàng chục frame
giống hệt nhau. Giữ lại thì:
- phình dung lượng vô ích,
- và tệ hơn: `discretize_actions` sẽ gán chúng thành `1 (tiến)` dù robot **không hề tiến** → dạy
  model điều sai.

Hai chi tiết đáng học:

1. **`math.hypot(dx, dy)`** = `sqrt(dx² + dy²)` nhưng chống tràn số. Đây là khoảng cách Euclid.
2. **`atan2(sin(Δ), cos(Δ))`** là **mẹo chuẩn hoá góc** về khoảng `(-π, π]`. Vì sao cần? Nếu
   `yaw` đi từ `+179°` sang `-179°`, phép trừ thẳng cho `-358°` — nhìn như quay gần trọn vòng, thực
   tế chỉ quay **2°**. Đưa qua `sin`/`cos` rồi `atan2` sẽ ra đúng `2°`.

   ```
   Δ thô  = -179 - 179 = -358°
   sin(-358°) = +0.0349 ,  cos(-358°) = +0.9994
   atan2    → +2°   ✅
   ```

3. So sánh với **frame đã GIỮ gần nhất** (`frames[-1]`), không phải frame đọc trước đó. Nhờ vậy
   robot bò rất chậm vẫn được ghi lại một frame mỗi khi tích luỹ đủ `min_move` — **không bị mất
   hoàn toàn**.

### Chốt lại ([248-249](tools/mcap2s2.py#L248))

```python
if len(frames) >= 4:
    episodes.append(EpisodeData(instruction=instruction, frames=frames))
```

Episode dưới 4 frame bị bỏ vì loader cũng bỏ:
[`if actions_len < 4: continue`](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L866).
Ghi ra đĩa rồi để loader bỏ thì chỉ tổ tốn chỗ và gây hoang mang khi đếm số liệu.

---

# GIAI ĐOẠN C — SINH NHÃN (trái tim của pipeline)

## 6. `camera_pose_from_base` — vị trí robot → ma trận camera 4×4

📍 [tools/mcap2s2.py:262-273](tools/mcap2s2.py#L262)

```python
def camera_pose_from_base(x, y, yaw, height_m, pitch_deg) -> np.ndarray:
    p = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    z_cam = Rz @ np.array([math.cos(p), 0.0, -math.sin(p)])   # trục quang
    x_cam = Rz @ np.array([0.0, -1.0, 0.0])                   # "phải" của ảnh
    y_cam = np.cross(z_cam, x_cam)                            # "xuống" của ảnh
    T = np.eye(4, dtype=np.float32)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_cam, y_cam, z_cam
    T[:3, 3] = np.array([x, y, height_m])
    return T
```

> 🧠 **Ma trận 4×4 "pose" là gì?** Là cách gói gọn *"vật này đang ở đâu và xoay thế nào"* vào một
> bảng số 4×4:
> ```
> ┌                    ┐
> │  R (3×3)   C (3×1) │   R = hướng xoay,  C = vị trí
> │  0  0  0      1    │
> └                    ┘
> ```
> Ở đây là ma trận **camera → world**: 3 cột đầu là 3 trục của camera *biểu diễn trong hệ thế giới*.

### Quy ước OpenCV (nhớ kỹ — sai là hỏng âm thầm)

| Cột | Tên | Hướng trong ảnh | Công thức |
|---|---|---|---|
| 0 | `x_cam` | sang **phải** ảnh | `Rz @ (0,-1,0)` — bên phải camera = **âm y** của robot |
| 1 | `y_cam` | **xuống** dưới ảnh | `z_cam × x_cam` (tích có hướng → tự vuông góc cả hai) |
| 2 | `z_cam` | **trục quang** (hướng camera nhìn) | `Rz @ (cos p, 0, −sin p)` |
| 3 | `C` | vị trí camera | `(x, y, height_m)` |

Giải thích `z_cam`: khi `pitch = 0` thì `(cos0, 0, −sin0) = (1,0,0)` — nhìn thẳng phía trước. Khi
`pitch = 30°` thì `(0.866, 0, −0.5)` — vẫn hướng trước nhưng **chúc xuống**. Dấu trừ vì trục z của
thế giới hướng **lên**, mà cúi là đi **xuống**.

`Rz` là ma trận xoay quanh trục thẳng đứng — nhân vào để "gắn" hướng camera theo hướng robot đang
quay.

⚠️ **Ba dấu hiệu bạn đã làm sai quy ước này** (và chúng đều KHÔNG báo lỗi):
- pixel goal luôn nằm ở nửa trên ảnh (đáng lẽ phải ở nửa dưới, vì đích nằm trên sàn);
- `goal` toàn `-1`;
- round-trip test ở [06 mục 4](06_pipeline_mcap_to_s2.md) cho ra quỹ đạo lệch trục (đáng lẽ tiến
  theo `+x` thì lại ra `+y`).

## 7. `project_to_pixel` — điểm 3D → toạ độ pixel

📍 [tools/mcap2s2.py:276-293](tools/mcap2s2.py#L276)

```python
R = cam_pose[:3, :3]
C = cam_pose[:3, 3]
p_cam = R.T @ (p_world - C)          # (1) đưa điểm về hệ camera
if p_cam[2] <= 1e-6:
    return -1, -1, False             # (2) điểm ở SAU lưng camera
u = K[0,0] * p_cam[0] / p_cam[2] + K[0,2]
v = K[1,1] * p_cam[1] / p_cam[2] + K[1,2]     # (3) phép chiếu phối cảnh
if not (0 <= u < w and 0 <= v < h):
    return -1, -1, False             # (4) rơi ra ngoài khung hình
return int(round(u)), int(round(v)), True
```

Đây là **phép chiếu camera lỗ kim** — nền tảng của toàn bộ thị giác máy tính.

**(1) Đổi hệ toạ độ.** `p_world - C` = vector từ camera tới điểm. Nhân với `Rᵀ` để **chuyển từ mô tả
theo hệ thế giới sang mô tả theo hệ camera**.

> 🧠 Vì sao **`R.T`** (chuyển vị) chứ không phải `R`? Vì `R` là ma trận *camera → world*, ta cần chiều
> ngược lại. Với ma trận xoay có một tính chất rất đẹp: **nghịch đảo = chuyển vị** (`R⁻¹ = Rᵀ`) —
> đảo chiều chỉ tốn một phép hoán vị chỉ số, không phải giải hệ phương trình.

**(2) Loại điểm sau lưng.** `p_cam[2]` là **độ sâu** (khoảng cách dọc trục quang). Âm nghĩa là điểm
nằm phía sau camera. Không chặn thì phép chia sẽ cho ra một toạ độ pixel *trông rất hợp lệ* nhưng
hoàn toàn sai — điểm ở sau lưng bị "lộn ngược" ra trước. Dùng `1e-6` thay vì `0` để tránh chia cho
số cực nhỏ.

**(3) Chia cho độ sâu.** Đây chính là lý do vật ở xa trông nhỏ hơn: cùng một `X`, `Z` càng lớn thì
`X/Z` càng nhỏ. `K` là **ma trận nội tại (intrinsics)** của camera:

```
K = ┌ fx   0  cx ┐     fx, fy = tiêu cự tính bằng pixel
    │  0  fy  cy │     cx, cy = tâm ảnh (thường ≈ W/2, H/2)
    └  0   0   1 ┘
```

**(4) Cắt biên.** Điểm chiếu ra ngoài `[0,W)×[0,H)` = camera **không thấy** → phải báo không thấy.

### 🔢 Ví dụ số (khớp với kết quả chạy thật ở [06 mục 3](06_pipeline_mcap_to_s2.md))

Camera cao `1.25 m`, cúi `30°`, `yaw=0`, `fx=fy=388.2`, `cx=320`, `cy=240`.
Đích là điểm trên sàn cách **2 m** phía trước: `p_world = (2, 0, 0)`, `C = (0, 0, 1.25)`.

```
d = p_world − C = (2, 0, −1.25)

X = x_cam · d = ( 0,   0, 0     )·d = 0
Y = y_cam · d = (−0.5, 0, −0.866)·d = −1 + 1.083 = 0.083
Z = z_cam · d = ( 0.866, 0, −0.5)·d = 1.732 + 0.625 = 2.357

u = 388.2 · 0/2.357 + 320 = 320       ← đúng giữa ảnh (đích thẳng phía trước ✓)
v = 388.2 · 0.083/2.357 + 240 = 254   ← hơi dưới tâm ảnh (đích nằm trên sàn ✓)
```

Thử lại với đích **gần hơn, cách 1 m**: `Z = 1.491`, `Y = 0.583` → `v = 392`. Điểm **trôi xuống
dưới**. Đúng trực giác: càng đến gần, đích càng tụt xuống đáy khung hình — và đến một lúc thì **rơi
khỏi khung** → `visible = False` → nhãn `-1`. Chính là ba số `-1` liên tiếp trước mỗi cú rẽ trong
kết quả thật.

## 8. `discretize_actions` — quỹ đạo liên tục → "nút bấm"

📍 [tools/mcap2s2.py:296-311](tools/mcap2s2.py#L296)

```python
actions = [-1]
for i in range(1, len(frames)):
    a, b = frames[i-1], frames[i]
    dyaw = math.atan2(math.sin(b.yaw - a.yaw), math.cos(b.yaw - a.yaw))
    if abs(math.degrees(dyaw)) >= min_turn_deg:
        actions.append(2 if dyaw > 0 else 3)   # yaw tăng = quay TRÁI
    else:
        actions.append(1)                       # còn lại = tiến
```

Robot đi trên một quỹ đạo **liên tục**, nhưng model học **tập lệnh rời rạc** (như bấm 4 nút).
Hàm này làm việc "số hoá" đó.

| Giá trị | Ý nghĩa | Hàm này có sinh ra không? |
|---|---|---|
| `-1` | mốc khởi đầu (frame 0 chưa làm gì) | ✅ luôn ở vị trí 0 |
| `1` | ↑ tiến | ✅ |
| `2` | ← rẽ trái | ✅ khi `dyaw > 0` |
| `3` | → rẽ phải | ✅ khi `dyaw < 0` |
| `0` | STOP | ❌ — **loader tự thêm** |
| `5` | ↓ cúi đầu | ❌ không dùng |

### ⚠️ Quy ước lệch pha — chỗ dễ sai nhất

> **`action[i]` = việc đã làm để đi từ frame `i-1` TỚI frame `i`.** Tức là action mô tả *quá khứ*,
> không phải *tương lai*.

Vì loader dịch trái một nhịp:
[`actions = item['actions'][1:] + [0]`](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L864)

```
ta ghi :  [-1,  1,  1,  3,  3,  1]
loader :  [ 1,  1,  3,  3,  1,  0]     ← bỏ -1 ở đầu, thêm STOP ở cuối
           ↑ frame 0 giờ mang action "việc sắp làm"    ↑ frame cuối = STOP ✓
```

Sau khi dịch, `action[i]` mới thành *"đứng ở frame i thì nên làm gì tiếp"* — đúng thứ model cần học.
Nếu bạn ghi sẵn theo nghĩa "tương lai", sau cú dịch của loader mọi nhãn sẽ **lệch một frame**.

`min_turn_deg` mặc định 5° ([dòng 638](tools/mcap2s2.py#L638)). Xoay dưới ngưỡng bị coi là "tiến" —
hợp lý vì robot đi thẳng vẫn luôn lắc nhẹ vài độ do nhiễu odometry.

## 9. `find_subgoal_frames` — chọn "đích trung gian"

📍 [tools/mcap2s2.py:314-331](tools/mcap2s2.py#L314)

```python
subgoals = []
for i in range(1, len(actions)):
    if actions[i] in (2, 3) and actions[i-1] == 1:   # đang tiến → bắt đầu xoay
        subgoals.append(i)
last = len(actions) - 1
if not subgoals or subgoals[-1] != last:
    subgoals.append(last)                            # luôn có đích cuối
```

**Luật:** mỗi khi robot **kết thúc một đoạn đi thẳng** (chuyển từ `1` sang `2`/`3`) thì frame đó là
một sub-goal. Cộng thêm frame cuối cùng của episode.

Ý nghĩa trực quan: *"đi thẳng tới chỗ rẽ"* — chính là thứ System 2 phải học chỉ ra trên ảnh.

```
action:  -1  1  1  1  1  1  3  3  3  1  1  1  1
index:    0  1  2  3  4  5  6  7  8  9 10 11 12
                            ↑                  ↑
                       sub-goal 6         sub-goal 12 (cuối episode)
```

> 📌 **Đây là chỗ "xấp xỉ" của script — và là chỗ bạn nên thay đầu tiên.** Data gốc R2R dùng các
> **viewpoint của đồ thị điều hướng** trong nhà mô phỏng làm sub-goal. Ta không có đồ thị đó nên lấy
> điểm rẽ làm xấp xỉ. Nếu robot của bạn có nguồn tốt hơn — waypoint của Nav2, điểm người vận hành
> bấm đánh dấu, node của bản đồ topo — thì **thay luật ở hàm này**, phần còn lại của pipeline giữ
> nguyên.

Điều kiện `if not subgoals or subgoals[-1] != last` tránh **thêm trùng** frame cuối khi episode
kết thúc đúng lúc đang rẽ.

## 10. `make_labels` — ráp 4 cột nhãn

📍 [tools/mcap2s2.py:334-358](tools/mcap2s2.py#L334)

Đây là hàm **điều phối** của giai đoạn C — gọi 4 hàm trên và ráp kết quả.

```python
actions  = discretize_actions(ep.frames, cfg.min_turn_deg)
subgoals = find_subgoal_frames(actions)

poses   = np.zeros((n, 4, 4), dtype=np.float32)
goals   = np.full((n, 2), -1, dtype=np.int32)     # mặc định = "không thấy"
rel_ids = np.full((n,),   -1, dtype=np.int32)

for t, fr in enumerate(ep.frames):
    poses[t] = camera_pose_from_base(fr.x, fr.y, fr.yaw, cfg.height_cm/100.0, cfg.pitch2)
    g = next((s for s in subgoals if s > t), None)     # sub-goal gần nhất Ở TƯƠNG LAI
    if g is None:
        continue
    p_world = np.array([ep.frames[g].x, ep.frames[g].y, 0.0], dtype=np.float32)
    u, v, visible = project_to_pixel(poses[t], K, p_world, cfg.width, cfg.height)
    if visible:
        goals[t]   = (u, v)
        rel_ids[t] = g - t
```

Bốn điểm cần nhớ:

**(a) `cfg.pitch2`, KHÔNG phải `pitch1`** ([dòng 346](tools/mcap2s2.py#L346)). Chuỗi `setting` mô tả
camera **nhìn cúi**, nên `pose.{setting}` phải là pose của camera đó. Đây là điều dễ sai số 1 ghi ở
đầu file script.

**(b) Khởi tạo `-1` là "mặc định an toàn".** Frame nào không tìm được đích nhìn thấy sẽ **giữ
nguyên** `-1` — không cần nhánh `else`. Loader hiểu `-1` = "mẫu turn"
([dòng 876](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L876)).

**(c) `next((s for s in subgoals if s > t), None)`** — lấy sub-goal **gần nhất nằm ở tương lai**.
Dấu `>` (chứ không `>=`) đảm bảo không lấy chính frame hiện tại làm đích. Frame cuối cùng luôn ra
`None` → nhãn `-1` → về sau loader biến nó thành mẫu STOP.

**(d) `z = 0`** ([dòng 352](tools/mcap2s2.py#L352)) — waypoint là **vị trí chân robot trên sàn**, không
phải vị trí camera. Chiếu điểm trên sàn thì pixel goal mới rơi vào **mặt sàn trong ảnh** — đúng thứ
model học chỉ. Nếu vô ý dùng `z = height_m`, điểm sẽ luôn nằm ở đường chân trời và thường rơi ra
ngoài khung.

**Giá trị `rel_id = g - t` và số phận của nó trong loader:**

| `rel_id` | Nghĩa | Loader làm gì ([876-905](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L876)) |
|---|---|---|
| `-1` | không thấy đích | mẫu **turn** (hoặc bỏ nếu đang đi thẳng) |
| `≥ 3` | đích hợp lệ, còn `k` frame nữa mới tới | mẫu **pixel_goal** ← **loại quan trọng nhất** |
| `1, 2` | đích quá gần | **bị bỏ** (`if goal_len < 3: continue`) |

---

# GIAI ĐOẠN D — GHI ẢNH

## 11. `write_images` — ghi 3 luồng ảnh

📍 [tools/mcap2s2.py:364-384](tools/mcap2s2.py#L364)

```python
chunk = f"chunk-{ep_idx // 1000:03d}"
base  = os.path.join(scene_dir, "videos", chunk)
d_front = os.path.join(base, f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p1_tag}deg")
d_down  = os.path.join(base, f"observation.images.rgb.{cfg.h_tag}cm_{cfg.p2_tag}deg")
d_depth = os.path.join(base, f"observation.images.depth.{cfg.h_tag}cm_{cfg.p2_tag}deg")
...
for i, fr in enumerate(ep.frames):
    stem = f"episode_{ep_idx:06d}_{i}"
```

Hàm này **không có logic gì thông minh** — toàn bộ giá trị của nó nằm ở việc **đặt đúng tên**. Loader
ghép đường dẫn bằng chuỗi thuần
([dòng 1014-1022](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L1015)):

```python
image_file = os.path.join(video, f"observation.images.rgb.{height}cm_{pitch_1}deg",
                          f"episode_{ep_id:06d}_{id}.jpg")
lookdown   = image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')
depth      = lookdown.replace('rgb', 'depth').replace('.jpg', '.png')
```

Ba hệ quả rút ra từ đoạn code trên:

1. Sai **một ký tự** trong tên thư mục → `FileNotFoundError` lúc `__getitem__` (giữa lúc train).
2. Loader **suy ra** đường dẫn ảnh cúi và depth bằng `.replace()` từ đường dẫn ảnh thẳng → cả ba thư
   mục phải nằm **cùng cấp**, tên chỉ khác đúng những chỗ được thay.
3. ⚠️ **Không được để chữ `rgb` xuất hiện trong thư mục cha** — `.replace('rgb','depth')` thay
   **mọi** lần xuất hiện. Đặt `--dataset-name rgb_robot` là đủ để hỏng đường dẫn depth.

`ep_idx // 1000` — LeRobot gom mỗi 1000 episode vào một `chunk` để tránh thư mục có hàng vạn file
(hệ thống file chậm hẳn khi vậy). `:03d` = đệm 0 cho đủ 3 chữ số → `chunk-000`.

`:06d` cho `episode_000000` — đệm 6 chữ số để sắp xếp theo tên trùng với sắp xếp theo số.

## 12. `_save_jpeg` — ghi ảnh màu

📍 [tools/mcap2s2.py:387-393](tools/mcap2s2.py#L387)

```python
if payload[:2] == b"\xff\xd8":            # 2 byte đầu của MỌI file JPEG
    with open(path, "wb") as f:
        f.write(payload)                   # ghi thẳng byte gốc
else:
    Image.open(io.BytesIO(payload)).convert("RGB").save(path, format="JPEG", quality=90)
```

`\xff\xd8` là **magic number** của JPEG (mọi file JPEG đều bắt đầu bằng 2 byte này). Nếu message đã
là JPEG rồi thì **chép thẳng byte** — vừa nhanh hơn, vừa **không mất chất lượng**.

> 🧠 **Vì sao "giải nén rồi nén lại" là xấu?** JPEG là nén **có mất mát**. Mỗi vòng
> giải-nén-rồi-nén-lại làm ảnh nhoè thêm một chút (gọi là *generation loss*). Với dữ liệu train,
> nhoè thừa = tín hiệu kém đi mà chẳng đổi lại được gì.

`io.BytesIO(payload)` = "giả vờ" mảng byte trong RAM là một file, để `Image.open` (vốn nhận file) đọc
được — tránh phải ghi ra đĩa tạm.

## 13. `_save_depth_png` — ghi ảnh độ sâu + có chốt chặn

📍 [tools/mcap2s2.py:396-405](tools/mcap2s2.py#L396)

```python
arr = np.array(Image.open(io.BytesIO(payload)))
if arr.dtype != np.uint16:
    raise SystemExit("❌ Depth không phải uint16 ...")
Image.fromarray(arr).save(path, format="PNG", optimize=True)
```

**Kiểm tra `uint16` là toàn bộ lý do hàm này tồn tại.** Loader chia độ sâu cho 1000 để ra mét
([dòng 1024-1026](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L1024),
`depth_scale=1000`).

| Nếu depth là… | Sau khi chia 1000 | Hậu quả |
|---|---|---|
| `uint16` milimét (đúng) | `2500 → 2.5 m` | ✅ |
| `uint8` (0-255) | `200 → 0.2 m` | mọi thứ "sát mặt", vô nghĩa |
| `float32` mét | `2.5 → 0.0025 m` | lệch **1000 lần** |

Cả ba trường hợp sai đều **chạy trót lọt, không báo lỗi** — nên script chọn **dừng hẳn** với thông
báo rõ ràng thay vì ghi ra dữ liệu hỏng. Đây là triết lý "fail fast, fail loud" — với pipeline dữ
liệu thì nó đáng giá gấp bội.

Dùng **PNG** chứ không JPEG vì PNG **nén không mất mát** và hỗ trợ 16-bit. JPEG chỉ 8-bit và làm
nhoè — cả hai đều phá số đo độ sâu.

⚠️ Với ROS thật: ảnh depth thường ở dạng `sensor_msgs/Image` **thô** (`encoding='16UC1'`), không phải
PNG. Khi đó `Image.open()` sẽ vỡ → phải dựng numpy từ `msg.data` + `msg.height/width` rồi mới lưu.

---

# GIAI ĐOẠN E — GHI PARQUET + META

## 14. `write_parquet` — ghi bảng số

📍 [tools/mcap2s2.py:411-441](tools/mcap2s2.py#L411)

> 🧠 **Parquet là gì?** Định dạng bảng **lưu theo cột** (thay vì theo hàng như CSV). Ưu điểm: nén tốt
> hơn nhiều và đọc được **chỉ vài cột** mà không phải nạp cả file. Đây là chuẩn của LeRobot.

```python
table = pa.table({
    "action":                       pa.array(labels["action"], type=pa.int32()),
    f"pose.{s}":                    pa.array([p.tolist() for p in labels["pose"]],
                                             type=pa.list_(pa.list_(pa.float32()))),
    f"goal.{s}":                    pa.array([g.tolist() for g in labels["goal"]],
                                             type=pa.list_(pa.int32(), 2)),
    f"relative_goal_frame_id.{s}":  pa.array(labels["rel_id"], type=pa.int32()),
    "timestamp":     ...float32,
    "frame_index":   ...int64,   "episode_index": ...int64,
    "index":         ...int64,   "task_index":    ...int64,
})
```

### ⚠️ Vì sao dtype phải chính xác đến từng kiểu

Loader gọi `.tolist()` **trên từng ô** của bảng
([dòng 786-789](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L786)):

```python
ep_poses = df[pose_key].apply(lambda x: x.tolist()).tolist()
```

`.tolist()` là phương thức của **numpy array**. Nếu ô đó ra kiểu Python thuần (`list`, `int`) thì
**không có** phương thức này → `AttributeError`. Và pandas quyết định trả về numpy array hay list
thuần **dựa trên dtype của cột parquet**. Nói cách khác: **ghi sai dtype = loader vỡ**.

Tệ hơn: nhiều đường trong loader bọc `try/except` → lỗi bị nuốt, scene bị bỏ **im lặng**, bạn chỉ
thấy `len(dataset) == 0` mà không hiểu vì sao. Đây là lý do có giai đoạn F.

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `action` | `int32` | |
| `pose.{s}` | `list<list<float32>>` | list lồng list = ma trận 4×4 |
| `goal.{s}` | `fixed_size_list<int32>[2]` | **cố định** 2 phần tử |
| `relative_goal_frame_id.{s}` | `int32` | |
| `timestamp` | `float32` | |
| `frame_index`/`episode_index`/`index`/`task_index` | `int64` | |

### Các cột "sổ sách" của LeRobot

| Cột | Công thức | Nghĩa |
|---|---|---|
| `frame_index` | `0..n-1` | thứ tự frame **trong episode** |
| `index` | `index_offset .. +n` | thứ tự frame **trong toàn dataset** (cộng dồn qua các episode) |
| `episode_index` | `ep_idx` lặp lại `n` lần | episode nào |
| `task_index` | `task_index` lặp lại | câu lệnh nào |
| `timestamp` | `arange(n) / fps` | thời điểm giả định |

📌 **`timestamp` là con số "giả".** Nó tính từ chỉ số chia cho fps, nên sau khi giai đoạn B lọc bỏ
frame đứng yên thì nó **không còn khớp thời gian thật**. Không sao — loader S2 **không dùng** cột
này; nó có mặt chỉ để đủ chuẩn LeRobot.

## 15. `write_meta` — ghi 4 file mô tả

📍 [tools/mcap2s2.py:444-516](tools/mcap2s2.py#L444)

| File | Nội dung | Loader S2 có đọc? |
|---|---|---|
| `episodes.jsonl` | mỗi dòng: `episode_index`, `tasks` (câu lệnh), `length` | ✅ **file DUY NHẤT loader đọc** ([dòng 765](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L765)) |
| `tasks.jsonl` | danh sách câu lệnh | ❌ |
| `episodes_stats.jsonl` | min/max/mean/std của từng cột | ❌ |
| `info.json` | phiên bản, tổng số episode/frame, mẫu đường dẫn, mô tả feature | ❌ |

> 🧠 **`.jsonl` khác `.json` thế nào?** JSONL = "JSON Lines": **mỗi dòng là một object JSON độc lập**.
> Ưu điểm: đọc/ghi từng dòng được, nối file bằng `cat` được, không phải nạp cả file vào RAM.

Ba file kia vẫn được ghi để dataset **đúng chuẩn LeRobot v2.1** — công cụ khác (visualizer, script
thống kê, phiên bản loader sau này) có thể cần.

### Hàm lồng `st()` — [dòng 471-479](tools/mcap2s2.py#L471)

```python
def st(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).reshape(n, -1)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [n]}
```

Một **closure** (hàm lồng trong hàm, dùng được biến `n` của hàm cha) tính thống kê cho một cột.

`reshape(n, -1)` là mẹo hay: `-1` nghĩa là *"tự tính chiều còn lại"*. Nhờ vậy cùng một hàm xử lý được
cả cột 1 chiều (`action` → `(n,1)`) lẫn cột 2 chiều (`goal` → `(n,2)`). Rồi `.min(0)` = lấy min
**theo từng cột**.

⚠️ Lưu ý nhỏ: thống kê tính trên **cả giá trị `-1`** (nghĩa là "không có đích"), nên `mean` của
`goal` không mang ý nghĩa hình học. Không ảnh hưởng gì vì loader không đọc file này.

### `info.json` — vài trường đáng chú ý ([496-514](tools/mcap2s2.py#L496))

```python
"total_chunks": (len(episodes) - 1) // 1000 + 1,   # số chunk cần thiết
"splits": {"train": f"0:{len(episodes)}"},          # toàn bộ dùng để train
"data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
"video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}_{frame}.jpg",
```

Hai trường `*_path` là **khuôn mẫu chuỗi** — chuẩn LeRobot dùng chúng để tự ghép đường dẫn.
(Loader S2 tự ghép tay nên không đọc, nhưng ghi đúng vẫn tốt.)

📌 Ở bản hiện tại, `tasks.jsonl` ghi **một task cho mỗi episode** (`task_index = i`), kể cả khi hai
episode trùng câu lệnh. Chuẩn LeRobot muốn task là *danh sách câu lệnh duy nhất*. Vô hại với S2
(không ai đọc file này), nhưng nếu bạn xuất dataset cho công cụ LeRobot khác thì nên khử trùng lặp.

---

# GIAI ĐOẠN F — TỰ KIỂM ĐỊNH

## 16. `self_check` — đóng vai loader để tự chấm điểm

📍 [tools/mcap2s2.py:522-587](tools/mcap2s2.py#L522)

> **Vì sao cần hàm này?** Vì mọi lỗi nguy hiểm ở pipeline này đều **không ném exception**. Sai tên
> cột → scene bị bỏ im lặng. Sai dtype → `try/except` nuốt lỗi. Camera không thấy sàn → `goal` toàn
> `-1`. Cả ba đều cho ra "dataset chạy được nhưng rỗng/vô dụng". Hàm này **chép lại logic cắt mẫu
> của loader** để trả lời câu hỏi duy nhất đáng quan tâm: *"dataset này ra được bao nhiêu mẫu
> train?"*

### Bản đối chiếu với loader thật

| `self_check` | Loader ([857-947](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L864)) |
|---|---|
| `actions = df["action"].tolist()[1:] + [0]` | `actions = item['actions'][1:] + [0]` |
| `if n < 4: continue` | `if actions_len < 4: continue` |
| `for k in range(n // sample_step + 1)` | `for n in range(num_rounds + 1)` |
| `if start in (n, n-1): continue` | `if n*step == actions_len or == actions_len-1: continue` |
| `rel[start] == -1` → mẫu turn (bỏ nếu `action == 1`) | `if pixel_goal[0] == -1: if action_flag == 1: continue` |
| `rel[start] < 3` → bỏ | `if goal_len < 3: continue` |
| còn lại → mẫu pixel_goal | `pixel_goal_list.append(...)` |
| `n_stop += 1` mỗi episode | `stop_list.append(...)` mỗi episode |
| `n_goal + n_turn + n_stop*5` | `list_data_dict += turn_list; += stop_list * 5` ([938-941](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L938)) |

**Vì sao `stop × 5`?** Mỗi episode chỉ có **đúng một** khoảnh khắc đáng dừng, trong khi có hàng chục
mẫu pixel_goal. Không nhân lên thì model gần như không bao giờ thấy ví dụ STOP và sẽ **không bao giờ
học dừng** — robot đi qua đích rồi đi mãi. Nhân 5 là cách thô sơ để cân bằng lớp.

**Vì sao bỏ `start ∈ {n, n-1}`?** Frame cuối và áp cuối không đủ chỗ để cắt một cửa sổ tương lai.

### Việc thứ hai: kiểm tra file ảnh có thật không ([561-572](tools/mcap2s2.py#L561))

```python
for fid in range(0, start + rel[start] + 1):
    for sub, ext in (...3 thư mục...):
        if not os.path.exists(p):
            missing += 1
```

Loader nạp **cả cửa sổ** `[0, end_frame_id)` chứ không chỉ frame `start` (nó cần lịch sử ảnh —
`num_history`). Thiếu **một** file thôi là crash giữa lúc train, có khi sau vài giờ. Kiểm trước ở
đây thì biết ngay trong 2 giây.

### Đọc kết quả ([575-587](tools/mcap2s2.py#L575))

```
mẫu pixel_goal : 14    ← loại quan trọng nhất
mẫu turn       : 3
mẫu stop       : 2   (loader nhân 5 khi pixel_goal_only=False)
bị bỏ (k < 3)  : 0
file ảnh thiếu : 0
→ train S2  (pixel_goal_only=False): 27 mẫu       = 14 + 3 + 2×5
→ train dual (pixel_goal_only=True): 14 mẫu       = chỉ pixel_goal
```

Hai cảnh báo cuối hàm là hai lỗi hay gặp nhất:
- `n_goal == 0` → camera **không thấy sàn** (góc cúi quá nhỏ) hoặc sai quy ước pose;
- `missing > 0` → sai tên thư mục ảnh.

📌 Con số này đã được **đối chiếu với loader thật** và trùng khớp ([06 mục 4](06_pipeline_mcap_to_s2.md)).
Nhưng `self_check` là *bản chép*, nên nếu loader trong repo đổi logic, hàm này sẽ nói dối. **Bước
kiểm định bằng chính loader thật vẫn là bước cuối cùng bắt buộc.**

⚠️ Một khác biệt nhỏ: loader nhân số mẫu theo **số câu lệnh** của mỗi episode (tách bằng
`<INSTRUCTION_SEP>`), còn `self_check` giả định mỗi episode 1 câu lệnh. Nếu bạn dùng nhiều câu lệnh
cho một lượt đi, con số thật sẽ **cao hơn** báo cáo của F.

---

# ĐIỀU PHỐI

## 17. `Config` + 4 property — nơi sinh ra chuỗi `setting`

📍 [tools/mcap2s2.py:593-617](tools/mcap2s2.py#L593)

```python
@property
def h_tag(self)  -> int: return int(round(self.height_cm))
@property
def p1_tag(self) -> int: return int(round(self.pitch1))
@property
def p2_tag(self) -> int: return int(round(self.pitch2))
@property
def setting(self) -> str: return f"{self.h_tag}cm_{self.p2_tag}deg"
```

> 🧠 **`@property` là gì?** Cho phép viết `cfg.setting` (như đọc một thuộc tính) nhưng thực chất
> Python **chạy một hàm** để tính. Lợi ích: giá trị **luôn nhất quán** với các trường gốc, không thể
> quên cập nhật.

**`setting` là chuỗi quan trọng nhất trong toàn bộ script.** Nó xuất hiện ở:
- tên cột parquet: `pose.125cm_30deg`, `goal.125cm_30deg`, `relative_goal_frame_id.125cm_30deg`
- tên thư mục ảnh cúi và depth
- `info.json`

⚠️ **`setting` dùng `p2_tag` (góc CÚI), không phải `p1_tag`.** Sai chỗ này → loader không tìm thấy
cột → in `Warning: Missing data for setting ...` rồi **bỏ nguyên scene** → dataset rỗng mà không có
lỗi nào rõ ràng.

`int(round(...))` để `125.0` ra `"125"` chứ không phải `"125.0"` — tên thư mục phải khớp **từng ký
tự** với chuỗi loader ghép ra.

## 18. `main` — nhạc trưởng

📍 [tools/mcap2s2.py:620-715](tools/mcap2s2.py#L620)

### 18.1. Khai báo tham số dòng lệnh ([621-644](tools/mcap2s2.py#L621))

| Nhóm | Cờ | Mặc định |
|---|---|---|
| Vào/ra | `--mcap` (bắt buộc), `--out`, `--dataset-name`, `--scene-id` | `./traj_data`, `myrobot`, `scene_0000` |
| Tên topic | `--topic-rgb-front/-down/-depth/-pose/-caminfo/-episode` | theo file demo |
| Thông số camera | `--height-cm`, `--pitch1`, `--pitch2`, `--fps` | `None` → **lấy từ metadata mcap** |
| Ngưỡng | `--tol-ms 60`, `--min-move 0.05`, `--min-turn-deg 5` | |
| Bổ sung | `--instruction-file` | `None` |

Mẹo thiết kế: mấy cờ thông số camera mặc định `None` chứ không phải một con số. Nhờ vậy `main` phân
biệt được *"người dùng không truyền"* (→ lấy từ metadata) với *"người dùng truyền đúng bằng giá trị
mặc định"*.

### 18.2. Đọc + hai chốt chặn sớm ([654-658](tools/mcap2s2.py#L654))

```python
if not raw.rgb_front:
    raise SystemExit("❌ Không có message nào ở topic ... — không có ảnh thì không train được.")
if not raw.caminfo:
    raise SystemExit("❌ Thiếu camera_info (ma trận K). Không có K thì KHÔNG chiếu được pixel-goal.")
```

**Dừng sớm, thông báo bằng tiếng người.** Không có `K` thì `project_to_pixel` không chạy được → mọi
`goal` sẽ là `-1` → dataset vô dụng. Phát hiện ở giây thứ 2 tốt hơn phát hiện ở phút thứ 10.

### 18.3. Dựng `Config` ([660-672](tools/mcap2s2.py#L660))

```python
cfg = Config(
    height_cm = args.height_cm if args.height_cm is not None else float(md.get("height_cm", 125)),
    ...
    width  = int(raw.caminfo[0][1]["width"]),
    height = int(raw.caminfo[0][1]["height"]),
    ...)
K = np.array(raw.caminfo[0][1]["K"], dtype=np.float64).reshape(3, 3)
```

Thứ tự ưu tiên **cờ dòng lệnh → metadata mcap → hằng số cứng**. Kích thước ảnh và `K` thì luôn lấy
từ `camera_info` (bản tin đầu tiên) — đây là thông tin do chính camera khai báo, không nên đoán.

⚠️ `K` và `width/height` lấy từ **`--topic-caminfo`**, mặc định là camera **lookdown**. Đúng, vì
pixel goal được chiếu vào **ảnh cúi**. Nếu bạn trỏ cờ này sang camera nhìn thẳng mà hai camera khác
độ phân giải/tiêu cự thì `goal` sẽ sai — sai âm thầm.

### 18.4. Chốt chặn câu lệnh ([678-687](tools/mcap2s2.py#L678))

```python
if args.instruction_file:
    overrides = json.load(open(args.instruction_file, encoding="utf-8"))
    for i, ep in enumerate(episodes):
        ep.instruction = overrides.get(str(i), ep.instruction)
for i, ep in enumerate(episodes):
    if not ep.instruction.strip():
        raise SystemExit("❌ Episode i không có câu lệnh. System 2 học từ NGÔN NGỮ ...")
```

Đây là chốt chặn **mang tính ML** chứ không phải kỹ thuật: System 2 là mô hình **thị giác-ngôn ngữ**.
Một mẫu không có câu lệnh thì model chẳng biết phải làm gì với tấm ảnh — mẫu rác. Chặn ngay còn hơn
để nó lẻn vào tập train.

`overrides.get(str(i), ...)` dùng `str(i)` vì **khoá JSON luôn là chuỗi** (`{"0": "...", "1": "..."}`).

### 18.5. Vòng lặp chính ([692-703](tools/mcap2s2.py#L692))

```python
index_offset = 0
for i, ep in enumerate(episodes):
    labels = make_labels(ep, K, cfg)                      # C
    write_images(i, ep, scene_dir, cfg)                   # D
    write_parquet(i, labels, scene_dir, cfg, task_index=i,
                  index_offset=index_offset, fps=fps)     # E
    index_offset += len(ep.frames)
    ...
write_meta(scene_dir, episodes, all_labels, cfg, fps)
```

`index_offset` cộng dồn để cột `index` **liên tục xuyên suốt dataset** — đúng quy ước LeRobot (mỗi
frame trên toàn bộ dataset có một số thứ tự duy nhất).

### 18.6. In hai bước tiếp theo ([707-715](tools/mcap2s2.py#L707))

Script tự in ra **đúng hai dòng bạn cần copy** để đăng ký dataset vào
[`data_dict`](../../../code/internnav/dataset/internvla_n1_lerobot_dataset.py#L127) và trỏ
`--vln_dataset_use`. Chi tiết: [06 mục 5](06_pipeline_mcap_to_s2.md).

## 19. `_hist` — đếm số lần mỗi action

📍 [tools/mcap2s2.py:718-720](tools/mcap2s2.py#L718)

```python
def _hist(labels: dict) -> str:
    vals, counts = np.unique(labels["action"], return_counts=True)
    return " ".join(f"{IDX2NAME.get(int(v), v)}×{c}" for v, c in zip(vals, counts))
```

Chỉ để **in cho dễ nhìn**: `np.unique(..., return_counts=True)` trả về hai mảng — các giá trị khác
nhau và số lần xuất hiện. Ghép với bảng tên `IDX2NAME` ([dòng 78](tools/mcap2s2.py#L78)) ra chuỗi
kiểu:

```
ep 0:  40 frame ·  25 frame có pixel-goal · action=start×1 ↑ tiến×31 ← trái×3 → phải×5
```

Nhìn dòng này là biết ngay tỉ lệ tiến/rẽ có hợp lý không. Toàn `↑ tiến` mà không có rẽ nào → nghi
ngờ `min_turn_deg` đặt quá cao hoặc dữ liệu pose bị phẳng.

---

## 20. Tra cứu nhanh: muốn đổi X thì sửa hàm nào?

| Muốn… | Sửa hàm | Dòng |
|---|---|---|
| Đọc log ROS 2 thật (mã hoá CDR) | `read_mcap` — thay `json.loads` | [110](tools/mcap2s2.py#L110) |
| Ảnh là `sensor_msgs/Image` thô | `_payload_bytes` + `_save_jpeg`/`_save_depth_png` | [174](tools/mcap2s2.py#L174), [387](tools/mcap2s2.py#L387) |
| Lấy pose từ `/tf` thay vì topic pose | `sync_frames` — thay chỗ đọc `m_pose[1]["pose"]` | [224](tools/mcap2s2.py#L224) |
| Siết/nới đồng bộ thời gian | cờ `--tol-ms` (không cần sửa code) | [636](tools/mcap2s2.py#L636) |
| Thay luật chọn sub-goal (waypoint Nav2, điểm người bấm…) | **`find_subgoal_frames`** ← nơi đáng sửa nhất | [314](tools/mcap2s2.py#L314) |
| Đổi ngưỡng phân biệt tiến/rẽ | cờ `--min-turn-deg` | [638](tools/mcap2s2.py#L638) |
| Robot có 3 bậc tự do (bay/dốc), cần cả pitch/roll | `_quat_to_yaw` + `camera_pose_from_base` | [168](tools/mcap2s2.py#L168), [262](tools/mcap2s2.py#L262) |
| Waypoint không nằm trên sàn (`z ≠ 0`) | `make_labels` dòng `p_world` | [352](tools/mcap2s2.py#L352) |
| Thêm cột nhãn mới vào parquet | `write_parquet` + `write_meta` (`features`) | [425](tools/mcap2s2.py#L425), [508](tools/mcap2s2.py#L508) |
| Nhiều câu lệnh cho một lượt đi | `write_meta` — nối bằng `<INSTRUCTION_SEP>` | [458](tools/mcap2s2.py#L458) |
| Loader đổi logic cắt mẫu | `self_check` — chép lại logic mới | [522](tools/mcap2s2.py#L522) |

---

## 21. Bảy chỗ dễ sai nhất (tổng hợp lại)

| # | Sai gì | Hậu quả | Hàm liên quan | Phát hiện bằng |
|---|---|---|---|---|
| 1 | `setting` dùng `pitch1` thay `pitch2` | scene bị bỏ **im lặng** → dataset rỗng | `Config.setting` | giai đoạn F báo thiếu cột |
| 2 | Depth không phải uint16 milimét | độ sâu lệch 1000 lần | `_save_depth_png` | script **dừng ngay** |
| 3 | Sai quy ước hệ toạ độ camera | nhãn quỹ đạo dual sai, **không báo lỗi** | `camera_pose_from_base` | round-trip test ([06 mục 4](06_pipeline_mcap_to_s2.md)) |
| 4 | Sai dtype parquet | `AttributeError` bị nuốt → dataset rỗng | `write_parquet` | kiểm bằng loader thật |
| 5 | Lệch pha quy ước `action` | nhãn lệch 1 frame | `discretize_actions` | so `action` với `rel_id` bằng mắt |
| 6 | Chữ `rgb` trong thư mục cha | đường dẫn depth hỏng | `write_images` | F báo "file ảnh thiếu" |
| 7 | Camera nhìn thẳng, không thấy sàn | `goal` toàn `-1` → 0 mẫu pixel_goal | `project_to_pixel` | F cảnh báo `⚠️ KHÔNG có mẫu pixel_goal nào!` |

---

*Xem tiếp: [06_pipeline_mcap_to_s2](06_pipeline_mcap_to_s2.md) (pipeline tổng thể + cách kiểm định bằng
loader thật) · [04_data_train_s2](04_data_train_s2.md) (hợp đồng dữ liệu) ·
[07_phu_luc_lerobot_format](07_phu_luc_lerobot_format.md) (giải nghĩa từng file vật lý).*

*Quay lại mục lục: [00_README](00_README.md).*
