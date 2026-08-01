# 10 — Phụ lục: `vln_ce/raw_data` là gì? (khác `traj_data` chỗ nào)

> **File này trả lời:** trong repo HF [`InternRobotics/InternData-N1`](https://huggingface.co/datasets/InternRobotics/InternData-N1),
> thư mục `vln_ce/raw_data/` có cấu trúc gì, bên trong file `.json.gz` chứa gì, ai đọc nó trong code
> InternNav, và **khi nào bạn cần / không cần** nó.
>
> **Mức độ tin cậy — đọc kỹ dòng này:**
> - ✅ **Đo thật**: cây thư mục + kích thước từng file (lấy từ HF API ngày **31/07/2026**), các trích
>   dẫn `file:line` trong `InternNav/code/`, nội dung `meta/*.jsonl` của scene có sẵn trên máy.
> - ⚠️ **Chưa mở được file `.json.gz`**: repo `InternData-N1` đang ở chế độ **gated** (tải về đòi
>   đăng nhập + được duyệt). Nên phần mô tả *các trường bên trong JSON* (mục 4–6) lấy từ **đặc tả
>   chính thức của VLN-CE** (<https://jacobkrantz.github.io/vlnce/data>) — định dạng mà InternNav
>   dùng lại nguyên vẹn (bằng chứng: code nạp bằng `type: R2RVLN-v1` của Habitat, xem mục 7).
>
> Bộ tài liệu: [04_data_train_s2](04_data_train_s2.md) · [07_phu_luc_lerobot_format](07_phu_luc_lerobot_format.md)

---

## 1. Trả lời trong 60 giây

Bộ `vln_ce` có **hai** thư mục con, và chúng là **hai giai đoạn của cùng một dữ liệu**:

```
vln_ce/
├── raw_data/    ← "ĐỀ BÀI"  : vài file JSON nén. Mỗi dòng = 1 nhiệm vụ:
│                             câu lệnh tiếng Anh + điểm xuất phát + điểm đích + đường đi mẫu.
│                             KHÔNG có một tấm ảnh nào.  (tổng ~370 MB, gần hết là RxR)
└── traj_data/   ← "BÀI GIẢI ĐÃ QUAY VIDEO": chạy các đề bài đó trong simulator Habitat,
                              chụp lại RGB + depth + hành động từng bước, đóng gói LeRobot.
                              (hàng trăm GB, chia theo từng scene)
```

| | `raw_data` | `traj_data` |
|---|---|---|
| Nội dung | **văn bản + toạ độ** | **ảnh + số + video** |
| Định dạng | `.json.gz` (JSON nén gzip) | LeRobot v2.1 (`parquet` + `.jpg`/`.png` + `.jsonl`) |
| Dung lượng | ~370 MB | rất lớn (mỗi scene vài trăm MB → 1.5 GB) |
| Chia theo | **split** (`train` / `val_seen` / `val_unseen`) | **scene** (`17DRP5sb8fy`, …) |
| Vai trò | định nghĩa **nhiệm vụ** & **thước đo** | dữ liệu **huấn luyện** thực tế |
| Ai dùng | **evaluator** (Habitat) chấm điểm model | **dataloader** lúc train System 2 |

📌 **Câu chốt cho người đang chuẩn bị data:** train System 2 **không đọc `raw_data`**. Loader
`NavPixelGoalDataset` chỉ đọc `traj_data` ([04](04_data_train_s2.md) mục 2). `raw_data` cần khi bạn
muốn **chạy đánh giá (eval) trong Habitat** hoặc muốn **tự render lại `traj_data` từ đầu**.

---

## 2. Cây thư mục thật (đo qua HF API, 31/07/2026)

```
vln_ce/raw_data/
├── r2r/
│   ├── train/
│   │   └── train.json.gz                    2,408,561 B  (~2.3 MB)
│   ├── val_seen/
│   │   ├── val_seen.json.gz                   225,341 B
│   │   ├── val_seen.json                    2,511,039 B   ← bản KHÔNG nén, cùng nội dung
│   │   └── val_seen/
│   │       └── val_seen.json.gz               225,341 B   ← 🐛 thư mục lồng TRÙNG (byte y hệt)
│   └── val_unseen/
│       ├── val_unseen.json.gz                 325,775 B
│       └── val_unseen1.json                 5,495,026 B   ← tên lạ, không nén (xem mục 9)
└── rxr/
    └── rxr.zip                            363,554,335 B  (~347 MB) ← 📦 phải GIẢI NÉN mới dùng được
```

**Ba nhận xét quan trọng:**

1. **R2R để "trần"**, RxR gói trong **một file zip duy nhất** → tải xong **bắt buộc `unzip`**, vì
   code trỏ thẳng vào đường dẫn bên trong zip (mục 7).
2. Thư mục `val_seen/val_seen/` là **lỗi đóng gói** (file con giống hệt file cha, 225.341 byte).
   Vô hại, cứ bỏ qua.
3. `raw_data` **không có** file `embeddings.json.gz` (bảng vector từ vựng). File đó chỉ nằm ở
   `vln_pe/raw_data/` — so sánh ở mục 10.

---

## 3. Vì sao lại tách "đề bài" khỏi "bài giải"?

Hình dung một **đề thi chạy việt dã**:

| Ví dụ đời thường | Trong dataset |
|---|---|
| Tờ đề: "xuất phát ở cổng A, chạy qua sân bóng, dừng ở nhà thi đấu" | 1 **episode** trong `raw_data` |
| Bản đồ trường | **scene 3D** Matterport3D (tải riêng, không nằm trong `InternData-N1`) |
| Video quay một vận động viên chạy đúng lộ trình | 1 **episode** trong `traj_data` |

Tách ra vì:

- **Đề bài rất nhẹ** (2 MB), **video rất nặng** (hàng trăm GB). Ai chỉ cần chấm điểm thì tải 2 MB là đủ.
- **Một đề bài có thể quay lại nhiều lần** với nhiều cấu hình camera khác nhau (`vln_ce` render 5
  setting camera: `125cm_0deg`, `125cm_30deg`, `125cm_45deg`, `60cm_15deg`, `60cm_30deg` —
  [07](07_phu_luc_lerobot_format.md) mục 2).
- **Chấm điểm phải công bằng**: mọi nhóm nghiên cứu dùng **chung một bộ đề** `val_unseen` thì con số
  SR/SPL mới so sánh được với nhau.

---

## 4. Bên trong một file `{split}.json.gz` có gì?

`.json.gz` = một file JSON **nén gzip**. Mở bằng Python:

```python
import gzip, json
with gzip.open('data/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz', 'rt') as f:
    d = json.load(f)
print(d.keys())            # dict_keys(['episodes', 'instruction_vocab'])
print(len(d['episodes']))  # số nhiệm vụ trong split này
```

Cấu trúc gồm **2 khoá cấp cao nhất**:

```json
{
  "episodes": [ … danh sách nhiệm vụ … ],
  "instruction_vocab": { … từ điển từ vựng … }
}
```

### 4.1. Một phần tử của `episodes` — bảng giải nghĩa từng trường

```json
{
  "episode_id": "1",
  "trajectory_id": 4332,
  "scene_id": "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
  "start_position": [ 6.03, 0.07, -2.68 ],
  "start_rotation": [ 0, 0.7071, 0, 0.7071 ],
  "instruction": {
    "instruction_text": "Exit the bedroom, enter the bathroom, wait at the toilet. ",
    "instruction_tokens": [ 12, 205, 8, 3, 77, … , 0, 0, 0 ]
  },
  "goals": [ { "position": [ 2.15, 0.07, -5.31 ], "radius": 3.0 } ],
  "reference_path": [ [6.03, 0.07, -2.68], … , [2.15, 0.07, -5.31] ],
  "info": { "geodesic_distance": 10.23 }
}
```

| Trường | Kiểu | Nghĩa (giải thích cho người mới) |
|---|---|---|
| `episode_id` | str | **Mã số đề bài**. Dùng để đối chiếu với file `_gt` và để ghi kết quả eval. |
| `trajectory_id` | int | Mã **lộ trình gốc** trong R2R. Một lộ trình thường có **3 câu lệnh** do 3 người viết → 3 episode khác nhau **cùng** `trajectory_id`. |
| `scene_id` | str | Đường dẫn tới **file scene 3D** (`.glb` của Matterport3D). ⚠️ Scene **không** nằm trong `InternData-N1`, phải xin riêng từ Matterport3D. |
| `start_position` | `[x, y, z]` | Điểm robot đứng lúc bắt đầu (mét, hệ Habitat: **y = lên trời**). |
| `start_rotation` | quaternion `[x,y,z,w]` | Hướng nhìn ban đầu. Quaternion = cách ghi phép quay bằng 4 số (xem [01](01_thuat_ngu.md)). |
| `instruction.instruction_text` | str | **Câu lệnh tiếng Anh** — thứ model đọc. Đây chính là "lời của con người". |
| `instruction.instruction_tokens` | list[int] | Câu lệnh đã đổi thành **dãy số** theo `instruction_vocab`, đệm `0` cho bằng độ dài. Dành cho model đời cũ (CMA/Seq2Seq) không có tokenizer riêng. Model VLM (Qwen2.5-VL) **bỏ qua** trường này. |
| `goals` | list | Danh sách đích. `radius: 3.0` nghĩa là **dừng trong bán kính 3 m là tính đúng**. |
| `reference_path` | list[[x,y,z]] | **Đường đi mẫu** (chuỗi waypoint từ xuất phát → đích). Dùng để tính nDTW và để render `traj_data`. |
| `info.geodesic_distance` | float | **Độ dài đường ngắn nhất đi bộ được** từ start → goal (mét). Là mẫu số của công thức **SPL**. |

> 💡 **`geodesic` khác `euclidean`**: đường chim bay xuyên tường là *euclidean*; đường **lách qua cửa,
> vòng qua tường** mới là *geodesic*. Robot chỉ đi được kiểu thứ hai.

### 4.2. `instruction_vocab` — quyển từ điển

```json
"instruction_vocab": {
  "word_list": ["<pad>", "<s>", "</s>", "<unk>", "exit", "the", "bedroom", …],
  "word2idx_dict": { "exit": 4, "the": 5, … },
  "itos": {…}, "stoi": {…},
  "num_vocab": 2504,
  "UNK_INDEX": 3,
  "PAD_INDEX": 0
}
```

Chỉ là bảng tra **từ ↔ số** (2504 từ). Cần cho model cũ; model VLM không dùng.

---

## 5. File `_gt.json.gz` — "đáp án chi tiết"

Ngoài `{split}.json.gz` (đề bài), VLN-CE còn có `{split}_gt.json.gz` (**ground truth** = đáp án).
Trong `vln_ce/raw_data` của InternData-N1, file này **nằm bên trong `rxr.zip`** (R2R bản này không kèm).

Cấu trúc: một `dict` **khoá = `episode_id`**:

```json
{
  "1": {
    "actions":       [3, 3, 1, 1, 2, 1, …, 0],
    "locations":     [[6.03,0.07,-2.68], …],
    "forward_steps": 27
  }
}
```

| Trường | Nghĩa |
|---|---|
| `actions` | Dãy **nút bấm** đúng: `1=tiến, 2=trái, 3=phải, 0=STOP` — cùng bảng mã với cột `action` của parquet ([07](07_phu_luc_lerobot_format.md) mục 2.1). |
| `locations` | Toạ độ robot **sau từng bước** — chuỗi điểm dày, khác `reference_path` (thưa). |
| `forward_steps` | Tổng số bước. |

**Code InternNav dùng nó ở đâu?** Để tính chỉ số **nDTW** (đo quỹ đạo model *giống* quỹ đạo người tới
mức nào), nạp cứng đường dẫn tại
[measures.py:176](../../../code/internnav/habitat_extensions/vln/measures.py#L176):

```python
gt_json_path = 'data/vln_ce/raw_data/rxr/val_unseen/val_unseen_guide_gt.json.gz'
with gzip.open(gt_json_path, "rt") as f:
    self.gt_json = json.load(f)
...
self.gt_locations = self.gt_json[episode.episode_id]["locations"]   # dòng 187
```

> ⚠️ **Đường dẫn này hard-code.** Nếu chưa giải nén `rxr.zip` ra đúng
> `raw_data/rxr/val_unseen/`, dùng measure `ndtw` sẽ **crash ngay khi khởi tạo**.

---

## 6. R2R và RxR khác nhau chỗ nào?

Hai thư mục con của `raw_data` là **hai benchmark khác nhau**, cùng chạy trên scene Matterport3D:

| | **R2R** (Room-to-Room) | **RxR** (Room-across-Room) |
|---|---|---|
| Câu lệnh | **Ngắn**, ~29 từ: *"Exit the bedroom, enter the bathroom, wait at the toilet."* | **Dài, tả rất kỹ**, ~100+ từ, kể cả vật mốc dọc đường |
| Ngôn ngữ | Chỉ tiếng Anh | **3 thứ tiếng**: en / hi (Hindi) / te (Telugu) → InternNav dùng bản `_en` |
| Quy mô | ~14k lệnh | ~126k lệnh (lớn hơn ~10×) → nên nặng 347 MB |
| Vai người nói | 1 loại | **guide** (người hướng dẫn) và **follower** (người đi theo) → tên file có `_guide_` |
| Tên file | `{split}.json.gz` | `{split}_guide_en.json.gz`, đáp án `{split}_guide_gt.json.gz` |

Vì thế hai config eval trỏ hai đường dẫn khác nhau:

```yaml
# scripts/eval/configs/vln_r2r.yaml:77
data_path: data/vln_ce/raw_data/r2r/{split}/{split}.json.gz

# scripts/eval/configs/vln_rxr.yaml:80
data_path: data/vln_ce/raw_data/rxr/{split}/{split}_guide_en.json.gz
```

`{split}` là **chỗ trống** Habitat tự điền bằng giá trị `dataset.split` (mặc định `val_unseen`).

### 6.1. `val_seen` vs `val_unseen` — vì sao phải có hai bộ?

| Split | Nghĩa | Dùng để |
|---|---|---|
| `train` | scene + lệnh **đã cho model học** | huấn luyện |
| `val_seen` | lệnh **mới**, nhưng trong **scene model đã thấy lúc train** | kiểm tra model có hiểu câu lệnh không |
| `val_unseen` | lệnh mới trong **căn nhà hoàn toàn lạ** | ⭐ **thước đo thật sự** — số liệu các paper đem đi so sánh |

Điểm `val_seen` bao giờ cũng cao hơn `val_unseen`. Khoảng cách giữa hai con số cho biết model
**thuộc lòng căn nhà** hay **thật sự biết điều hướng**.

---

## 7. Ai đọc `raw_data` trong code InternNav? (đo thật)

| File:line | Đường dẫn dùng | Bối cảnh |
|---|---|---|
| [vln_r2r.yaml:77](../../../code/scripts/eval/configs/vln_r2r.yaml#L77) | `vln_ce/raw_data/r2r/{split}/{split}.json.gz` | Config eval Habitat cho R2R |
| [vln_rxr.yaml:80](../../../code/scripts/eval/configs/vln_rxr.yaml#L80) | `vln_ce/raw_data/rxr/{split}/{split}_guide_en.json.gz` | Config eval Habitat cho RxR |
| [measures.py:176](../../../code/internnav/habitat_extensions/vln/measures.py#L176) | `vln_ce/raw_data/rxr/val_unseen/val_unseen_guide_gt.json.gz` | Đáp án để tính **nDTW** |
| [habitat_dual_system_cfg.py:22](../../../code/scripts/eval/configs/habitat_dual_system_cfg.py#L22) | trỏ tới `vln_r2r.yaml` | ⭐ Eval **InternVLA-N1 đầy đủ (S1+S2)** |
| [habitat_s2_cfg.py:22](../../../code/scripts/eval/configs/habitat_s2_cfg.py#L22) | trỏ tới `vln_r2r.yaml` | ⭐ Eval **chỉ System 2** |

**Ai thực sự parse file JSON?** Không phải code InternNav, mà là **Habitat-lab**. Config khai báo:

```yaml
# vln_r2r.yaml:74
dataset:
  type: R2RVLN-v1          # ← tên dataset ĐÃ ĐĂNG KÝ SẴN trong habitat-lab
  split: val_unseen
  scenes_dir: data/scene_data/mp3d_ce
```

`R2RVLN-v1` là lớp `R2RVLNDatasetV1` của habitat-lab. Đây là **bằng chứng mạnh** rằng file JSON phải
đúng chuẩn VLN-CE gốc — nếu InternNav đổi schema thì lớp này đã không đọc được.

> 🔎 **Tự kiểm chứng:** `grep -rn "raw_data" InternNav/code/` — bạn sẽ thấy đúng 5 chỗ trên cho
> `vln_ce`, phần còn lại thuộc `vln_pe` hoặc `interiornav_data`.

---

## 8. `raw_data` sinh ra `traj_data` như thế nào? (bằng chứng đo thật trên máy)

Trên máy đã có scene `InternNav/data/vln_ce/traj_data/r2r/17DRP5sb8fy`. Mở file mô tả episode:

```bash
head -2 data/vln_ce/traj_data/r2r/17DRP5sb8fy/meta/episodes.jsonl
```

```json
{"episode_index": 0, "tasks": ["Exit the bedroom, enter the bathroom, wait at the toilet. "], "length": 46}
{"episode_index": 1, "tasks": ["Walk out of the dining area and walk straight into the bedroom that's past the living room. …"], "length": 54}
```

Câu `"Exit the bedroom, enter the bathroom, wait at the toilet. "` **chính là** một
`instruction.instruction_text` của R2R trong scene `17DRP5sb8fy` — kể cả **dấu cách thừa ở cuối**
được giữ nguyên. Đây là bằng chứng trực tiếp cho luồng:

```
raw_data/r2r/train/train.json.gz
   │  (1) đọc từng episode: câu lệnh + start_position + reference_path
   ▼
Habitat + scene .glb  ── (2) cho robot ảo đi theo reference_path, chụp ảnh mỗi bước
   ▼
traj_data/r2r/17DRP5sb8fy/
   ├── meta/episodes.jsonl   ← câu lệnh chép sang đây (trường "tasks")
   ├── data/…parquet         ← action, pose, goal từng frame
   └── videos/…              ← RGB .jpg + depth .png từng frame
```

Mất mát khi chuyển: `traj_data` **không giữ** `episode_id`, `scene_id`, `geodesic_distance`,
`goals.radius`. Nên **không thể chấm điểm SPL từ `traj_data`** — muốn eval vẫn phải có `raw_data`.

---

## 9. Những file "lạ" trong `raw_data/r2r` — giải thích

| File | Nhận xét |
|---|---|
| `val_seen/val_seen.json` (2.5 MB) | Bản **không nén** của `val_seen.json.gz` (225 KB). Nén gzip ~11× vì JSON toàn chữ lặp. Code chỉ mở bản `.gz` → file này chỉ để xem bằng mắt. |
| `val_seen/val_seen/val_seen.json.gz` | **Trùng lặp do đóng gói lỗi** — đúng 225.341 byte, y hệt file cha. Bỏ qua. |
| `val_unseen/val_unseen1.json` (5.5 MB) | ⚠️ Tên có hậu tố `1`, **không nén**, và **lớn gấp đôi** `val_seen.json` — nhiều khả năng là **bản mở rộng/biến thể** do nhóm tác giả thêm vào, **không** file config nào trỏ tới. **Đây là suy đoán**, chưa mở được file để xác nhận (repo gated). Cứ dùng `val_unseen.json.gz`. |
| Thiếu `train_gt.json.gz`, `val_*_gt.json.gz` cho R2R | Bản R2R ở đây **không kèm đáp án chi tiết**. Nếu cần nDTW cho R2R, phải tải thêm từ [trang gốc VLN-CE](https://jacobkrantz.github.io/vlnce/data). |
| Không có split `test` | R2R-VLNCE gốc có `test` (giấu đáp án, nộp lên leaderboard). Bản này lược bỏ. |

---

## 10. So sánh nhanh 3 thư mục `raw_data` trong InternData-N1

| | `vln_ce/raw_data` | `vln_pe/raw_data` | `vln_n1` |
|---|---|---|---|
| Có `raw_data`? | ✅ | ✅ | ❌ (chỉ có `traj_data`) |
| Nội dung | `r2r/{train,val_seen,val_unseen}` + `rxr/rxr.zip` | `r2r/{train,val_seen,val_unseen}` + **`embeddings.json.gz`** (1.0 MB) | — |
| `train.json.gz` | 2.408.561 B | 2.655.407 B ← **khác file**, đã lọc lại cho robot có vật lý | — |
| Dùng bởi | eval Habitat (S2 & dual-system) | train/eval baseline **CMA, Seq2Seq, RDP** trong InternUtopia/Isaac | train **System 1 (NavDP)** |
| Trích dẫn code | [vln_r2r.yaml:77](../../../code/scripts/eval/configs/vln_r2r.yaml#L77) | [cma.py:18-19](../../../code/internnav/configs/model/cma.py#L18), [h1_cma_cfg.py:45](../../../code/scripts/eval/configs/h1_cma_cfg.py#L45) | [03b](03b_code_train_s1.md) mục 126 |

`embeddings.json.gz` (chỉ có ở `vln_pe`) = **vector nhúng sẵn cho từng từ** trong từ điển, để model
cũ khỏi phải học lại embedding từ đầu — khai báo tại
[cma.py:18](../../../code/internnav/configs/model/cma.py#L18).

---

## 11. Thực hành: tải và bày đúng chỗ

`InternData-N1` là repo **gated** → phải đăng nhập HF và được duyệt trước.

```bash
huggingface-cli login

# Chỉ tải phần raw_data của vln_ce (~370 MB), bỏ qua traj_data khổng lồ
huggingface-cli download InternRobotics/InternData-N1 \
  --repo-type dataset \
  --include "vln_ce/raw_data/**" \
  --local-dir data/

# BẮT BUỘC: giải nén RxR tại chỗ
cd data/vln_ce/raw_data/rxr && unzip rxr.zip && cd -
```

Kết quả cần đạt (khớp với đường dẫn hard-code trong code):

```
InternNav/
└── data/
    ├── vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz          ← vln_r2r.yaml:77
    ├── vln_ce/raw_data/rxr/val_unseen/val_unseen_guide_en.json.gz ← vln_rxr.yaml:80
    ├── vln_ce/raw_data/rxr/val_unseen/val_unseen_guide_gt.json.gz ← measures.py:176
    └── scene_data/mp3d_ce/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb        ← xin riêng từ Matterport3D
```

> ⚠️ Đường dẫn trong config là **tương đối so với thư mục chạy lệnh** (thường là `InternNav/code/`),
> không phải tuyệt đối. Chạy sai chỗ → `FileNotFoundError`.

**Kiểm tra nhanh sau khi tải:**

```python
import gzip, json
d = json.load(gzip.open('data/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz', 'rt'))
print(len(d['episodes']))                                 # số episode
print(d['episodes'][0]['instruction']['instruction_text']) # câu lệnh đầu tiên
print(d['episodes'][0]['scene_id'])                        # scene nào → tải scene đó
```

---

## 12. Vậy **bạn** có cần `raw_data` không?

| Bạn đang làm gì | Cần `vln_ce/raw_data`? |
|---|---|
| Train System 2 trên data có sẵn ([03](03_code_train_s2.md), [04](04_data_train_s2.md)) | ❌ Không. Loader chỉ đọc `traj_data`. |
| Sinh data S2 từ `.mcap` / `.db3` của robot thật ([06](06_pipeline_mcap_to_s2.md), [06c](06c_pipeline_db3_to_s2.md)) | ❌ Không. Bạn **tự sinh** phần tương đương. |
| Train System 1 / NavDP ([03b](03b_code_train_s1.md), [05](05_data_train_s1.md)) | ❌ Không. Dùng `vln_n1/traj_data`. |
| **Chạy eval trong Habitat** để lấy SR / SPL / nDTW | ✅ **Có, bắt buộc.** |
| **Tự render lại `traj_data`** với cấu hình camera riêng | ✅ Có — nó là "đề bài" để render. |
| Chỉ muốn **đọc xem câu lệnh R2R trông thế nào** | 💡 Không cần tải: xem `meta/tasks.jsonl` trong bất kỳ scene `traj_data` nào đã có (mục 8). |

---

## 13. Tóm tắt 5 gạch đầu dòng

1. `raw_data` = **đề bài dạng văn bản** (JSON nén): câu lệnh + start + goal + đường mẫu. **Không có ảnh.**
2. `traj_data` = **bài giải đã quay video** từ chính các đề bài đó, đóng gói LeRobot. Đây mới là data train.
3. Cấu trúc: `r2r/{train,val_seen,val_unseen}/*.json.gz` + `rxr/rxr.zip` (**phải giải nén**).
4. Trong code InternNav, `vln_ce/raw_data` chỉ xuất hiện ở **5 chỗ**, tất cả đều thuộc nhánh **evaluation** —
   không dính gì tới vòng lặp train.
5. Train S2 với data tự thu (`.mcap`/`.db3`) → **bỏ qua `raw_data` hoàn toàn**.
