"""
mcap_inspect.py — Khảo sát CẤU TRÚC của một file `.mcap` lạ (không biết trước ai ghi, ghi bằng gì).

=============================================================================
DÙNG KHI NÀO
-----------------------------------------------------------------------------
Ai đó gửi cho bạn một file `.mcap`. Bạn chưa biết nó có topic gì, message type gì,
mã hoá kiểu gì. Tool này trả lời đúng 3 câu hỏi đó rồi ghi ra JSON — KHÔNG đọc/giải mã
nội dung dữ liệu (không ảnh, không toạ độ, không giá trị mẫu).

    file_la.mcap  ──►  file_la.mcap.inspect.json
                       {
                         "file", "header",        # ai ghi, thư viện nào, profile gì
                         "statistics",            # bao nhiêu message, bao lâu, nén gì
                         "encodings",             # tổng quan message/schema encoding
                         "metadata_records",      # ghi chú người ghi nhét vào file
                         "attachments",
                         "topics":  [...],        # MỌI topic + type + số message + nhịp
                         "schemas": [...],        # cấu trúc TỪNG message type (cây field)
                         "warnings": [...]        # chỗ nào tool không chắc → nói thẳng
                       }

=============================================================================
CÁCH CHẠY
-----------------------------------------------------------------------------
    pip install mcap                    # bắt buộc
    pip install protobuf                # tuỳ chọn: để mở schema protobuf

    python mcap_inspect.py --mcap file_la.mcap              # nhanh, không đọc message
    python mcap_inspect.py --mcap file_la.mcap --deep       # + nhịp/cỡ byte chính xác
    python mcap_inspect.py --mcap file_la.mcap --out report.json --no-fields

Hai chế độ khác nhau chỗ nào:
    mặc định  chỉ đọc summary + schema ở cuối file. Vài mili giây kể cả file 10 GB.
              `rate_hz` là ƯỚC LƯỢNG (chia đều theo thời lượng cả file) — bảng in dấu `~`.
    --deep    đọc tuần tự từ byte đầu tới byte cuối: nhịp thật của từng topic, cỡ byte
              min/max/trung bình. Vẫn KHÔNG giải mã nội dung message.

File bị cắt ngang / ghi dở / không có index (rất hay gặp khi người ta copy giữa chừng):
tool tự chuyển sang đọc tuần tự, giữ lại mọi thứ đọc được và ghi rõ dừng ở đâu trong
`warnings` — thay vì báo file hỏng rồi bỏ cuộc.

=============================================================================
CẤU TRÚC LẤY TỪ ĐÂU (quan trọng — quyết định mức độ tin cậy)
-----------------------------------------------------------------------------
File mcap TỰ MÔ TẢ: mỗi topic trỏ tới một record Schema mô tả sẵn cấu trúc message.
Tool đọc thẳng schema đó, nên KHÔNG cần ROS, không cần biết trước định dạng:

    ros2msg / ros1msg  → parser .msg viết sẵn trong file này (kể cả type lồng nhau)
    jsonschema         → đọc `properties` / `items` / `required`
    protobuf           → giải FileDescriptorSet (cần `pip install protobuf`)
    ros2idl / omgidl   → parser `struct` rút gọn
    flatbuffer, khác   → chưa parse được: in schema thô để bạn tự đọc

Chỉ khi một topic KHÔNG có schema (mcap cho phép) tool mới đọc vài message đầu để đoán
tên + kiểu field — và chỉ khi message là JSON. Trường hợp này `structure_source` sẽ ghi
`sampled:json` thay vì `schema:<encoding>`; field vắng trong mẫu vẫn có thể tồn tại.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from mcap.reader import make_reader

try:  # console Windows mặc định cp1252 sẽ làm vỡ tiếng Việt ở cả báo cáo lẫn thông báo lỗi
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

NS = 1_000_000_000
MAGIC = b"\x89MCAP0\r\n"  # 8 byte mở đầu mọi file mcap hợp lệ
MAX_DEPTH = 12  # chặn type tự tham chiếu (ví dụ cây TF lồng nhau) làm đệ quy vô hạn

# Kiểu nguyên thuỷ của ROS: gặp thì dừng, không đi tìm định nghĩa con.
ROS_PRIMITIVES = {
    "bool", "byte", "char", "float32", "float64", "int8", "uint8", "int16", "uint16",
    "int32", "uint32", "int64", "uint64", "string", "wstring", "time", "duration",
}


# =============================================================================
# GIAI ĐOẠN A — PARSER SCHEMA: ros1msg / ros2msg
# =============================================================================
def strip_comment(line: str) -> str:
    """Bỏ comment `#` nhưng chừa dấu `#` nằm trong chuỗi (default value của ROS 2)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def split_ros_blocks(text: str) -> List[Tuple[Optional[str], str]]:
    """Tách định nghĩa .msg nối chuỗi thành `[(tên_type, thân), ...]`; phần tử đầu là type gốc.

    mcap nhét TOÀN BỘ cây type vào một schema, ngăn nhau bằng dòng `====...` rồi `MSG: pkg/Type`.
    """
    blocks: List[Tuple[Optional[str], str]] = []
    name: Optional[str] = None
    body: List[str] = []
    for line in text.splitlines():
        if line.startswith("===="):
            blocks.append((name, "\n".join(body)))
            name, body = None, []
            continue
        if name is None and not body and line.strip().startswith("MSG:"):
            name = line.split(":", 1)[1].strip()
            continue
        body.append(line)
    blocks.append((name, "\n".join(body)))
    return blocks


ROS_FIELD_RE = re.compile(
    r"^\s*(?P<type>[A-Za-z_][\w/]*(?:<=\d+)?)"      # kiểu, có thể bounded string<=10
    r"(?P<array>\[[^\]]*\])?"                        # [] | [36] | [<=5]
    r"\s+(?P<name>[A-Za-z_]\w*)\s*(?P<rest>.*)$"     # tên + (hằng số | default)
)


def parse_array_spec(spec: Optional[str]) -> Optional[Dict[str, Any]]:
    """`[]` → unbounded, `[36]` → fixed 36, `[<=5]` → bounded 5."""
    if spec is None:
        return None
    inner = spec[1:-1].strip()
    if not inner:
        return {"kind": "unbounded"}
    if inner.startswith("<="):
        return {"kind": "bounded", "max": int(inner[2:] or 0)}
    return {"kind": "fixed", "size": int(inner)} if inner.isdigit() else {"kind": "unbounded", "raw": inner}


def parse_ros_msg(text: str, root_name: str) -> Dict[str, Any]:
    """Dựng cây field từ schema ros1msg/ros2msg, mở luôn các type lồng nhau."""
    blocks = split_ros_blocks(text)
    defs: Dict[str, str] = {}
    for name, body in blocks[1:]:
        if name:
            defs[name.replace("/msg/", "/")] = body

    def lookup(type_name: str) -> Optional[str]:
        key = type_name.replace("/msg/", "/")
        if key in defs:
            return defs[key]
        # ROS 1 hay ghi type tương đối ("Header"): khớp theo phần đuôi.
        matches = [b for k, b in defs.items() if k.rsplit("/", 1)[-1] == key.rsplit("/", 1)[-1]]
        return matches[0] if len(matches) == 1 else None

    def build(body: str, depth: int, seen: Tuple[str, ...]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        constants: List[Dict[str, Any]] = []
        for raw in body.splitlines():
            line = strip_comment(raw).rstrip()
            if not line.strip():
                continue
            m = ROS_FIELD_RE.match(line)
            if not m:
                continue
            type_name, name, rest = m.group("type"), m.group("name"), m.group("rest").strip()
            if rest.startswith("="):  # `uint8 STATUS_OK=1` là hằng số, không phải field
                constants.append({"name": name, "type": type_name, "value": rest[1:].strip()})
                continue

            node: Dict[str, Any] = {"type": type_name}
            array = parse_array_spec(m.group("array"))
            if array:
                node["array"] = array
            if rest:
                node["default"] = rest

            base = type_name.split("<=")[0]
            if base not in ROS_PRIMITIVES and depth < MAX_DEPTH and base not in seen:
                sub = lookup(base)
                if sub is not None:
                    child = build(sub, depth + 1, seen + (base,))
                    node.update({k: v for k, v in child.items() if k in ("fields", "constants")})
                else:
                    node["note"] = "không tìm thấy định nghĩa trong schema"
            elif base in seen:
                node["note"] = "đệ quy — đã mở ở cấp trên"
            fields[name] = node

        out: Dict[str, Any] = {"type": "message", "fields": fields}
        if constants:
            out["constants"] = constants
        return out

    tree = build(blocks[0][1], 0, (root_name,))
    tree["type"] = root_name
    return tree


# =============================================================================
# GIAI ĐOẠN A — PARSER SCHEMA: jsonschema
# =============================================================================
def parse_jsonschema(node: Any, root: Any = None, depth: int = 0) -> Dict[str, Any]:
    """Dựng cây field từ JSON Schema; `$ref` nội bộ (`#/...`) được mở, ref ngoài thì ghi chú."""
    if not isinstance(node, dict):
        return {"type": "any"}
    root = root if root is not None else node

    ref = node.get("$ref")
    if ref and depth < MAX_DEPTH:
        if ref.startswith("#/"):
            target: Any = root
            for part in ref[2:].split("/"):
                target = target.get(part) if isinstance(target, dict) else None
            if target is not None:
                return parse_jsonschema(target, root, depth + 1)
        return {"type": "any", "note": f"$ref chưa mở: {ref}"}

    jtype = node.get("type") or ("object" if "properties" in node else "any")
    if isinstance(jtype, list):
        jtype = " | ".join(jtype)
    out: Dict[str, Any] = {"type": jtype}
    for key in ("format", "contentEncoding", "description", "enum"):
        if key in node:
            out[key] = node[key]

    if jtype == "array":
        out = dict(parse_jsonschema(node.get("items", {}), root, depth + 1))  # kiểu = kiểu phần tử
        lo, hi = node.get("minItems"), node.get("maxItems")
        if hi is not None and lo == hi:
            out["array"] = {"kind": "fixed", "size": hi}
        elif hi is not None:
            out["array"] = {"kind": "bounded", "max": hi}
        else:
            out["array"] = {"kind": "unbounded"}
    elif jtype == "object" and depth < MAX_DEPTH:
        required = set(node.get("required", []))
        fields = {}
        for name, sub in (node.get("properties") or {}).items():
            child = parse_jsonschema(sub, root, depth + 1)
            if required and name not in required:
                child["optional"] = True
            fields[name] = child
        out["fields"] = fields
    return out


# =============================================================================
# GIAI ĐOẠN A — PARSER SCHEMA: protobuf
# =============================================================================
def parse_protobuf(data: bytes, root_name: str) -> Dict[str, Any]:
    """Giải FileDescriptorSet (schema protobuf trong mcap là nhị phân, không phải .proto text)."""
    from google.protobuf import descriptor_pb2  # import trong hàm: đây là dependency tuỳ chọn

    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(data)
    field_proto = descriptor_pb2.FieldDescriptorProto
    type_names = {v: k[len("TYPE_"):].lower() for k, v in field_proto.Type.items()}
    labels = {v: k[len("LABEL_"):].lower() for k, v in field_proto.Label.items()}

    messages: Dict[str, Any] = {}
    enums: Dict[str, List[str]] = {}

    def collect(prefix: str, msg) -> None:
        full = f"{prefix}.{msg.name}" if prefix else msg.name
        messages[full] = msg
        for nested in msg.nested_type:
            collect(full, nested)
        for enum in msg.enum_type:
            enums[f"{full}.{enum.name}"] = [v.name for v in enum.value]

    for file in fds.file:
        for msg in file.message_type:
            collect(file.package, msg)
        for enum in file.enum_type:
            enums[f"{file.package}.{enum.name}" if file.package else enum.name] = [v.name for v in enum.value]

    def build(msg, depth: int, seen: Tuple[str, ...]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        for f in msg.field:
            node: Dict[str, Any] = {"type": type_names.get(f.type, str(f.type)), "number": f.number}
            if labels.get(f.label) == "repeated":
                node["array"] = {"kind": "unbounded"}
            elif labels.get(f.label) == "optional" and f.proto3_optional:
                node["optional"] = True
            if f.type_name:
                ref = f.type_name.lstrip(".")
                node["type"] = ref
                if ref in enums:
                    node["enum"] = enums[ref]
                elif ref in messages and depth < MAX_DEPTH and ref not in seen:
                    node["fields"] = build(messages[ref], depth + 1, seen + (ref,)).get("fields", {})
                elif ref in seen:
                    node["note"] = "đệ quy — đã mở ở cấp trên"
            fields[f.name] = node
        return {"type": "message", "fields": fields}

    root = messages.get(root_name) or messages.get(root_name.replace("/", "."))
    if root is None:  # tên schema không khớp message nào → lấy message đầu tiên
        if not messages:
            raise ValueError("FileDescriptorSet rỗng")
        name, root = next(iter(messages.items()))
        tree = build(root, 0, (name,))
        tree["type"] = name
        tree["note"] = f"không tìm thấy message tên {root_name!r}, hiển thị {name!r}"
        return tree
    tree = build(root, 0, (root_name,))
    tree["type"] = root_name
    return tree


# =============================================================================
# GIAI ĐOẠN A — PARSER SCHEMA: IDL (ros2idl / omgidl) — rút gọn
# =============================================================================
IDL_STRUCT_RE = re.compile(r"struct\s+(\w+)\s*\{(.*?)\}", re.S)
IDL_FIELD_RE = re.compile(r"^\s*(?P<type>[\w:<>, ]+?)\s+(?P<name>\w+)\s*(?P<array>\[\d*\])?\s*;")


def parse_idl(text: str, root_name: str) -> Dict[str, Any]:
    """Bóc các `struct` trong IDL. Không phải parser IDL đầy đủ — đủ để nhìn ra field."""
    structs: Dict[str, Dict[str, Any]] = {}
    for name, body in IDL_STRUCT_RE.findall(text):
        fields: Dict[str, Any] = {}
        for line in body.splitlines():
            m = IDL_FIELD_RE.match(strip_comment(line))
            if not m:
                continue
            node: Dict[str, Any] = {"type": m.group("type").strip()}
            if m.group("array"):
                node["array"] = parse_array_spec(m.group("array"))
            if node["type"].startswith("sequence<"):
                node["type"] = node["type"][len("sequence<"):].rstrip(">").split(",")[0].strip()
                node["array"] = {"kind": "unbounded"}
            fields[m.group("name")] = node
        structs[name] = fields

    short = root_name.rsplit("/", 1)[-1].rsplit("::", 1)[-1]
    fields = structs.get(short) or (next(iter(structs.values())) if structs else {})
    tree: Dict[str, Any] = {"type": root_name, "fields": fields}
    if len(structs) > 1:
        tree["other_structs"] = [k for k in structs if k != short]
    return tree


# =============================================================================
# GIAI ĐOẠN A — ĐIỀU PHỐI: chọn parser theo `schema.encoding`
# =============================================================================
def parse_schema(schema) -> Dict[str, Any]:
    """Chuyển record Schema thành `{id, name, encoding, structure, structure_source, ...}`."""
    out: Dict[str, Any] = {
        "id": schema.id,
        "name": schema.name,
        "encoding": schema.encoding,
        "size_bytes": len(schema.data),
    }
    enc = (schema.encoding or "").lower()
    text: Optional[str] = None
    if enc not in ("protobuf", "flatbuffer"):
        try:
            text = schema.data.decode("utf-8")
        except UnicodeDecodeError:
            pass

    try:
        if enc in ("ros1msg", "ros2msg") and text is not None:
            out["structure"] = parse_ros_msg(text, schema.name)
        elif enc == "jsonschema" and text is not None:
            out["structure"] = parse_jsonschema(json.loads(text))
            out["structure"]["type"] = schema.name
        elif enc == "protobuf":
            out["structure"] = parse_protobuf(schema.data, schema.name)
        elif enc in ("ros2idl", "omgidl", "idl") and text is not None:
            out["structure"] = parse_idl(text, schema.name)
        else:
            out["warning"] = f"chưa có parser cho schema encoding {schema.encoding!r} — xem definition thô"
    except ImportError:
        out["warning"] = "schema protobuf: cần `pip install protobuf` để mở cấu trúc"
    except Exception as exc:  # parser hỏng KHÔNG được làm hỏng cả báo cáo
        out["warning"] = f"không parse được schema ({type(exc).__name__}: {exc}) — xem definition thô"

    out["structure_source"] = f"schema:{schema.encoding}" if "structure" in out else "unavailable"
    if text is not None:
        out["definition_text"] = text
    elif enc != "protobuf" or "structure" not in out:
        out["definition_base64"] = base64.b64encode(schema.data).decode("ascii")
    return out


# =============================================================================
# GIAI ĐOẠN B — DỰ PHÒNG: đoán cấu trúc từ payload JSON (chỉ khi topic KHÔNG có schema)
# =============================================================================
def describe_shape(value: Any) -> Dict[str, Any]:
    """Mô tả KIỂU của một giá trị JSON. Cố tình không giữ lại giá trị — ta chỉ cần cấu trúc."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):  # bool là subclass của int → phải xét trước
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        items: Optional[Dict[str, Any]] = None
        for elem in value[:64]:
            items = merge_shape(items, describe_shape(elem))
        node = dict(items or {"type": "unknown"})
        node["array"] = {"kind": "fixed", "size": len(value)}
        return node
    if isinstance(value, dict):
        return {"type": "object", "fields": {k: describe_shape(v) for k, v in value.items()}}
    return {"type": type(value).__name__}


def merge_shape(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> Dict[str, Any]:
    """Hợp nhất mô tả nhiều mẫu; field không có ở mọi mẫu bị đánh dấu `optional`."""
    if a is None:
        return b
    if a.get("type") != b.get("type"):
        return {"type": " | ".join(sorted({str(a.get("type")), str(b.get("type"))}))}

    out = dict(a)
    if "fields" in a or "fields" in b:
        fa, fb = a.get("fields", {}), b.get("fields", {})
        fields: Dict[str, Any] = {}
        for key in list(fa) + [k for k in fb if k not in fa]:
            if key in fa and key in fb:
                fields[key] = merge_shape(fa[key], fb[key])
            else:
                fields[key] = dict(fa.get(key) or fb[key], optional=True)
        out["fields"] = fields
    bounds = [x for x in (array_bounds(a), array_bounds(b)) if x]
    if len(bounds) == 2:  # gộp độ dài quan sát được; khác nhau giữa các mẫu ⇒ mảng động
        lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
        out["array"] = {"kind": "fixed", "size": lo} if lo == hi else {"kind": "variable", "min": lo, "max": hi}
    return out


def array_bounds(node: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Độ dài (min, max) đã quan sát của một node mảng; None nếu node không phải mảng."""
    array = node.get("array") or {}
    if array.get("kind") == "fixed":
        return array["size"], array["size"]
    if array.get("kind") == "variable":
        return array["min"], array["max"]
    return None


# =============================================================================
# GIAI ĐOẠN C — QUÉT FILE
# =============================================================================
def stream_scan(path: str, samples: int = 0) -> Dict[str, Any]:
    """Đọc TUẦN TỰ từ byte đầu tiên, gom mọi record trong một lượt.

    Cố tình không dùng reader cấp cao: chúng cần footer/index nên bó tay với file bị cắt
    ngang hoặc ghi dở — mà file người khác gửi thì rất hay như vậy. `StreamReader` đọc tới
    đâu lấy tới đó; gặp lỗi thì ta GIỮ phần đã đọc được và ghi vào `error`.

    Chỉ lấy `len(message.data)`, không giải mã payload.
    """
    from mcap.stream_reader import StreamReader

    out: Dict[str, Any] = {
        "books": {}, "channels": {}, "schemas": {}, "metadata": [], "attachments": [],
        "statistics": None, "records": 0, "error": None, "payloads": {},
    }
    try:
        with open(path, "rb") as f:
            for rec in StreamReader(f).records:
                out["records"] += 1
                kind = type(rec).__name__
                if kind == "Schema":
                    out["schemas"].setdefault(rec.id, rec)
                elif kind == "Channel":
                    out["channels"].setdefault(rec.id, rec)
                elif kind == "Metadata":
                    out["metadata"].append({"name": rec.name, "metadata": dict(rec.metadata)})
                elif kind == "Attachment":
                    out["attachments"].append({"name": rec.name, "media_type": rec.media_type,
                                               "size_bytes": len(rec.data), "log_time_ns": rec.log_time})
                elif kind == "Statistics":
                    out["statistics"] = rec
                elif kind == "Message":
                    book = out["books"].setdefault(
                        rec.channel_id, {"n": 0, "t0": None, "t1": None, "lo": None, "hi": None, "total": 0}
                    )
                    size = len(rec.data)
                    book["n"] += 1
                    book["total"] += size
                    book["t0"] = rec.log_time if book["t0"] is None else min(book["t0"], rec.log_time)
                    book["t1"] = rec.log_time if book["t1"] is None else max(book["t1"], rec.log_time)
                    book["lo"] = size if book["lo"] is None else min(book["lo"], size)
                    book["hi"] = size if book["hi"] is None else max(book["hi"], size)
                    # Channel không khai schema: giữ vài payload JSON để lát nữa đoán cấu trúc.
                    channel = out["channels"].get(rec.channel_id)
                    if (samples and channel is not None and not channel.schema_id
                            and (channel.message_encoding or "").lower() == "json"):
                        kept = out["payloads"].setdefault(channel.topic, [])
                        if len(kept) < samples:
                            kept.append(rec.data)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def peek_compression(path: str) -> Optional[List[str]]:
    """Nén bằng gì? Đọc tới Chunk ĐẦU TIÊN với `emit_chunks` (không giải nén) rồi dừng.

    Cần lượt riêng vì lượt quét chính giải nén chunk nên không còn thấy record Chunk nữa.
    Trả `[]` nếu file không dùng chunk, `None` nếu không xác định được.
    """
    from mcap.stream_reader import StreamReader

    try:
        with open(path, "rb") as f:
            for rec in StreamReader(f, emit_chunks=True).records:
                kind = type(rec).__name__
                if kind == "Chunk":
                    return [rec.compression] if rec.compression else []
                if kind in ("Message", "DataEnd"):  # message nằm ngoài chunk ⇒ không nén
                    return []
    except Exception:
        return None
    return None


def shape_of(payloads: List[bytes]) -> Optional[Dict[str, Any]]:
    """Hợp nhất cấu trúc của vài payload JSON. `None` nếu không cái nào là JSON hợp lệ."""
    shape: Optional[Dict[str, Any]] = None
    for raw in payloads:
        try:
            shape = merge_shape(shape, describe_shape(json.loads(raw)))
        except (ValueError, UnicodeDecodeError):
            continue
    return shape


def sample_shapes(path: str, topics: List[str], samples: int) -> Dict[str, Dict[str, Any]]:
    """Đọc tối đa `samples` message của riêng vài topic để đoán cấu trúc khi thiếu schema.

    Dùng reader có index nên chỉ chạm vào đúng chunk cần — nhanh kể cả với file lớn.
    """
    kept: Dict[str, List[bytes]] = {t: [] for t in topics}
    with open(path, "rb") as f:
        for _, channel, message in make_reader(f).iter_messages(topics=topics):
            box = kept.setdefault(channel.topic, [])
            if len(box) < samples:
                box.append(message.data)
            if all(len(v) >= samples for v in kept.values()):
                break
    return {topic: shape for topic, raw in kept.items() if (shape := shape_of(raw)) is not None}


def short_list(items: List[str], limit: int = 6) -> str:
    """Nối danh sách tên topic, cắt bớt để một cảnh báo không chiếm cả màn hình."""
    head = ", ".join(items[:limit])
    return head + (f" … (+{len(items) - limit})" if len(items) > limit else "")


def inspect(path: str, samples: int, deep: bool, allow_sample: bool) -> Dict[str, Any]:
    """Khảo sát toàn bộ file, trả về báo cáo dict. Mọi lỗi cục bộ đều thành `warnings`."""
    warnings: List[str] = []
    header = None
    summary = None
    with open(path, "rb") as f:
        try:
            reader = make_reader(f)
            header = reader.get_header()
        except Exception as exc:
            warnings.append(f"Không đọc được Header: {type(exc).__name__}: {exc}")
            reader = None
        try:
            summary = reader.get_summary() if reader else None
        except Exception as exc:
            warnings.append(f"Không đọc được Summary: {type(exc).__name__}: {exc}")

    channels = dict(summary.channels) if summary else {}
    raw_schemas = dict(summary.schemas) if summary else {}
    stats = summary.statistics if summary else None
    if summary is None or not channels:
        warnings.append("File không có summary/index (bị cắt, ghi dở, hoặc writer không ghi index) "
                        "— chuyển sang đọc tuần tự từ đầu file.")

    # Quét message chỉ khi thật cần: file thiếu index, hoặc người dùng yêu cầu --deep.
    books: Dict[int, Dict[str, Any]] = {}
    scan: Dict[str, Any] = {}
    scanned = deep or summary is None or not channels or stats is None
    if scanned:
        scan = stream_scan(path, samples if allow_sample else 0)
        books = scan["books"]
        for cid, channel in scan["channels"].items():
            channels.setdefault(cid, channel)
        for sid, schema in scan["schemas"].items():
            raw_schemas.setdefault(sid, schema)
        stats = stats or scan["statistics"]
        if scan["error"]:
            warnings.append(
                f"FILE KHÔNG ĐỌC HẾT ĐƯỢC: dừng sau {scan['records']} record — {scan['error']}. "
                "Mọi số liệu dưới đây chỉ tính trên phần đọc được."
            )

    schemas = {sid: parse_schema(s) for sid, s in raw_schemas.items()}
    for s in schemas.values():
        if s.get("warning"):
            warnings.append(f"schema #{s['id']} {s['name']!r}: {s['warning']}")

    counts = dict(stats.channel_message_counts) if stats else {cid: b["n"] for cid, b in books.items()}
    starts = [b["t0"] for b in books.values() if b.get("t0") is not None]
    ends = [b["t1"] for b in books.values() if b.get("t1") is not None]
    t_start = stats.message_start_time if stats and stats.message_start_time else (min(starts) if starts else None)
    t_end = stats.message_end_time if stats and stats.message_end_time else (max(ends) if ends else None)
    total_span = ((t_end - t_start) / NS) if t_start is not None and t_end else None

    # Hai loại topic thiếu cấu trúc, phân biệt rạch ròi vì cách xử lý khác hẳn nhau.
    no_schema = sorted(c.topic for c in channels.values() if not c.schema_id)
    unparsed = sorted(
        c.topic for c in channels.values()
        if c.schema_id and "structure" not in schemas.get(c.schema_id, {})
    )
    if no_schema:
        warnings.append(f"{len(no_schema)} topic không khai báo schema: {short_list(no_schema)}")
    if unparsed:
        warnings.append(f"{len(unparsed)} topic có schema nhưng tool chưa parse được: {short_list(unparsed)}")

    # Chỉ đoán được khi payload là JSON tự mô tả; với cdr/protobuf thì đọc byte thô cũng vô nghĩa.
    guessable = sorted(
        c.topic for c in channels.values()
        if c.topic in set(no_schema) | set(unparsed) and (c.message_encoding or "").lower() == "json"
    )
    shapes: Dict[str, Dict[str, Any]] = {}
    for topic, payloads in (scan.get("payloads") or {}).items():  # lượt quét vừa rồi đã nhặt sẵn
        shapes[topic] = shape_of(payloads)
    missing = [t for t in guessable if t not in shapes]
    if missing and allow_sample:
        try:
            shapes.update(sample_shapes(path, missing, samples))
        except Exception as exc:
            warnings.append(f"Không lấy được mẫu để đoán cấu trúc: {type(exc).__name__}: {exc}")
    shapes = {t: s for t, s in shapes.items() if s is not None}
    unknown = sorted(t for t in set(no_schema) | set(unparsed) if t not in shapes)
    if unknown:
        warnings.append(
            f"{len(unknown)} topic không suy được cấu trúc, payload không phải JSON — "
            f"phải hỏi người gửi message type: {short_list(unknown)}"
        )

    topics: List[Dict[str, Any]] = []
    for cid, channel in sorted(channels.items(), key=lambda kv: kv[1].topic):
        book = books.get(cid, {})
        n = counts.get(cid, book.get("n", 0))
        schema = schemas.get(channel.schema_id, {})
        span = ((book["t1"] - book["t0"]) / NS) if book.get("t0") is not None and book.get("t1") else None
        rate = (n / span) if span else ((n / total_span) if total_span else None)
        entry: Dict[str, Any] = {
            "topic": channel.topic,
            "channel_id": cid,
            "message_encoding": channel.message_encoding,
            "schema": {"id": channel.schema_id or None, "name": schema.get("name"),
                       "encoding": schema.get("encoding")},
            "message_count": n,
            "rate_hz": round(rate, 4) if rate else None,
            "rate_is_estimate": None if span else (True if rate else None),
            "duration_s": round(span, 6) if span else None,
            "log_time_ns": {"first": book["t0"], "last": book["t1"]} if book.get("t0") is not None else None,
            "message_size_bytes": (
                {"min": book["lo"], "max": book["hi"], "mean": round(book["total"] / book["n"], 1),
                 "total": book["total"]}
                if book.get("n") else None
            ),
            "channel_metadata": dict(channel.metadata),
            "structure_source": schema.get("structure_source", "unavailable"),
        }
        if "structure" in schema:
            entry["field_list"] = flatten(schema["structure"])
        elif channel.topic in shapes:
            entry["structure_source"] = "sampled:json"
            entry["inferred_structure"] = shapes[channel.topic]
            entry["field_list"] = flatten(shapes[channel.topic])
        else:
            entry["field_list"] = []
        topics.append(entry)

    used = {c.schema_id for c in channels.values()}
    for sid, schema in schemas.items():
        schema["used_by_topics"] = sorted(c.topic for c in channels.values() if c.schema_id == sid)
        if sid not in used:
            warnings.append(f"schema #{sid} {schema['name']!r} không được topic nào dùng")

    return {
        "file": {"path": os.path.abspath(path), "size_bytes": os.path.getsize(path)},
        "header": {"profile": header.profile, "library": header.library} if header else None,
        "statistics": {
            "message_count": stats.message_count if stats else sum(counts.values()),
            "schema_count": len(schemas),
            "channel_count": len(channels),
            "chunk_count": stats.chunk_count if stats else None,
            "attachment_count": stats.attachment_count if stats else 0,
            "metadata_count": stats.metadata_count if stats else 0,
            "message_start_time_ns": t_start,
            "message_end_time_ns": t_end,
            "duration_s": round(total_span, 6) if total_span else None,
        },
        "encodings": {
            "message": sorted({c.message_encoding for c in channels.values() if c.message_encoding}),
            "schema": sorted({s["encoding"] for s in schemas.values() if s.get("encoding")}),
            "compression": (
                sorted({ci.compression for ci in summary.chunk_indexes if ci.compression})
                if summary and summary.chunk_indexes else peek_compression(path)
            ),
        },
        # Lượt quét tuần tự (nếu có chạy) đã gom sẵn; nếu không thì đọc riêng qua index.
        "metadata_records": scan.get("metadata") or read_metadata(path, warnings),
        "attachments": [
            {"name": a.name, "media_type": a.media_type, "size_bytes": a.data_size, "log_time_ns": a.log_time}
            for a in (summary.attachment_indexes if summary else [])
        ] or scan.get("attachments", []),
        "topics": topics,
        "schemas": [schemas[k] for k in sorted(schemas)],
        "warnings": warnings,
        "scan": {
            "mode": "deep (đọc tuần tự toàn file)" if scanned else "shallow (chỉ đọc summary + schema)",
            "records_read": scan.get("records"),
            "stopped_early": scan.get("error"),
            "samples_per_schemaless_topic": samples if allow_sample else 0,
            "note": None if scanned else "rate_hz là ƯỚC LƯỢNG theo thời lượng cả file; chạy --deep để đo thật.",
        },
    }


def read_metadata(path: str, warnings: List[str]) -> List[Dict[str, Any]]:
    """Đọc các record `metadata` — nơi người ghi hay nhét thông số hiệu chỉnh, tên robot, phiên bản."""
    try:
        with open(path, "rb") as f:
            return [{"name": r.name, "metadata": dict(r.metadata)} for r in make_reader(f).iter_metadata()]
    except Exception as exc:
        warnings.append(f"Không đọc được metadata records: {type(exc).__name__}: {exc}")
        return []


# =============================================================================
# GIAI ĐOẠN D — IN RA MÀN HÌNH + GHI FILE
# =============================================================================
def type_label(node: Dict[str, Any]) -> str:
    """Ghép nhãn kiểu để in: `float64[36]`, `string[]`, `Pose ?`."""
    label = str(node.get("type", "?"))
    array = node.get("array")
    if array:
        kind = array.get("kind")
        label += {"fixed": f"[{array.get('size')}]", "bounded": f"[<={array.get('max')}]",
                  "variable": f"[{array.get('min')}..{array.get('max')}]"}.get(kind, "[]")
    if node.get("optional"):
        label += " ?"
    if node.get("enum"):
        label += " (enum)"
    return label


def flatten(node: Dict[str, Any], prefix: str = "", depth: int = 0) -> List[str]:
    """Làm phẳng cây thành `đường.dẫn: kiểu` — dạng dễ đối chiếu nhất khi viết converter."""
    rows: List[str] = []
    for name, child in (node.get("fields") or {}).items():
        path = f"{prefix}.{name}" if prefix else name
        rows.append(f"{path}: {type_label(child)}")
        if depth < MAX_DEPTH:
            rows.extend(flatten(child, path, depth + 1))
    return rows


def print_report(rep: Dict[str, Any], show_fields: bool) -> None:
    st, enc = rep["statistics"], rep["encodings"]
    print(f"\n{'=' * 78}\nFILE  {rep['file']['path']}")
    print(f"      {rep['file']['size_bytes'] / 1e6:.2f} MB", end="")
    if rep["header"]:
        print(f" | profile={rep['header']['profile']!r} library={rep['header']['library']!r}", end="")
    print(f"\n      {st['message_count']} message · {st['channel_count']} topic · "
          f"{st['schema_count']} schema · {st['duration_s']} s")
    compression = "không rõ" if enc["compression"] is None else (", ".join(enc["compression"]) or "không nén")
    print(f"      message_encoding: {', '.join(enc['message']) or '—'}"
          f"   |   schema_encoding: {', '.join(enc['schema']) or '—'}"
          f"   |   nén: {compression}")
    print(f"      chế độ quét: {rep['scan']['mode']}")

    if rep["metadata_records"]:
        print(f"\n{'-' * 78}\nMETADATA RECORDS")
        for rec in rep["metadata_records"]:
            print(f"  [{rec['name']}]")
            for k, v in rec["metadata"].items():
                v = str(v)
                print(f"      {k:<24} = {v if len(v) <= 100 else v[:100] + '…'}")

    if rep["attachments"]:
        print(f"\n{'-' * 78}\nATTACHMENTS")
        for a in rep["attachments"]:
            print(f"  {a['name']:<40} {a['media_type']:<20} {a['size_bytes']} B")

    print(f"\n{'-' * 78}\nTOPICS  ({'~' } = nhịp ước lượng)")
    print(f"  {'topic':<38} {'#msg':>7} {'Hz':>9}  {'msg enc':<9} type")
    for t in rep["topics"]:
        rate = "-" if not t["rate_hz"] else (f"~{t['rate_hz']:.2f}" if t["rate_is_estimate"] else f"{t['rate_hz']:.2f}")
        print(f"  {t['topic']:<38} {t['message_count']:>7} {rate:>9}  "
              f"{t['message_encoding'] or '?':<9} {t['schema']['name'] or '(không có schema)'}")

    if show_fields:
        print(f"\n{'-' * 78}\nCẤU TRÚC MESSAGE")
        for schema in rep["schemas"]:
            print(f"\n  ▸ {schema['name']}   [{schema['encoding']}]"
                  f"   ← {', '.join(schema['used_by_topics']) or '(không topic nào dùng)'}")
            if "structure" in schema:
                for row in flatten(schema["structure"]):
                    print(f"      {row}")
            else:
                print(f"      (chưa parse được: {schema.get('warning')})")
        for t in rep["topics"]:
            if t["structure_source"] == "sampled:json":
                print(f"\n  ▸ {t['topic']}   [KHÔNG có schema — đoán từ "
                      f"{rep['scan']['samples_per_schemaless_topic']} message JSON đầu tiên]")
                for row in t["field_list"]:
                    print(f"      {row}")

    if rep["warnings"]:
        print(f"\n{'-' * 78}\nCẢNH BÁO")
        for w in rep["warnings"]:
            print(f"  ! {w}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Khảo sát cấu trúc một file .mcap lạ: topic, message type, cây field → JSON."
    )
    ap.add_argument("--mcap", required=True, help="đường dẫn file .mcap cần khảo sát")
    ap.add_argument("--out", default=None, help="file JSON đầu ra (mặc định: <mcap>.inspect.json)")
    ap.add_argument("--deep", action="store_true",
                    help="quét mọi message để đo nhịp/cỡ byte thật (chậm hơn với file lớn)")
    ap.add_argument("--samples", type=int, default=5,
                    help="số message đọc thử cho topic KHÔNG có schema (mặc định 5)")
    ap.add_argument("--no-sample", action="store_true",
                    help="tuyệt đối không đọc message nào, kể cả topic thiếu schema")
    ap.add_argument("--no-fields", action="store_true", help="không in cây field ra màn hình")
    args = ap.parse_args()

    # Kiểm tra 8 byte magic trước: đỡ ném traceback khó hiểu khi cầm nhầm file.
    if not os.path.isfile(args.mcap):
        sys.exit(f"✖ Không tìm thấy file: {args.mcap}")
    with open(args.mcap, "rb") as f:
        magic = f.read(len(MAGIC))
    if magic != MAGIC:
        sys.exit(f"✖ {args.mcap} không phải file MCAP "
                 f"({'file rỗng' if not magic else '8 byte đầu là ' + repr(magic)}).")

    rep = inspect(args.mcap, args.samples, args.deep, allow_sample=not args.no_sample)
    out = args.out or args.mcap + ".inspect.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print_report(rep, show_fields=not args.no_fields)
    print(f"✔ Đã ghi báo cáo: {os.path.abspath(out)} ({os.path.getsize(out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
