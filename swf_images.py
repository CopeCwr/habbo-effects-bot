"""Extractor de imágenes de efectos de Habbo (.swf) sin dependencias externas de Flash.

Estructura real de estos .swf (verificada con Spin, LeavesBlown y Mystic):
  * Tags DefineBitsLossless2 (36) con los sprites (ARGB con alfa premultiplicado).
  * SymbolClass (76): nombre -> id de bitmap. VARIOS nombres pueden compartir el
    mismo bitmap (Flash deduplica imágenes idénticas).
  * DefineBinaryData (87) con dos XML:
      <Lib>_manifest  -> assets (offset = punto de registro), aliases (link + fliph)
      <Lib>_animation -> sprites (capa, ink, member, dz) y frames (línea de tiempo)
  * Nombre de asset: {prefijo}_{member}_{direccion}_{frame}
        p. ej. h_std_fx242_1_1_2_0  -> member=std_fx242_1_1, dir=2, frame=0
"""
from __future__ import annotations

import io
import lzma
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

TAG_LOSSLESS, TAG_JPEG2, TAG_JPEG3, TAG_LOSSLESS2 = 20, 21, 35, 36
TAG_SYMBOLCLASS, TAG_BINARY = 76, 87

SCALE = 1                    # tamaño real (1 px del sprite = 1 px de la imagen)
BG = (43, 45, 49)            # fondo oscuro tipo Discord
TICK_MS = 83                 # duración aproximada de un "tick" de animación (~12 fps)
ADDITIVE_INKS = {"33", "34"}  # Director: Add Pin / Add
DIR_PREFERENCE = [2, 0, 4, 6, 1, 3, 5, 7]
ASSET_RE = re.compile(r"^(?P<pre>[a-z]+)_(?P<member>.+)_(?P<dir>\d+)_(?P<frame>\d+)$")


# ----------------------------------------------------------------- SWF bajo nivel
def _decompress(data: bytes) -> bytes:
    sig = data[:3]
    if sig == b"FWS":
        return data[8:]
    if sig == b"CWS":
        return zlib.decompress(data[8:])
    if sig == b"ZWS":
        props, stream = data[12:17], data[17:]
        return lzma.decompress(props + struct.pack("<q", -1) + stream, format=lzma.FORMAT_ALONE)
    raise ValueError("No es un archivo SWF válido")


def _iter_tags(body: bytes):
    nbits = body[0] >> 3
    pos = (5 + 4 * nbits + 7) // 8 + 4  # RECT + framerate(2) + framecount(2)
    while pos + 2 <= len(body):
        (h,) = struct.unpack_from("<H", body, pos)
        pos += 2
        code, length = h >> 6, h & 0x3F
        if length == 0x3F:
            (length,) = struct.unpack_from("<I", body, pos)
            pos += 4
        yield code, body[pos : pos + length]
        pos += length
        if code == 0:
            break


def _lossless(tag: int, data: bytes) -> tuple[int, Image.Image]:
    cid, fmt, w, h = struct.unpack_from("<HBHH", data, 0)
    pos, ncolors = 7, 0
    if fmt == 3:
        ncolors, pos = data[pos] + 1, pos + 1
    raw = zlib.decompress(data[pos:])

    if fmt == 5:  # ARGB 32 bits
        arr = np.frombuffer(raw, np.uint8, count=w * h * 4).reshape(h, w, 4)
        if tag == TAG_LOSSLESS2:
            a = arr[..., 0].astype(np.uint16)
            rgb = arr[..., 1:].astype(np.uint16)
            safe = np.maximum(a, 1)[..., None]
            rgb = np.where(a[..., None] > 0, np.minimum(255, (rgb * 255 + safe // 2) // safe), 0)
            out = np.dstack([rgb.astype(np.uint8), a.astype(np.uint8)])
        else:  # Lossless sin alfa: xRGB
            out = np.dstack([arr[..., 1:], np.full((h, w), 255, np.uint8)])
    elif fmt == 3:  # paleta
        esz = 4 if tag == TAG_LOSSLESS2 else 3
        pal = np.frombuffer(raw, np.uint8, count=ncolors * esz).reshape(ncolors, esz)
        if esz == 3:
            pal = np.hstack([pal, np.full((ncolors, 1), 255, np.uint8)])
        stride = (w + 3) & ~3
        idx = np.frombuffer(raw, np.uint8, offset=ncolors * esz, count=stride * h).reshape(h, stride)[:, :w]
        out = pal[np.minimum(idx, ncolors - 1)]
    else:
        raise ValueError(f"Formato lossless no soportado: {fmt}")
    return cid, Image.fromarray(out, "RGBA")


def _jpeg(tag: int, data: bytes) -> tuple[int, Image.Image]:
    (cid,) = struct.unpack_from("<H", data, 0)
    if tag == TAG_JPEG2:
        return cid, Image.open(io.BytesIO(data[2:])).convert("RGBA")
    (alpha_off,) = struct.unpack_from("<I", data, 2)
    img = Image.open(io.BytesIO(data[6 : 6 + alpha_off])).convert("RGBA")
    alpha = zlib.decompress(data[6 + alpha_off :])
    if len(alpha) == img.width * img.height:
        img.putalpha(Image.frombytes("L", img.size, alpha))
    return cid, img


# ------------------------------------------------------------------- modelo
@dataclass
class Effect:
    lib: str = ""
    images: dict[str, Image.Image] = field(default_factory=dict)      # nombre de asset -> imagen
    offsets: dict[str, tuple[int, int]] = field(default_factory=dict)  # nombre -> punto de registro
    aliases: dict[str, tuple[str, bool]] = field(default_factory=dict)  # nombre -> (destino, fliph)
    sprites: dict[str, dict] = field(default_factory=dict)             # id -> {ink, member, dz{dir}}
    frames: list[tuple[int, list[tuple[str, int]]]] = field(default_factory=list)
    declared_assets: int = 0


def _parse_xml(raw: bytes) -> ET.Element | None:
    text = raw[4:].decode("utf-8", errors="replace").replace("\x00", "").strip()
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        return None


def parse_swf(swf_bytes: bytes) -> Effect:
    body = _decompress(swf_bytes)
    bitmaps: dict[int, Image.Image] = {}
    symbols: dict[int, list[str]] = {}
    binaries: dict[int, bytes] = {}

    for code, data in _iter_tags(body):
        try:
            if code in (TAG_LOSSLESS, TAG_LOSSLESS2):
                cid, img = _lossless(code, data)
                bitmaps[cid] = img
            elif code in (TAG_JPEG2, TAG_JPEG3):
                cid, img = _jpeg(code, data)
                bitmaps[cid] = img
            elif code == TAG_SYMBOLCLASS:
                (n,) = struct.unpack_from("<H", data, 0)
                pos = 2
                for _ in range(n):
                    (cid,) = struct.unpack_from("<H", data, pos)
                    end = data.index(b"\x00", pos + 2)
                    symbols.setdefault(cid, []).append(data[pos + 2 : end].decode("utf-8", "replace"))
                    pos = end + 1
            elif code == TAG_BINARY:
                (cid,) = struct.unpack_from("<H", data, 0)
                binaries[cid] = data
        except Exception:  # noqa: BLE001  una pieza rota no debe tumbar el resto
            continue

    fx = Effect(lib=(symbols.get(0) or [""])[0])
    prefix = fx.lib + "_"

    # Imágenes: UNA entrada por nombre (varios nombres pueden compartir bitmap)
    for cid, names in symbols.items():
        if cid in bitmaps:
            for name in names:
                fx.images[name[len(prefix):] if name.startswith(prefix) else name] = bitmaps[cid]

    # XML de manifest y animation
    for cid, names in symbols.items():
        if cid not in binaries:
            continue
        root = _parse_xml(binaries[cid])
        if root is None:
            continue
        if root.tag == "manifest":
            for a in root.iter("asset"):
                for p in a.iter("param"):
                    if p.get("key") == "offset":
                        x, y = (int(v) for v in p.get("value").split(","))
                        fx.offsets[a.get("name")] = (x, y)
                fx.declared_assets += 1
            for al in root.iter("alias"):
                fx.aliases[al.get("name")] = (al.get("link"), al.get("fliph") == "1")
        elif root.tag == "animation":
            for s in root.findall("sprite"):
                fx.sprites[s.get("id")] = {
                    "ink": s.get("ink"),
                    "member": s.get("member"),
                    "dz": {int(d.get("id")): int(d.get("dz", 0)) for d in s.findall("direction") if d.get("id")},
                }
            for fr in root.findall("frame"):
                items = [(f.get("id"), int(f.get("frame", 0))) for f in fr.findall("fx")]
                fx.frames.append((int(fr.get("repeats", 1)), items))
    return fx


def extract_images(swf_bytes: bytes) -> list[tuple[str, Image.Image]]:
    """Compatibilidad: lista (nombre, imagen)."""
    return list(parse_swf(swf_bytes).images.items())


# --------------------------------------------------------------- utilidades
def _resolve(fx: Effect, name: str):
    """Devuelve (imagen, (regx, regy)) resolviendo alias y volteado horizontal."""
    flip = False
    seen = 0
    while name not in fx.images and name in fx.aliases and seen < 5:
        name, f = fx.aliases[name]
        flip ^= f
        seen += 1
    img = fx.images.get(name)
    if img is None:
        return None
    ox, oy = fx.offsets.get(name, (0, 0))
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        ox = img.width - ox
    return img, (ox, oy)


def _scaled(img: Image.Image, scale: int = SCALE) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def _blend_into(canvas: Image.Image, img: Image.Image, pos: tuple[int, int], additive: bool) -> None:
    """Pega `img` (RGBA) sobre `canvas` (RGB). Con tinta aditiva el negro desaparece."""
    if not additive:
        canvas.paste(img, pos, img)
        return
    x, y = pos
    box = (max(x, 0), max(y, 0), min(x + img.width, canvas.width), min(y + img.height, canvas.height))
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    src = np.asarray(img, np.float32)[box[1] - y : box[3] - y, box[0] - x : box[2] - x]
    dst = np.asarray(canvas.crop(box), np.float32)
    out = np.minimum(255, dst + src[..., :3] * (src[..., 3:] / 255.0))
    canvas.paste(Image.fromarray(out.astype(np.uint8), "RGB"), box[:2])


def _member_inks(fx: "Effect") -> dict[str, bool]:
    return {s["member"]: s["ink"] in ADDITIVE_INKS for s in fx.sprites.values() if s["member"]}


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# --------------------------------------------------------------- hojas de sprites
def build_sheets(fx: Effect, max_height: int = 2200, max_width: int = 1500) -> list[tuple[str, bytes]]:
    """Una sección por (member, dirección); los frames van en orden numérico y alineados
    por su punto de registro, de modo que al leerlos de izquierda a derecha se ve el movimiento."""
    groups: dict[tuple[str, str, int], list[tuple[int, str]]] = {}
    for name in fx.images:
        m = ASSET_RE.match(name)
        key, frame = ((m["pre"], m["member"], int(m["dir"])), int(m["frame"])) if m else (("", name, 0), 0)
        groups.setdefault(key, []).append((frame, name))

    pad, label_h, min_w = 4, 14, 330
    inks = _member_inks(fx)
    blocks: list[Image.Image] = []
    for (pre, member, direction), items in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        items.sort()
        info = []
        for frame, name in items:
            img = fx.images[name]
            ox, oy = fx.offsets.get(name, (0, 0))
            info.append((frame, img, ox, oy))
        # caja común alrededor del punto de registro
        left = max(ox for _, _, ox, _ in info)
        right = max(i.width - ox for _, i, ox, _ in info)
        top = max(oy for _, _, _, oy in info)
        bottom = max(i.height - oy for _, i, _, oy in info)
        cw, ch = (left + right) * SCALE + pad, (top + bottom) * SCALE + pad
        cols = max(1, (max_width - pad) // cw)
        rows = -(-len(info) // cols)

        block = Image.new("RGB", (max(min_w, min(len(info), cols) * cw + pad), label_h + rows * (ch + label_h) + pad), BG)
        d = ImageDraw.Draw(block)
        d.text((pad, 1), f"{pre}_{member}  dir {direction}  ({len(info)} frame{'s' if len(info) != 1 else ''})", fill=(210, 210, 210))
        for n, (frame, img, ox, oy) in enumerate(info):
            r, c = divmod(n, cols)
            x0 = pad + c * cw
            y0 = label_h + r * (ch + label_h)
            d.text((x0, y0), str(frame), fill=(120, 120, 120))
            s = _scaled(img)
            _blend_into(block, s, (x0 + (left - ox) * SCALE, y0 + label_h + (top - oy) * SCALE), inks.get(member, False))
        blocks.append(block)

    # empaquetar bloques en PNGs sin pasarse de alto
    sheets: list[tuple[str, bytes]] = []
    current: list[Image.Image] = []
    height = 0

    def flush():
        nonlocal current, height
        if not current:
            return
        w = max(b.width for b in current)
        sheet = Image.new("RGB", (w, sum(b.height for b in current)), BG)
        y = 0
        for b in current:
            sheet.paste(b, (0, y))
            y += b.height
        sheets.append((f"{fx.lib}_sprites_{len(sheets) + 1}.png", _to_png(sheet)))
        current, height = [], 0

    for b in blocks:
        if current and height + b.height > max_height:
            flush()
        current.append(b)
        height += b.height
    flush()
    return sheets


# --------------------------------------------------------------- vista animada
def _pick_direction(fx: Effect) -> int | None:
    for d in DIR_PREFERENCE:
        for spr in fx.sprites.values():
            if spr["member"] and _resolve(fx, f"h_{spr['member']}_{d}_0"):
                return d
    return None


def build_preview(fx: Effect) -> list[tuple[str, bytes]]:
    """Animación completa con FONDO TRANSPARENTE, componiendo las capas según la línea de tiempo.

    Devuelve dos archivos:
      * GIF: transparencia de 1 bit (limitación del formato; los bordes suaves se recortan).
      * APNG (.png animado): transparencia completa, conserva los degradados del brillo.
    Las capas con tinta aditiva no tienen alfa propio (el negro "desaparece" al sumarse),
    así que su brillo se convierte en opacidad.
    """
    if not fx.images or not fx.frames or not fx.sprites:
        return []
    direction = _pick_direction(fx)
    if direction is None:
        return []

    # 1) colocar cada capa de cada frame y calcular la caja total
    timeline = []
    x0 = y0 = 10**6
    x1 = y1 = -(10**6)
    for repeats, items in fx.frames:
        layers = []
        for sid, fidx in items:
            spr = fx.sprites.get(sid)
            if not spr or not spr["member"]:
                continue
            res = _resolve(fx, f"h_{spr['member']}_{direction}_{fidx}")
            if not res:
                continue
            img, (ox, oy) = res
            layers.append((spr["dz"].get(direction, 0), spr["ink"] in ADDITIVE_INKS, img, -ox, -oy))
            x0, y0 = min(x0, -ox), min(y0, -oy)
            x1, y1 = max(x1, -ox + img.width), max(y1, -oy + img.height)
        timeline.append((repeats, sorted(layers, key=lambda l: l[0])))
    if x1 <= x0:
        return []

    # 2) componer en RGBA (color premultiplicado + alfa acumulado)
    W, H = x1 - x0, y1 - y0
    rgba_frames: list[np.ndarray] = []
    durations: list[int] = []
    for repeats, layers in timeline:
        C = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W, 1), np.float32)
        for _, additive, img, lx, ly in layers:
            src = np.asarray(img, np.float32) / 255.0
            a = src[..., 3:]
            sc = src[..., :3] * a  # color premultiplicado de la capa
            ys, xs = ly - y0, lx - x0
            c_reg = C[ys : ys + img.height, xs : xs + img.width]
            a_reg = A[ys : ys + img.height, xs : xs + img.width]
            if additive:
                a_reg[:] = np.minimum(1.0, a_reg + sc.max(axis=2, keepdims=True))
                c_reg[:] = np.minimum(a_reg, c_reg + sc)
            else:
                c_reg[:] = sc + c_reg * (1 - a)
                a_reg[:] = a + a_reg * (1 - a)
        rgb = np.where(A > 0, C / np.maximum(A, 1e-6), 0.0)
        out = np.dstack([np.clip(rgb, 0, 1) * 255, A * 255]).round().astype(np.uint8)
        rgba_frames.append(out)
        durations.append(max(1, repeats) * TICK_MS)

    def upscale(arr: np.ndarray) -> Image.Image:
        im = Image.fromarray(arr, "RGBA")
        return im if SCALE == 1 else im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)

    files: list[tuple[str, bytes]] = []

    # 3a) GIF con transparencia de 1 bit y paleta global (evita parpadeos de color)
    scaled = [np.asarray(upscale(f)) for f in rgba_frames]
    opaque = np.concatenate([f[f[..., 3] >= 128][:, :3] for f in scaled])
    gif_frames = []
    if len(opaque):
        pal = Image.fromarray(opaque.reshape(1, -1, 3), "RGB").quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE)
        palette = pal.getpalette()[:765]
        palette += [0] * (768 - len(palette))
        for f in scaled:
            idx = np.array(Image.fromarray(f[..., :3], "RGB").quantize(palette=pal, dither=Image.NONE))
            idx[f[..., 3] < 128] = 255  # índice reservado para transparente
            p = Image.fromarray(idx, "P")
            p.putpalette(palette)
            gif_frames.append(p)
        buf = io.BytesIO()
        gif_frames[0].save(buf, "GIF", save_all=True, append_images=gif_frames[1:], duration=durations,
                           loop=0, transparency=255, disposal=2, optimize=False)
        files.append((f"{fx.lib}_animacion.gif", buf.getvalue()))

    # 3b) APNG con transparencia completa
    pngs = [upscale(f) for f in rgba_frames]
    buf = io.BytesIO()
    pngs[0].save(buf, "PNG", save_all=True, append_images=pngs[1:], duration=durations, loop=0, disposal=1)
    files.append((f"{fx.lib}_animacion_transparente.png", buf.getvalue()))
    return files


# --------------------------------------------------------------- API para el bot
@dataclass
class Rendered:
    files: list[tuple[str, bytes]]
    note: str | None = None  # mensaje informativo (p. ej. "sin imágenes")


def render_effect(swf_bytes: bytes, lib: str) -> Rendered:
    fx = parse_swf(swf_bytes)
    fx.lib = fx.lib or lib
    if not fx.images:
        extra = f" (solo contiene datos de animación: {len(fx.frames)} frames)" if fx.frames else ""
        return Rendered([], f"El .swf de **{lib}** no contiene imágenes{extra}.")
    files = build_sheets(fx)
    files.extend(build_preview(fx))
    missing = fx.declared_assets - len([n for n in fx.images if n in fx.offsets])
    note = f"Aviso: faltan {missing} imágenes declaradas en el manifest." if missing > 0 else None
    return Rendered(files, note)
