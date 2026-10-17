"""Vigila los efectos nuevos de Habbo y los publica en Discord mediante un WEBHOOK.

Pensado para GitHub Actions: cada ejecución hace UNA comprobación y termina.
  python monitor.py                 comprobación normal
  python monitor.py --ping          envía un mensaje de prueba al webhook
  python monitor.py --test-swf X    renderiza un .swf local y lo publica (prueba)
"""
import asyncio
import json
import logging
import mimetypes
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import aiohttp

from swf_images import render_effect

DOMAINS = [
    "https://sandbox.habbo.com/gamedata/external_variables/1",
    "https://habbo.com/gamedata/external_variables/1",
    "https://habbo.es/gamedata/external_variables/1",
    "https://habbo.com.br/gamedata/external_variables/1",
    "https://habbo.fi/gamedata/external_variables/1",
    "https://habbo.it/gamedata/external_variables/1",
    "https://habbo.de/gamedata/external_variables/1",
    "https://habbo.com.tr/gamedata/external_variables/1",
    "https://habbo.nl/gamedata/external_variables/1",
    "https://habbo.fr/gamedata/external_variables/1",
]

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
WEBHOOK_USERNAME = os.environ.get("WEBHOOK_USERNAME", "Habbo Efectos")
STATE_FILE = Path("state.json")
DOWNLOADS = Path("downloads")

MAX_FILES_PER_MESSAGE = 10          # límite de Discord
MAX_BYTES_PER_MESSAGE = 8 * 1024 * 1024  # margen bajo el límite de subida

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HabboEffectsBot/1.0)"}
CLIENT_URL_RE = re.compile(r"^\s*flash\.client\.url\s*=\s*(.+?)\s*$", re.MULTILINE)
VERSION_RE = re.compile(r"(PRODUCTION-[\w-]+)")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("habbo-monitor")


# ---------------------------------------------------------------- estado
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"urls": {}, "effects": [], "initialized": False}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ Habbo
async def fetch(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                log.info("HTTP %s en %s", r.status, url)
                return None
            return await r.read()
    except Exception as e:  # noqa: BLE001
        log.warning("Error descargando %s: %s", url, e)
        return None


async def get_client_url(session: aiohttp.ClientSession, domain_url: str) -> str | None:
    data = await fetch(session, domain_url)
    if not data:
        return None
    m = CLIENT_URL_RE.search(data.decode("utf-8", errors="replace"))
    if not m:
        return None
    url = m.group(1).strip()
    if url.startswith("//"):
        url = "https:" + url
    if not url.endswith("/"):
        url += "/"
    return url


def candidate_urls(base: str, filename: str) -> list[str]:
    """effectmap.xml y los .swf viven en /gordon/flash-assets-PRODUCTION-.../
    Probamos esa ruta primero y luego la que viene tal cual en flash.client.url."""
    urls = []
    if "/gordon/flash-assets-" not in base:
        urls.append(base.replace("/gordon/", "/gordon/flash-assets-", 1) + filename)
    urls.append(base + filename)
    return urls


async def download_asset(session, base: str, filename: str) -> bytes | None:
    for url in candidate_urls(base, filename):
        data = await fetch(session, url)
        if data:
            return data
    return None


# ---------------------------------------------------------------- Discord
async def post_webhook(session: aiohttp.ClientSession, content: str | None = None, files=()) -> None:
    """Envía un mensaje (con hasta 10 archivos) al webhook. Reintenta si hay rate limit."""
    if not WEBHOOK_URL:
        raise RuntimeError("Falta la variable DISCORD_WEBHOOK_URL")

    for _ in range(5):
        payload = {"username": WEBHOOK_USERNAME, "allowed_mentions": {"parse": []}}
        if content:
            payload["content"] = content
        form = aiohttp.FormData()
        form.add_field("payload_json", json.dumps(payload), content_type="application/json")
        for i, (name, data) in enumerate(files):
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            form.add_field(f"files[{i}]", data, filename=name, content_type=ctype)

        async with session.post(
            WEBHOOK_URL, data=form, params={"wait": "true"}, timeout=aiohttp.ClientTimeout(total=120)
        ) as r:
            if r.status == 429:
                try:
                    wait = float((await r.json()).get("retry_after", 2))
                except Exception:  # noqa: BLE001
                    wait = 2
                log.warning("Rate limit de Discord, esperando %.1fs", wait)
                await asyncio.sleep(wait + 0.5)
                continue
            if r.status >= 400:
                raise RuntimeError(f"Webhook respondió {r.status}: {(await r.text())[:300]}")
            return
    raise RuntimeError("Discord sigue limitando las peticiones; se reintentará en la próxima ejecución")


def split_batches(files: list[tuple[str, bytes]]) -> tuple[list[list[tuple[str, bytes]]], list[str]]:
    """Agrupa archivos respetando máx. 10 por mensaje y un tamaño total razonable."""
    batches, current, size, skipped = [], [], 0, []
    for name, data in files:
        if len(data) > MAX_BYTES_PER_MESSAGE:
            skipped.append(name)
            continue
        if current and (len(current) >= MAX_FILES_PER_MESSAGE or size + len(data) > MAX_BYTES_PER_MESSAGE):
            batches.append(current)
            current, size = [], 0
        current.append((name, data))
        size += len(data)
    if current:
        batches.append(current)
    return batches, skipped


async def announce_effect(session, lib: str, swf: bytes | None) -> None:
    content = f"👉 Se ha encontrado el siguiente efecto 👈\n**{lib}**"
    files: list[tuple[str, bytes]] = []

    if swf is None:
        content += f"\nNo se pudo descargar {lib}.swf."
    else:
        try:
            result = await asyncio.to_thread(render_effect, swf, lib)
            files = result.files
            if result.note:
                content += f"\n{result.note}"
        except Exception:  # noqa: BLE001
            log.exception("Error procesando %s.swf", lib)
            content += "\nNo se pudieron procesar las imágenes."

    batches, skipped = split_batches(files)
    if skipped:
        content += "\nArchivos omitidos por tamaño: " + ", ".join(skipped)

    if not batches:
        await post_webhook(session, content)
        return
    await post_webhook(session, content, batches[0])  # el texto va con el primer lote
    for batch in batches[1:]:
        await post_webhook(session, None, batch)


# --------------------------------------------------------------- ciclo principal
async def check_once() -> None:
    state = load_state()
    known = set(state["effects"])
    first_run = not state["initialized"]
    DOWNLOADS.mkdir(exist_ok=True)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        urls = await asyncio.gather(*(get_client_url(session, d) for d in DOMAINS))

        changed: dict[str, list[str]] = {}
        for domain, url in zip(DOMAINS, urls):
            if url and state["urls"].get(domain) != url:
                changed.setdefault(url, []).append(domain)

        if not changed:
            log.info("Sin cambios en flash.client.url")

        for base, domains in changed.items():
            log.info("Cambio detectado: %s (%s)", base, ", ".join(domains))
            data = await download_asset(session, base, "effectmap.xml")
            if not data:
                log.warning("No se pudo descargar effectmap.xml; se reintentará")
                continue  # no guardamos la URL: se reintenta en la próxima ejecución

            version = (VERSION_RE.search(base) or [None, "unknown"])[1]
            (DOWNLOADS / f"effectmap_{version}.xml").write_bytes(data)

            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                log.warning("effectmap.xml inválido en %s", base)
                continue

            libs = {e.get("lib") for e in root.iter("effect") if e.get("lib")}
            new_libs = sorted(libs - known)

            if first_run:
                known |= libs
            else:
                for lib in new_libs:
                    swf = await download_asset(session, base, f"{lib}.swf")
                    await announce_effect(session, lib, swf)
                    known.add(lib)                       # se guarda tras CADA aviso enviado,
                    state["effects"] = sorted(known)     # así un fallo no repite los anteriores
                    save_state(state)

            for d in domains:
                state["urls"][d] = base
            state["effects"] = sorted(known)
            save_state(state)

        if first_run:
            state["initialized"] = True
            save_state(state)
            log.info("Línea base creada con %d efectos (no se anuncian).", len(known))


async def ping() -> None:
    async with aiohttp.ClientSession() as session:
        await post_webhook(session, "Webhook configurado correctamente.")
    log.info("Mensaje de prueba enviado")


async def test_swf(path: str) -> None:
    lib = Path(path).stem
    async with aiohttp.ClientSession() as session:
        await announce_effect(session, lib, Path(path).read_bytes())


def main() -> int:
    try:
        if "--ping" in sys.argv:
            asyncio.run(ping())
        elif "--test-swf" in sys.argv:
            asyncio.run(test_swf(sys.argv[sys.argv.index("--test-swf") + 1]))
        else:
            asyncio.run(check_once())
        return 0
    except Exception:  # noqa: BLE001
        log.exception("La ejecución falló")
        return 1


if __name__ == "__main__":
    sys.exit(main())
