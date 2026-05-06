"""Captura de tela e overlays para o TextVision.

Componentes:
    - ScreenStream: captura contínua do desktop em thread (mss).
    - RegionSelector: janela transparente fullscreen para o usuário
      arrastar e selecionar uma região de interesse na tela.
    - CursorMagnifier: janela topmost que segue o mouse e mostra a
      região sob o cursor já com o pipeline de visão aplicado.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_dict(self) -> dict:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    def is_valid(self) -> bool:
        return self.width >= 8 and self.height >= 8


# --------------------------------------------------------------------------
# Stream contínuo da tela
# --------------------------------------------------------------------------

class ScreenStream:
    """Captura região da tela em uma thread dedicada usando mss.

    A região pode ser alterada em runtime via set_region(); quando None,
    captura o monitor inteiro (índice 1; mss.monitors[0] é a virtual
    desktop combinada).
    """

    def __init__(self, target_fps: int = 30) -> None:
        self.target_fps = target_fps
        self._region: Region | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        # Lista de monitores disponíveis (preenchida em start)
        self.monitors: list[dict] = []
        self.virtual: dict | None = None

    def start(self) -> bool:
        try:
            with mss.mss() as sct:
                # mss.monitors[0] = virtual desktop, [1..n] = monitores físicos
                self.monitors = list(sct.monitors)
                self.virtual = self.monitors[0]
                if self._region is None and len(self.monitors) > 1:
                    m = self.monitors[1]
                    self._region = Region(m["left"], m["top"], m["width"], m["height"])
        except Exception as exc:
            self._error = f"mss não pôde abrir o desktop: {exc}"
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        # mss precisa ser instanciada na própria thread
        period = 1.0 / max(1, self.target_fps)
        last = 0.0
        try:
            with mss.mss() as sct:
                while self._running:
                    if self._paused or self._region is None:
                        time.sleep(0.03)
                        continue
                    now = time.perf_counter()
                    delay = period - (now - last)
                    if delay > 0:
                        time.sleep(delay)
                    last = time.perf_counter()
                    try:
                        raw = sct.grab(self._region.as_dict())
                    except Exception:
                        time.sleep(0.05)
                        continue
                    arr = np.asarray(raw, dtype=np.uint8)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                    with self._lock:
                        self._frame = bgr
                    time.sleep(0.001)
        except Exception as exc:
            self._error = f"loop de captura falhou: {exc}"

    def grab_once(self, region: Region) -> np.ndarray | None:
        """Captura síncrona única — útil para o magnifier."""
        try:
            with mss.mss() as sct:
                raw = sct.grab(region.as_dict())
            arr = np.asarray(raw, dtype=np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            self._error = str(exc)
            return None

    def set_region(self, region: Region | None) -> None:
        self._region = region

    def get_region(self) -> Region | None:
        return self._region

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def pause(self, paused: bool) -> None:
        self._paused = paused

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def error(self) -> str | None:
        return self._error

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None


# --------------------------------------------------------------------------
# Seletor de região (overlay fullscreen)
# --------------------------------------------------------------------------

class RegionSelector:
    """Overlay transparente onde o usuário arrasta para selecionar uma
    região do desktop. Suporta múltiplos monitores via virtual desktop.

    Uso:
        sel = RegionSelector(parent_root, virtual_bbox)
        sel.run(on_done)   # on_done(Region | None)
    """

    BG = "#000000"
    LINE = "#9ec5ff"

    def __init__(self, parent: tk.Misc, virtual_bbox: dict, dpi_scale: float = 1.0) -> None:
        self.parent = parent
        self.bbox = virtual_bbox  # dimensões em pixels físicos (mss)
        self.dpi_scale = dpi_scale  # fator físico/lógico (>1 em HiDPI no Windows)
        self.top: tk.Toplevel | None = None
        self.canvas: tk.Canvas | None = None
        self._start: tuple[int, int] | None = None
        self._rect: int | None = None
        self._on_done = None
        self._result: Region | None = None

    def run(self, on_done) -> None:
        self._on_done = on_done
        self.top = tk.Toplevel(self.parent)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.top.attributes("-alpha", 0.30)
        self.top.configure(bg=self.BG)
        # Converter bbox físico (mss) → lógico (Tkinter) para posicionar a janela
        x = int(self.bbox["left"] / self.dpi_scale)
        y = int(self.bbox["top"] / self.dpi_scale)
        w = int(self.bbox["width"] / self.dpi_scale)
        h = int(self.bbox["height"] / self.dpi_scale)
        self.top.geometry(f"{w}x{h}+{x}+{y}")
        self.top.config(cursor="cross")
        self.canvas = tk.Canvas(self.top, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # dica
        self.canvas.create_text(
            w // 2, 36,
            text="Arraste para selecionar uma região  ·  Esc cancela",
            fill="#ffffff", font=("Segoe UI", 14),
        )
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Retorna "break" para impedir que o Escape propague para o bind_all
        # do root, que poderia fechar o aplicativo indevidamente.
        self.top.bind("<Escape>", lambda e: self._finish(None) or "break")
        self.top.focus_force()
        self.top.grab_set()

    def _canvas_offset(self) -> tuple[int, int]:
        """Offset do canvas em pixels lógicos (canto superior esquerdo do overlay)."""
        return int(self.bbox["left"] / self.dpi_scale), int(self.bbox["top"] / self.dpi_scale)

    def _on_press(self, event) -> None:
        self._start = (event.x_root, event.y_root)
        if self._rect is not None:
            self.canvas.delete(self._rect)
        ox, oy = self._canvas_offset()
        cx = event.x_root - ox
        cy = event.y_root - oy
        self._rect = self.canvas.create_rectangle(
            cx, cy, cx, cy, outline=self.LINE, width=2,
        )

    def _on_drag(self, event) -> None:
        if self._start is None or self._rect is None:
            return
        ox, oy = self._canvas_offset()
        cx0 = self._start[0] - ox
        cy0 = self._start[1] - oy
        cx1 = event.x_root - ox
        cy1 = event.y_root - oy
        self.canvas.coords(self._rect, cx0, cy0, cx1, cy1)

    def _on_release(self, event) -> None:
        if self._start is None:
            self._finish(None)
            return
        x0, y0 = self._start  # coords lógicas do Tkinter
        x1, y1 = event.x_root, event.y_root
        left, top = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Converter pixels lógicos → físicos para que o mss capture a área correta
        region = Region(
            int(left * self.dpi_scale),
            int(top * self.dpi_scale),
            max(1, int(width * self.dpi_scale)),
            max(1, int(height * self.dpi_scale)),
        )
        if not region.is_valid():
            self._finish(None)
        else:
            self._finish(region)

    def _finish(self, region: Region | None) -> None:
        try:
            if self.top is not None:
                self.top.grab_release()
                self.top.destroy()
        except Exception:
            pass
        self.top = None
        self.canvas = None
        if self._on_done:
            self._on_done(region)


# --------------------------------------------------------------------------
# Lupa flutuante que segue o cursor
# --------------------------------------------------------------------------

class CursorMagnifier:
    """Janela topmost que captura uma região ao redor do cursor e a
    exibe ampliada com o pipeline de visão aplicado.

    O caller injeta:
        - settings_provider(): retorna VisionSettings atual.
        - process_fn(frame, settings): retorna o frame processado.
    """

    CAPTURE_W = 320
    CAPTURE_H = 140
    WINDOW_W = 720
    WINDOW_H = 320
    OFFSET = 28
    UPDATE_MS = 33

    def __init__(self, parent: tk.Misc, settings_provider, process_fn) -> None:
        self.parent = parent
        self._settings_provider = settings_provider
        self._process_fn = process_fn
        self.top: tk.Toplevel | None = None
        self.label: tk.Label | None = None
        self._photo = None
        self._after_id: str | None = None
        self._last_size: tuple[int, int] = (0, 0)

    @property
    def is_open(self) -> bool:
        return self.top is not None

    def open(self) -> None:
        if self.is_open:
            return
        from PIL import ImageTk  # import tardio
        self._ImageTk = ImageTk
        self.top = tk.Toplevel(self.parent)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        try:
            self.top.attributes("-alpha", 0.97)
        except tk.TclError:
            pass
        self.top.configure(bg="#0e0e10")
        self.label = tk.Label(self.top, bg="#000", bd=0)
        self.label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.top.geometry(f"{self.WINDOW_W}x{self.WINDOW_H}+50+50")
        self.top.bind("<Escape>", lambda e: self.close())
        self._tick()

    def close(self) -> None:
        if self._after_id is not None:
            try:
                self.parent.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.top is not None:
            try:
                self.top.destroy()
            except Exception:
                pass
        self.top = None
        self.label = None
        self._photo = None

    def _tick(self) -> None:
        if not self.is_open:
            return
        try:
            self._render_once()
        except Exception:
            pass
        self._after_id = self.parent.after(self.UPDATE_MS, self._tick)

    def _render_once(self) -> None:
        from PIL import Image
        cx = self.parent.winfo_pointerx()
        cy = self.parent.winfo_pointery()
        cap = Region(
            left=cx - self.CAPTURE_W // 2,
            top=cy - self.CAPTURE_H // 2,
            width=self.CAPTURE_W,
            height=self.CAPTURE_H,
        )
        # Esconder a janela durante a captura para não nos vermos a nós mesmos.
        try:
            with mss.mss() as sct:
                raw = sct.grab(cap.as_dict())
        except Exception:
            return
        arr = np.asarray(raw, dtype=np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        settings = self._settings_provider()
        processed = self._process_fn(bgr, settings)
        # Ajusta para a janela
        h, w = processed.shape[:2]
        scale = min(self.WINDOW_W / w, self.WINDOW_H / h)
        if scale != 1.0:
            processed = cv2.resize(
                processed,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if self._photo is None or self._last_size != pil.size:
            self._photo = self._ImageTk.PhotoImage(pil)
            self._last_size = pil.size
        else:
            self._photo.paste(pil)
        if self.label is not None:
            self.label.configure(image=self._photo)
        # Posiciona a janela offset do cursor, evitando bordas
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        wx = cx + self.OFFSET
        wy = cy + self.OFFSET
        if wx + self.WINDOW_W > sw:
            wx = cx - self.OFFSET - self.WINDOW_W
        if wy + self.WINDOW_H > sh:
            wy = cy - self.OFFSET - self.WINDOW_H
        if self.top is not None:
            self.top.geometry(f"{self.WINDOW_W}x{self.WINDOW_H}+{wx}+{wy}")
