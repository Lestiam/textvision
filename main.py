"""TextVision - Lupa digital com OCR e leitura em voz alta.

Atalhos:
    + / -        zoom
    F            alterna filtro
    C            CLAHE
    [ / ]        brilho
    , / .        contraste
    O            OCR
    L            ler em voz alta
    S            alterna fonte (câmera ↔ tela)
    R            selecionar região (modo tela)
    M            lupa flutuante (segue o cursor)
    P            pausar / retomar fonte
    H            congelar / descongelar
    Esc          sair
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk, font as tkfont, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from ocr_engine import OCREngine, OCRResult, TTSEngine
from screen_capture import CursorMagnifier, RegionSelector, Region, ScreenStream
from vision import (
    FILTER_LABELS,
    VisionSettings,
    cycle_filter,
    process_for_ocr,
    process_frame,
)


CAMERA_INDEX = 0
TARGET_FPS = 30
FRAME_INTERVAL_MS = max(1, int(1000 / TARGET_FPS))
DISPLAY_MAX_W = 1100
DISPLAY_MAX_H = 620

# Paleta minimalista — dark com um único tom de destaque suave
C_BG          = "#0e0e10"
C_SURFACE     = "#16161a"
C_SURFACE_HI  = "#1d1d22"
C_BORDER      = "#26262e"
C_TEXT        = "#ececef"
C_TEXT_DIM    = "#9a9aa3"
C_TEXT_MUTED  = "#5a5a63"
C_ACCENT      = "#9ec5ff"
C_ACCENT_DIM  = "#3e5a85"
C_WARN        = "#f3b06a"


# --------------------------------------------------------------------------
# Captura de câmera em thread dedicada
# --------------------------------------------------------------------------

class CameraStream:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap or not self.cap.isOpened():
            self._error = "Não foi possível abrir a câmera."
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        assert self.cap is not None
        while self._running:
            if self._paused:
                time.sleep(0.03)
                continue
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(0.001)

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
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


# --------------------------------------------------------------------------
# Aplicação
# --------------------------------------------------------------------------

class TextVisionApp:
    SOURCE_CAMERA = "camera"
    SOURCE_SCREEN = "screen"

    LANG_CHOICES = ("por+eng", "por", "eng")

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TextVision")
        self.root.configure(bg=C_BG)
        self.root.geometry("1320x840")
        self.root.minsize(1120, 740)

        self._closing = False
        self.settings = VisionSettings()
        self.frozen = False
        self.frozen_frame: np.ndarray | None = None
        self.last_processed: np.ndarray | None = None
        self.last_source_frame: np.ndarray | None = None
        self.last_ocr_text: str = ""
        self.fps_ema: float = 0.0
        self.latency_ema: float = 0.0
        self._last_tick: float = time.time()
        self._photo_ref: ImageTk.PhotoImage | None = None
        self._photo_size: tuple[int, int] = (0, 0)
        self._ocr_running = False

        # Fontes
        self.camera = CameraStream(CAMERA_INDEX)
        self.screen = ScreenStream(target_fps=TARGET_FPS)
        self.source = self.SOURCE_CAMERA

        self.ocr = OCREngine(requested_languages="por+eng")
        self.tts = TTSEngine()
        self.magnifier: CursorMagnifier | None = None

        self._setup_fonts()
        self._setup_styles()
        self._build_ui()
        self._bind_keys()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if not self.camera.start():
            self._set_status(self.camera.error or "Câmera indisponível.")
            messagebox.showwarning(
                "Câmera indisponível",
                (self.camera.error or "Sem câmera detectada.")
                + "\n\nVocê ainda pode usar o modo Tela.",
            )

        # Inicializa screen stream sob demanda; já popula self.screen.virtual
        self.screen.start()

        self._refresh_lang_status()
        self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    # ---------- Tipografia / Tema ----------

    def _setup_fonts(self) -> None:
        ui = self._first_available(
            ["Segoe UI Variable", "Segoe UI", "Inter", "Helvetica Neue", "Arial"]
        )
        mono = self._first_available(["Cascadia Code", "Consolas", "Courier New"])
        # Fonte de ícones nativa do Windows. Fallback para emoji ou texto.
        icon = self._first_available(
            ["Segoe Fluent Icons", "Segoe MDL2 Assets", "Segoe UI Symbol", ui]
        )
        self.f_icon_name = icon
        self.f_brand   = (ui, 19, "normal")
        self.f_subtle  = (ui, 10)
        self.f_section = (ui, 9, "bold")
        self.f_value   = (ui, 18)
        self.f_unit    = (ui, 10)
        self.f_icon    = (icon, 16)
        self.f_button  = (ui, 11)
        self.f_kbd     = (mono, 9)
        self.f_status  = (ui, 9)
        self.f_ocr     = (ui, 14)
        self.f_ocr_hint = (ui, 11)

    @staticmethod
    def _first_available(names) -> str:
        avail = set(tkfont.families())
        for n in names:
            if n in avail:
                return n
        return names[-1]

    def _setup_styles(self) -> None:
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(
            "Mini.Horizontal.TScale",
            background=C_BG, troughcolor=C_BORDER, bordercolor=C_BG,
            lightcolor=C_ACCENT, darkcolor=C_ACCENT,
            sliderthickness=14, sliderlength=14,
        )
        s.map(
            "Mini.Horizontal.TScale",
            background=[("active", C_BG)],
            troughcolor=[("active", C_BORDER)],
        )

    # ---------- UI ----------

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=C_BG)
        header.pack(fill=tk.X, padx=28, pady=(20, 8))
        tk.Label(header, text="TextVision", bg=C_BG, fg=C_TEXT,
                 font=self.f_brand).pack(side=tk.LEFT)
        tk.Label(header, text="lupa digital · OCR · leitura por voz",
                 bg=C_BG, fg=C_TEXT_MUTED, font=self.f_subtle
                 ).pack(side=tk.LEFT, padx=(14, 0), pady=(8, 0))

        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=28)

        sidebar = tk.Frame(body, bg=C_BG, width=270)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(24, 0))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        video_wrap = tk.Frame(
            body, bg=C_SURFACE,
            highlightthickness=1, highlightbackground=C_BORDER,
        )
        video_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(video_wrap, bg="#000", bd=0)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        status_bar = tk.Frame(self.root, bg=C_BG)
        status_bar.pack(fill=tk.X, padx=28, pady=(10, 6))
        self.status_var = tk.StringVar(value="Iniciando…")
        tk.Label(status_bar, textvariable=self.status_var, bg=C_BG, fg=C_TEXT_DIM,
                 font=self.f_status, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._build_ocr_panel()

    def _build_sidebar(self, parent: tk.Frame) -> None:
        # ---- Fonte (switch com ícones) ----
        self._section_header(parent, "FONTE", "S")
        self._build_source_switch(parent)
        self._spacer(parent, 8)
        self._action_row(parent, "Selecionar região", "R", self._select_region)
        self._action_row(parent, "Lupa do cursor",    "M", self._toggle_magnifier)

        self._divider(parent)

        # ---- Sliders ----
        self.zoom_var = tk.DoubleVar(value=self.settings.zoom)
        self._zoom_value = self._slider_block(
            parent, "ZOOM", self.zoom_var, 1.0, 10.0,
            lambda v: self._set_zoom(float(v)),
            formatter=lambda: f"{self.settings.zoom:.1f}",
            unit="×",
        )
        self.brightness_var = tk.IntVar(value=self.settings.brightness)
        self._brightness_value = self._slider_block(
            parent, "BRILHO", self.brightness_var, -100, 100,
            lambda v: self._set_brightness(int(float(v))),
            formatter=lambda: f"{self.settings.brightness:+d}",
            unit="",
        )
        self.contrast_var = tk.DoubleVar(value=self.settings.contrast)
        self._contrast_value = self._slider_block(
            parent, "CONTRASTE", self.contrast_var, 0.5, 3.0,
            lambda v: self._set_contrast(float(v)),
            formatter=lambda: f"{self.settings.contrast:.1f}",
            unit="",
        )

        self._divider(parent)

        # ---- Filtro / CLAHE ----
        self._section_header(parent, "FILTRO", "F")
        self.filter_var = tk.StringVar(value=FILTER_LABELS[self.settings.filter_mode])
        self._filter_btn = self._button_pill(parent, self.filter_var, self._cycle_filter)

        self._spacer(parent, 14)

        self.clahe_var = tk.BooleanVar(value=self.settings.clahe_enabled)
        self._clahe_row, self._clahe_dot = self._toggle_row(
            parent, "CLAHE", "C", self.clahe_var, self._toggle_clahe_action
        )

        self._divider(parent)

        # ---- OCR ----
        self._section_header(parent, "OCR")
        self.lang_var = tk.StringVar(value="por+eng")
        self._lang_btn = self._button_pill(parent, self.lang_var, self._cycle_language)
        self._spacer(parent, 6)
        self._action_row(parent, "Reconhecer texto", "O", self._do_ocr)
        self._action_row(parent, "Ler em voz alta",  "L", self._do_speak)

        self._divider(parent)

        # ---- Câmera ----
        self._section_header(parent, "CÂMERA")
        self._action_row(parent, "Pausar fonte",     "P", self._toggle_pause)
        self._action_row(parent, "Congelar quadro",  "H", self._toggle_freeze)

    def _build_ocr_panel(self) -> None:
        wrap = tk.Frame(self.root, bg=C_BG)
        wrap.pack(fill=tk.X, padx=28, pady=(0, 20))
        inner = tk.Frame(wrap, bg=C_SURFACE,
                         highlightthickness=1, highlightbackground=C_BORDER)
        inner.pack(fill=tk.X)
        head = tk.Frame(inner, bg=C_SURFACE)
        head.pack(fill=tk.X, padx=18, pady=(12, 0))
        tk.Label(head, text="TEXTO RECONHECIDO", bg=C_SURFACE, fg=C_TEXT_MUTED,
                 font=self.f_section).pack(side=tk.LEFT)
        self.ocr_meta_var = tk.StringVar(value="")
        tk.Label(head, textvariable=self.ocr_meta_var, bg=C_SURFACE,
                 fg=C_TEXT_MUTED, font=self.f_section).pack(side=tk.RIGHT)
        self.ocr_text = tk.Text(
            inner, height=3, bg=C_SURFACE, fg=C_TEXT,
            insertbackground=C_TEXT, selectbackground=C_ACCENT_DIM,
            font=self.f_ocr, wrap=tk.WORD, padx=18, pady=10,
            relief=tk.FLAT, bd=0, highlightthickness=0,
        )
        self.ocr_text.pack(fill=tk.X, expand=True, pady=(2, 8))
        self._set_ocr_placeholder()

    # ---------- Switch de fonte (Câmera / Tela) ----------

    def _build_source_switch(self, parent: tk.Frame) -> None:
        # Glyphs do Segoe Fluent Icons / MDL2 Assets:
        #   E722 = Camera   ·   E7F4 = TVMonitor (PC/screen)
        # Se a fonte de ícones não for nenhum desses símbolos (fallback
        # para Segoe UI Symbol / family genérica), usamos texto curto.
        has_mdl2 = self.f_icon_name in ("Segoe Fluent Icons", "Segoe MDL2 Assets")
        cam_glyph = "" if has_mdl2 else "📷"
        scr_glyph = "" if has_mdl2 else "🖥"

        container = tk.Frame(
            parent, bg=C_SURFACE,
            highlightthickness=1, highlightbackground=C_BORDER,
        )
        container.pack(fill=tk.X)

        # Segmento Câmera
        self._cam_seg = tk.Frame(container, bg=C_SURFACE_HI, cursor="hand2")
        self._cam_seg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cam_inner = tk.Frame(self._cam_seg, bg=C_SURFACE_HI)
        cam_inner.pack(padx=6, pady=10)
        self._cam_icon = tk.Label(
            cam_inner, text=cam_glyph, font=self.f_icon,
            bg=C_SURFACE_HI, fg=C_ACCENT,
        )
        self._cam_icon.pack(side=tk.LEFT, padx=(8, 6))
        self._cam_label = tk.Label(
            cam_inner, text="Câmera", font=self.f_button,
            bg=C_SURFACE_HI, fg=C_TEXT,
        )
        self._cam_label.pack(side=tk.LEFT, padx=(0, 8))

        # Segmento Tela
        self._scr_seg = tk.Frame(container, bg=C_SURFACE, cursor="hand2")
        self._scr_seg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr_inner = tk.Frame(self._scr_seg, bg=C_SURFACE)
        scr_inner.pack(padx=6, pady=10)
        self._scr_icon = tk.Label(
            scr_inner, text=scr_glyph, font=self.f_icon,
            bg=C_SURFACE, fg=C_TEXT_MUTED,
        )
        self._scr_icon.pack(side=tk.LEFT, padx=(8, 6))
        self._scr_label = tk.Label(
            scr_inner, text="Tela", font=self.f_button,
            bg=C_SURFACE, fg=C_TEXT_MUTED,
        )
        self._scr_label.pack(side=tk.LEFT, padx=(0, 8))

        # Bind: clique em qualquer parte do segmento alterna a fonte
        for w in (self._cam_seg, cam_inner, self._cam_icon, self._cam_label):
            w.bind("<Button-1>", lambda e: self._set_source(self.SOURCE_CAMERA))
        for w in (self._scr_seg, scr_inner, self._scr_icon, self._scr_label):
            w.bind("<Button-1>", lambda e: self._set_source(self.SOURCE_SCREEN))

        self._cam_widgets = (self._cam_seg, cam_inner)
        self._scr_widgets = (self._scr_seg, scr_inner)

    def _refresh_source_switch(self) -> None:
        on_cam = self.source == self.SOURCE_CAMERA
        cam_bg = C_SURFACE_HI if on_cam else C_SURFACE
        scr_bg = C_SURFACE_HI if not on_cam else C_SURFACE
        cam_icon_fg = C_ACCENT if on_cam else C_TEXT_MUTED
        scr_icon_fg = C_ACCENT if not on_cam else C_TEXT_MUTED
        cam_text_fg = C_TEXT if on_cam else C_TEXT_MUTED
        scr_text_fg = C_TEXT if not on_cam else C_TEXT_MUTED
        for w in self._cam_widgets:
            w.configure(bg=cam_bg)
        self._cam_icon.configure(bg=cam_bg, fg=cam_icon_fg)
        self._cam_label.configure(bg=cam_bg, fg=cam_text_fg)
        for w in self._scr_widgets:
            w.configure(bg=scr_bg)
        self._scr_icon.configure(bg=scr_bg, fg=scr_icon_fg)
        self._scr_label.configure(bg=scr_bg, fg=scr_text_fg)

    # ---------- helpers de widgets ----------

    def _section_header(self, parent: tk.Frame, label: str, key: str | None = None) -> None:
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(row, text=label, bg=C_BG, fg=C_TEXT_DIM,
                 font=self.f_section).pack(side=tk.LEFT)
        if key:
            tk.Label(row, text=key, bg=C_BG, fg=C_TEXT_MUTED,
                     font=self.f_kbd).pack(side=tk.RIGHT)

    def _slider_block(self, parent, label, var, frm, to, cmd, formatter, unit):
        self._section_header(parent, label)
        scale = ttk.Scale(parent, from_=frm, to=to, variable=var,
                          orient=tk.HORIZONTAL, command=cmd,
                          style="Mini.Horizontal.TScale")
        scale.pack(fill=tk.X)
        value_row = tk.Frame(parent, bg=C_BG)
        value_row.pack(fill=tk.X, pady=(4, 0))
        value = tk.Label(value_row, text=formatter(), bg=C_BG, fg=C_TEXT,
                         font=self.f_value)
        value.pack(side=tk.LEFT)
        if unit:
            tk.Label(value_row, text=unit, bg=C_BG, fg=C_TEXT_DIM,
                     font=self.f_unit).pack(side=tk.LEFT, padx=(2, 0), pady=(8, 0))
        self._spacer(parent, 18)
        return lambda: value.configure(text=formatter())

    def _button_pill(self, parent, text_var, cmd):
        btn = tk.Label(
            parent, textvariable=text_var, bg=C_SURFACE, fg=C_TEXT,
            font=self.f_button, padx=14, pady=11, anchor="w",
            cursor="hand2",
        )
        btn.pack(fill=tk.X)
        self._hover_swap(btn, C_SURFACE, C_SURFACE_HI)
        btn.bind("<Button-1>", lambda e: cmd())
        return btn

    def _toggle_row(self, parent, label, key, var, cmd):
        row = tk.Frame(parent, bg=C_BG, cursor="hand2")
        row.pack(fill=tk.X, pady=2)
        pad = tk.Frame(row, bg=C_BG)
        pad.pack(fill=tk.X, padx=2, pady=8)
        dot = tk.Label(pad, text=self._dot(var.get()), bg=C_BG,
                       fg=C_ACCENT if var.get() else C_TEXT_MUTED,
                       font=self.f_button)
        dot.pack(side=tk.LEFT, padx=(0, 12))
        lbl = tk.Label(pad, text=label, bg=C_BG, fg=C_TEXT, font=self.f_button)
        lbl.pack(side=tk.LEFT)
        kbd = tk.Label(pad, text=key, bg=C_BG, fg=C_TEXT_MUTED, font=self.f_kbd)
        kbd.pack(side=tk.RIGHT)
        for w in (row, pad, dot, lbl, kbd):
            w.bind("<Button-1>", lambda e: cmd())
        return row, dot

    def _action_row(self, parent, label, key, cmd):
        row = tk.Frame(parent, bg=C_BG, cursor="hand2")
        row.pack(fill=tk.X, pady=1)
        pad = tk.Frame(row, bg=C_BG)
        pad.pack(fill=tk.X, padx=2, pady=2)
        lbl = tk.Label(pad, text=label, bg=C_BG, fg=C_TEXT,
                       font=self.f_button, anchor="w", padx=12, pady=10)
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        kbd = tk.Label(pad, text=key, bg=C_BG, fg=C_TEXT_MUTED,
                       font=self.f_kbd, padx=12, pady=10)
        kbd.pack(side=tk.RIGHT)
        widgets = (row, pad, lbl, kbd)
        for w in widgets:
            w.bind("<Button-1>", lambda e: cmd())
            w.bind("<Enter>", lambda e: self._set_bg(widgets, C_SURFACE))
            w.bind("<Leave>", lambda e: self._set_bg(widgets, C_BG))

    @staticmethod
    def _set_bg(widgets, color: str) -> None:
        for w in widgets:
            try:
                w.configure(bg=color)
            except tk.TclError:
                pass

    @staticmethod
    def _hover_swap(widget: tk.Widget, normal: str, hover: str) -> None:
        widget.bind("<Enter>", lambda e: widget.configure(bg=hover))
        widget.bind("<Leave>", lambda e: widget.configure(bg=normal))

    def _divider(self, parent: tk.Frame) -> None:
        self._spacer(parent, 14)
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill=tk.X)
        self._spacer(parent, 16)

    @staticmethod
    def _spacer(parent: tk.Frame, h: int) -> None:
        tk.Frame(parent, bg=C_BG, height=h).pack(fill=tk.X)

    @staticmethod
    def _dot(on: bool) -> str:
        return "●" if on else "○"

    # ---------- Atalhos ----------

    def _bind_keys(self) -> None:
        b = self.root.bind
        b("<plus>",        lambda e: self._set_zoom(self.settings.zoom + 0.5))
        b("<KP_Add>",      lambda e: self._set_zoom(self.settings.zoom + 0.5))
        b("<equal>",       lambda e: self._set_zoom(self.settings.zoom + 0.5))
        b("<minus>",       lambda e: self._set_zoom(self.settings.zoom - 0.5))
        b("<KP_Subtract>", lambda e: self._set_zoom(self.settings.zoom - 0.5))
        for k in ("f", "F"): b(f"<{k}>", lambda e: self._cycle_filter())
        for k in ("c", "C"): b(f"<{k}>", lambda e: self._toggle_clahe_action())
        b("<bracketleft>",  lambda e: self._set_brightness(self.settings.brightness - 10))
        b("<bracketright>", lambda e: self._set_brightness(self.settings.brightness + 10))
        b("<comma>",        lambda e: self._set_contrast(self.settings.contrast - 0.1))
        b("<period>",       lambda e: self._set_contrast(self.settings.contrast + 0.1))
        for k in ("o", "O"): b(f"<{k}>", lambda e: self._do_ocr())
        for k in ("l", "L"): b(f"<{k}>", lambda e: self._do_speak())
        for k in ("p", "P"): b(f"<{k}>", lambda e: self._toggle_pause())
        for k in ("h", "H"): b(f"<{k}>", lambda e: self._toggle_freeze())
        for k in ("s", "S"): b(f"<{k}>", lambda e: self._toggle_source())
        for k in ("r", "R"): b(f"<{k}>", lambda e: self._select_region())
        for k in ("m", "M"): b(f"<{k}>", lambda e: self._toggle_magnifier())
        b("<Escape>", lambda e: self._on_close())

    # ---------- Setters / toggles ----------

    def _set_zoom(self, value: float) -> None:
        value = max(1.0, min(10.0, round(value * 2) / 2))
        self.settings.zoom = value
        self.zoom_var.set(value)
        self._zoom_value()

    def _set_brightness(self, value: int) -> None:
        value = max(-100, min(100, int(value)))
        self.settings.brightness = value
        self.brightness_var.set(value)
        self._brightness_value()

    def _set_contrast(self, value: float) -> None:
        value = max(0.5, min(3.0, round(value, 1)))
        self.settings.contrast = value
        self.contrast_var.set(value)
        self._contrast_value()

    def _cycle_filter(self) -> None:
        self.settings.filter_mode = cycle_filter(self.settings.filter_mode, 1)
        self.filter_var.set(FILTER_LABELS[self.settings.filter_mode])

    def _toggle_clahe_action(self) -> None:
        self.clahe_var.set(not self.clahe_var.get())
        self.settings.clahe_enabled = bool(self.clahe_var.get())
        self._clahe_dot.configure(
            text=self._dot(self.settings.clahe_enabled),
            fg=C_ACCENT if self.settings.clahe_enabled else C_TEXT_MUTED,
        )

    def _toggle_pause(self) -> None:
        if self.source == self.SOURCE_CAMERA:
            self.camera.pause(not self.camera.paused)
        else:
            self.screen.pause(not self.screen.paused)

    def _toggle_freeze(self) -> None:
        if self.frozen:
            self.frozen = False
            self.frozen_frame = None
            return
        frame = self._read_source()
        if frame is not None:
            self.frozen = True
            self.frozen_frame = frame.copy()

    def _toggle_source(self) -> None:
        target = (self.SOURCE_SCREEN if self.source == self.SOURCE_CAMERA
                  else self.SOURCE_CAMERA)
        self._set_source(target)

    def _set_source(self, target: str) -> None:
        if target == self.source:
            return
        self.source = target
        self.frozen = False
        self.frozen_frame = None
        self._photo_size = (0, 0)  # forçar recriação do PhotoImage
        self._refresh_source_switch()
        if target == self.SOURCE_SCREEN:
            self._set_status("Modo Tela ativado · use R para selecionar uma região")
        else:
            self._set_status("Modo Câmera")

    def _cycle_language(self) -> None:
        idx = self.LANG_CHOICES.index(self.lang_var.get()) \
            if self.lang_var.get() in self.LANG_CHOICES else 0
        new = self.LANG_CHOICES[(idx + 1) % len(self.LANG_CHOICES)]
        self.lang_var.set(new)
        self.ocr.set_language(new)
        self._refresh_lang_status()

    def _refresh_lang_status(self) -> None:
        msg = self.ocr.language_status()
        # apenas no rodapé, e quando há aviso de idioma faltando, no painel também
        if not self.ocr.is_ready or "ausente" in msg.lower() or "indispon" in msg.lower():
            self._show_ocr_text(msg, "")
        self._set_status(msg)

    # ---------- OCR / TTS ----------

    def _set_ocr_placeholder(self) -> None:
        self.ocr_text.configure(state=tk.NORMAL, fg=C_TEXT_MUTED, font=self.f_ocr_hint)
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert("1.0", "Pressione O para reconhecer o texto da imagem.")
        self.ocr_text.configure(state=tk.DISABLED)
        self.ocr_meta_var.set("")

    def _show_ocr_text(self, text: str, meta: str = "", warn: bool = False) -> None:
        self.ocr_text.configure(
            state=tk.NORMAL,
            fg=C_WARN if warn else C_TEXT,
            font=self.f_ocr,
        )
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert("1.0", text)
        self.ocr_text.configure(state=tk.DISABLED)
        self.ocr_meta_var.set(meta)

    def _read_source(self) -> np.ndarray | None:
        if self.source == self.SOURCE_CAMERA:
            return self.camera.read()
        return self.screen.read()

    def _do_ocr(self) -> None:
        if self._ocr_running:
            return
        if not self.ocr.is_ready:
            self._show_ocr_text(self.ocr.language_status(), "", warn=True)
            return
        # Usa o último frame ORIGINAL e aplica o pipeline para OCR (sem
        # filtros de cor) — assim o OCR herda zoom/CLAHE/brilho/contraste
        # mas mantém tons de cinza naturais que o Tesseract lê melhor.
        if self.frozen and self.frozen_frame is not None:
            source = self.frozen_frame
        else:
            source = self.last_source_frame if self.last_source_frame is not None \
                else self._read_source()
        if source is None:
            self._set_status("Sem quadro disponível para OCR.")
            return
        prepared = process_for_ocr(source, self.settings)
        self._ocr_running = True
        self._show_ocr_text("Reconhecendo texto…", "", warn=False)
        threading.Thread(target=self._ocr_worker, args=(prepared,), daemon=True).start()

    def _ocr_worker(self, frame: np.ndarray) -> None:
        result = self.ocr.recognize(frame)
        if not self._closing:
            self.root.after(0, lambda: self._on_ocr_done(result))

    def _on_ocr_done(self, result: OCRResult) -> None:
        self._ocr_running = False
        if not result.ok:
            self._show_ocr_text(result.note or "OCR falhou.", "", warn=True)
            self._set_status("OCR indisponível.")
            return
        if not result.text:
            self._show_ocr_text(
                result.note or "(nenhum texto detectado)", "",
                warn=True,
            )
            self._set_status("OCR concluído sem detecções.")
            return
        self.last_ocr_text = result.text
        meta = (
            f"{result.word_count} palavra(s) · "
            f"{result.confidence:.0f}% conf · "
            f"PSM {result.psm_used} · "
            f"{result.elapsed_ms:.0f} ms"
        )
        self._show_ocr_text(result.text, meta, warn=result.low_confidence)
        if result.note:
            self._set_status(result.note)
        else:
            self._set_status(f"OCR concluído em {result.elapsed_ms:.0f} ms")

    def _do_speak(self) -> None:
        if not self.last_ocr_text.strip():
            self._set_status("Execute o OCR antes de ler em voz alta.")
            return
        ok = self.tts.speak(self.last_ocr_text)
        if not ok and self.tts.error:
            self._set_status(f"TTS indisponível: {self.tts.error}")
        else:
            self._set_status("Lendo em voz alta…")

    # ---------- Tela: seleção e magnifier ----------

    def _select_region(self) -> None:
        if self.source != self.SOURCE_SCREEN:
            self._toggle_source()
        if self.screen.virtual is None:
            self._set_status("Captura de tela indisponível.")
            return
        # esconde a janela durante a seleção para não nos vermos a nós mesmos
        self.root.withdraw()
        self.root.update_idletasks()
        time.sleep(0.15)

        def done(region: Region | None) -> None:
            self.root.deiconify()
            if region and region.is_valid():
                self.screen.set_region(region)
                self._set_status(
                    f"Região: {region.width}×{region.height} em ({region.left},{region.top})"
                )
            else:
                self._set_status("Seleção cancelada.")
            self._photo_size = (0, 0)

        sel = RegionSelector(self.root, self.screen.virtual)
        sel.run(done)

    def _toggle_magnifier(self) -> None:
        if self.magnifier and self.magnifier.is_open:
            self.magnifier.close()
            self.magnifier = None
            self._set_status("Lupa fechada.")
            return
        self.magnifier = CursorMagnifier(
            self.root,
            settings_provider=lambda: self.settings,
            process_fn=process_frame,
        )
        self.magnifier.open()
        self._set_status("Lupa do cursor ativa · Esc fecha a janelinha")

    # ---------- Loop de exibição ----------

    def _update_frame(self) -> None:
        if self._closing:
            return
        t_start = time.perf_counter()
        try:
            if self.frozen and self.frozen_frame is not None:
                source = self.frozen_frame
            else:
                source = self._read_source()
            if source is not None:
                self.last_source_frame = source
                processed = process_frame(source, self.settings)
                self.last_processed = processed
                display = self._fit_to_display(processed, DISPLAY_MAX_W, DISPLAY_MAX_H)
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                # Reutiliza o PhotoImage quando o tamanho não muda — evita
                # alocação por frame e reduz pressão de GC.
                if self._photo_ref is None or self._photo_size != pil.size:
                    self._photo_ref = ImageTk.PhotoImage(pil)
                    self._photo_size = pil.size
                else:
                    self._photo_ref.paste(pil)
                self.video_label.configure(image=self._photo_ref)
                self.root.update_idletasks()

                # FPS / latência
                now = time.time()
                dt = now - self._last_tick
                self._last_tick = now
                if dt > 0:
                    fps = 1.0 / dt
                    self.fps_ema = fps if self.fps_ema == 0 else 0.85 * self.fps_ema + 0.15 * fps
                latency = (time.perf_counter() - t_start) * 1000.0
                self.latency_ema = (
                    latency if self.latency_ema == 0
                    else 0.85 * self.latency_ema + 0.15 * latency
                )
                if not self._ocr_running:
                    self._set_status(self._status_text())
        except Exception as exc:
            self._set_status(f"Erro: {exc}")
        finally:
            if not self._closing:
                self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    @staticmethod
    def _fit_to_display(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale >= 1.0:
            return frame
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _status_text(self) -> str:
        bits = [
            "câmera" if self.source == self.SOURCE_CAMERA else "tela",
            f"{self.settings.zoom:.1f}×",
            f"{self.fps_ema:.0f} FPS",
            f"{self.latency_ema:.0f} ms",
            FILTER_LABELS[self.settings.filter_mode].lower(),
        ]
        if self.settings.clahe_enabled:
            bits.append("clahe")
        paused = (self.camera.paused if self.source == self.SOURCE_CAMERA
                  else self.screen.paused)
        if paused:
            bits.append("pausado")
        if self.frozen:
            bits.append("congelado")
        if self.magnifier and self.magnifier.is_open:
            bits.append("lupa")
        return "  ·  ".join(bits)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    # ---------- Encerramento ----------

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            if self.magnifier:
                self.magnifier.close()
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            self.screen.stop()
        except Exception:
            pass
        try:
            self.tts.shutdown()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    root = tk.Tk()
    TextVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
