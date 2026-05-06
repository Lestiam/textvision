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

from blip_engine import BLIPEngine, BLIPDescribeResult
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
        self._screen_indicator: tk.Toplevel | None = None
        self._tts_loading: bool = False
        self._tts_spinner_frame: int = 0
        self._tts_speak_start: float = 0.0
        self.blip = BLIPEngine()
        self._describe_mode: str | None = None  # "camera" quando ativo
        self._describe_loading: bool = False
        self._describe_spinner_frame: int = 0
        self._describe_start_time: float = 0.0
        self._screen_reader_active: bool = False
        self._screen_reader_thread: threading.Thread | None = None
        self._screen_reader_last_text: str = ""
        self._reader_status_var: tk.StringVar | None = None
        self._indicator_filter_var: tk.StringVar | None = None

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

        # Fator de escala DPI: razão entre pixels físicos (mss) e lógicos (Tkinter).
        # No Windows com HiDPI sem DPI-awareness, Tkinter reporta pixels lógicos
        # enquanto o mss usa pixels físicos — sem este ajuste as coordenadas do
        # seletor de região ficam deslocadas em relação ao que o usuário vê.
        self._dpi_scale = 1.0
        if len(self.screen.monitors) > 1:
            phys_w = self.screen.monitors[1].get("width", 0)  # monitor primário físico
            tk_w = self.root.winfo_screenwidth()               # largura lógica Tkinter
            if tk_w > 0 and phys_w > 0:
                self._dpi_scale = phys_w / tk_w

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

        sidebar_outer = tk.Frame(body, bg=C_BG, width=270)
        sidebar_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(24, 0))
        sidebar_outer.pack_propagate(False)

        self._sidebar_canvas = tk.Canvas(
            sidebar_outer, bg=C_BG, bd=0, highlightthickness=0,
        )
        _sb_scroll = ttk.Scrollbar(
            sidebar_outer, orient=tk.VERTICAL,
            command=self._sidebar_canvas.yview,
        )
        _sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._sidebar_canvas.configure(yscrollcommand=_sb_scroll.set)

        sidebar = tk.Frame(self._sidebar_canvas, bg=C_BG)
        _wid = self._sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda e: self._sidebar_canvas.configure(
            scrollregion=self._sidebar_canvas.bbox("all")
        ))
        self._sidebar_canvas.bind("<Configure>", lambda e:
            self._sidebar_canvas.itemconfig(_wid, width=e.width)
        )
        self._build_sidebar(sidebar)
        self._bind_sidebar_scroll(sidebar)

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

        # ---- Atalhos ----
        self._action_row(parent, "Atalhos", "?", self._show_shortcuts)
        self._spacer(parent, 8)

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
        self._build_tts_action_row(parent)
        self._build_describe_action_row(parent)

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
        self._ocr_panel_title_var = tk.StringVar(value="TEXTO RECONHECIDO")
        tk.Label(head, textvariable=self._ocr_panel_title_var, bg=C_SURFACE,
                 fg=C_TEXT_MUTED, font=self.f_section).pack(side=tk.LEFT)
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

    def _build_tts_action_row(self, parent: tk.Frame) -> None:
        """Linha 'Ler em voz alta' com indicador de carregamento animado."""
        row = tk.Frame(parent, bg=C_BG, cursor="hand2")
        row.pack(fill=tk.X, pady=1)
        pad = tk.Frame(row, bg=C_BG)
        pad.pack(fill=tk.X, padx=2, pady=2)
        lbl = tk.Label(pad, text="Ler em voz alta", bg=C_BG, fg=C_TEXT,
                       font=self.f_button, anchor="w", padx=12, pady=10)
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._tts_spinner_label = tk.Label(pad, text="", bg=C_BG, fg=C_ACCENT,
                                           font=self.f_button, padx=4, pady=10)
        self._tts_spinner_label.pack(side=tk.LEFT)
        kbd = tk.Label(pad, text="L", bg=C_BG, fg=C_TEXT_MUTED,
                       font=self.f_kbd, padx=12, pady=10)
        kbd.pack(side=tk.RIGHT)
        widgets = (row, pad, lbl, kbd)
        for w in widgets:
            w.bind("<Button-1>", lambda e: self._do_speak())
            w.bind("<Enter>", lambda e: self._set_bg(widgets, C_SURFACE))
            w.bind("<Leave>", lambda e: self._set_bg(widgets, C_BG))

    def _show_shortcuts(self) -> None:
        """Popup com todos os atalhos de teclado, estilizado."""
        win = tk.Toplevel(self.root)
        win.title("Atalhos")
        win.configure(bg=C_BG)
        win.resizable(False, False)
        win.attributes("-topmost", True)
        self.root.update_idletasks()
        rx, ry = self.root.winfo_x(), self.root.winfo_y()
        rw = self.root.winfo_width()
        w, h = 360, 444
        win.geometry(f"{w}x{h}+{rx + (rw - w) // 2}+{ry + 60}")

        outer = tk.Frame(win, bg=C_BORDER)
        outer.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        inner = tk.Frame(outer, bg=C_BG)
        inner.pack(fill=tk.BOTH, expand=True)

        hdr = tk.Frame(inner, bg=C_SURFACE_HI)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="ATALHOS DE TECLADO", bg=C_SURFACE_HI, fg=C_TEXT_DIM,
                 font=self.f_section, padx=18, pady=14).pack(side=tk.LEFT)
        close = tk.Label(hdr, text="✕", bg=C_SURFACE_HI, fg=C_TEXT_MUTED,
                         font=self.f_button, padx=16, pady=14, cursor="hand2")
        close.pack(side=tk.RIGHT)
        close.bind("<Button-1>", lambda e: win.destroy())
        win.bind("<Escape>", lambda e: win.destroy())
        self._hover_swap(close, C_SURFACE_HI, C_BORDER)

        shortcuts = [
            ("+ / −",  "Aumentar / diminuir zoom"),
            ("F",      "Alternar filtro de cor"),
            ("C",      "Ativar / desativar CLAHE"),
            ("[ / ]",  "Diminuir / aumentar brilho"),
            (", / .",  "Diminuir / aumentar contraste"),
            ("O",      "Reconhecer texto (OCR)"),
            ("L",      "Ler em voz alta"),
            ("S",      "Alternar fonte  câmera ↔ tela"),
            ("R",      "Selecionar região  (modo tela)"),
            ("M",      "Lupa flutuante  (segue o cursor)"),
            ("P",      "Pausar / retomar fonte"),
            ("H",      "Congelar / descongelar quadro"),
            ("Esc",    "Sair"),
        ]
        content = tk.Frame(inner, bg=C_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)
        for i, (key, desc) in enumerate(shortcuts):
            row_bg = C_SURFACE if i % 2 == 0 else C_BG
            r = tk.Frame(content, bg=row_bg)
            r.pack(fill=tk.X)
            tk.Label(r, text=key, bg=row_bg, fg=C_ACCENT,
                     font=self.f_kbd, width=8, anchor="e",
                     padx=8, pady=7).pack(side=tk.LEFT)
            tk.Frame(r, bg=C_BORDER, width=1).pack(
                side=tk.LEFT, fill=tk.Y, pady=3)
            tk.Label(r, text=desc, bg=row_bg, fg=C_TEXT,
                     font=self.f_button, anchor="w",
                     padx=12, pady=7).pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        # bind_all garante que os atalhos funcionam mesmo quando o foco
        # está no indicador flutuante (root iconificada).
        b = self.root.bind_all
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
        for k in ("d", "D"): b(f"<{k}>", lambda e: self._do_describe())
        for k in ("p", "P"): b(f"<{k}>", lambda e: self._toggle_pause())
        for k in ("h", "H"): b(f"<{k}>", lambda e: self._toggle_freeze())
        for k in ("s", "S"): b(f"<{k}>", lambda e: self._toggle_source())
        for k in ("r", "R"): b(f"<{k}>", lambda e: self._select_region())
        for k in ("m", "M"): b(f"<{k}>", lambda e: self._toggle_magnifier())
        b("<space>",  lambda e: self._capture_for_describe())
        b("<Escape>", lambda e: self._on_esc())

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
        if self._indicator_filter_var is not None:
            try:
                self._indicator_filter_var.set(
                    f"Filtro: {FILTER_LABELS[self.settings.filter_mode]}"
                )
            except tk.TclError:
                pass

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
            self._set_status("Modo Tela ativado · clique em 'Ler em voz alta' para iniciar")
            self._last_normal_geom = self.root.geometry()
            self.root.iconify()
            self.root.update_idletasks()
            self._show_screen_indicator()
        else:
            self._stop_screen_reader()
            self._hide_screen_indicator()
            self.root.deiconify()
            self.root.lift()
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

    def _show_ocr_text(self, text: str, meta: str = "", warn: bool = False,
                       panel_title: str = "TEXTO RECONHECIDO") -> None:
        try:
            self._ocr_panel_title_var.set(panel_title)
        except AttributeError:
            pass
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
        if self.source == self.SOURCE_SCREEN:
            self._open_window_picker()
            return
        if not self.last_ocr_text.strip():
            self._set_status("Execute o OCR antes de ler em voz alta.")
            return
        ok = self.tts.speak(self.last_ocr_text)
        if not ok and self.tts.error:
            self._set_status(f"TTS indisponível: {self.tts.error}")
        else:
            self._set_status("Lendo em voz alta…")
            self._start_tts_spinner()

    # ---------- Tela: seleção e magnifier ----------

    def _select_region(self) -> None:
        if self.source != self.SOURCE_SCREEN:
            self._toggle_source()
        if self.screen.virtual is None:
            self._set_status("Captura de tela indisponível.")
            return
        if self._screen_indicator is not None:
            try:
                self._screen_indicator.withdraw()
            except Exception:
                pass
        # No Windows, Toplevels filhos ficam ocultos quando o owner está
        # iconic/withdrawn. Mantemos root em estado "normal" mas invisível
        # (alpha=0 + fora da tela) para que o seletor apareça corretamente.
        # Usamos after() em vez de time.sleep() para não bloquear o event loop,
        # garantindo que as mudanças de estado da janela sejam processadas antes
        # de criar o RegionSelector.
        self._overlay_hide_root()
        self.root.after(250, self._open_region_selector)

    def _open_region_selector(self) -> None:
        def done(region: Region | None) -> None:
            self._hide_screen_indicator()
            self._overlay_show_root()
            if region and region.is_valid():
                self.screen.set_region(region)
                self._set_status(
                    f"Região: {region.width}×{region.height} em ({region.left},{region.top})"
                )
            else:
                self._set_status("Seleção cancelada.")
            self._photo_size = (0, 0)

        sel = RegionSelector(self.root, self.screen.virtual, self._dpi_scale)
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

    # ---------- Descrição de imagem (BLIP) ----------

    def _build_describe_action_row(self, parent: tk.Frame) -> None:
        """Linha 'Descrever imagem' com spinner de carregamento."""
        row = tk.Frame(parent, bg=C_BG, cursor="hand2")
        row.pack(fill=tk.X, pady=1)
        pad = tk.Frame(row, bg=C_BG)
        pad.pack(fill=tk.X, padx=2, pady=2)
        lbl = tk.Label(pad, text="Descrever imagem", bg=C_BG, fg=C_TEXT,
                       font=self.f_button, anchor="w", padx=12, pady=10)
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._describe_spinner_label = tk.Label(pad, text="", bg=C_BG, fg=C_ACCENT,
                                                font=self.f_button, padx=4, pady=10)
        self._describe_spinner_label.pack(side=tk.LEFT)
        kbd = tk.Label(pad, text="D", bg=C_BG, fg=C_TEXT_MUTED,
                       font=self.f_kbd, padx=12, pady=10)
        kbd.pack(side=tk.RIGHT)
        widgets = (row, pad, lbl, kbd)
        for w in widgets:
            w.bind("<Button-1>", lambda e: self._do_describe())
            w.bind("<Enter>", lambda e: self._set_bg(widgets, C_SURFACE))
            w.bind("<Leave>", lambda e: self._set_bg(widgets, C_BG))

    def _bind_sidebar_scroll(self, widget: tk.Widget) -> None:
        """Vincula roda do mouse ao canvas da sidebar recursivamente."""
        def _scroll(event):
            self._sidebar_canvas.yview_scroll(-1 * (event.delta // 120), "units")
        widget.bind("<MouseWheel>", _scroll)
        for child in widget.winfo_children():
            self._bind_sidebar_scroll(child)

    def _do_describe(self) -> None:
        if self._describe_loading:
            return
        if self.source == self.SOURCE_SCREEN:
            self._stop_screen_reader()
            self._describe_screen_region()
        else:
            self._enter_describe_camera_mode()

    def _enter_describe_camera_mode(self) -> None:
        self._describe_mode = "camera"
        self._set_status(
            "Centralize o objeto na mira  ·  ESPAÇO para capturar  ·  Esc para cancelar"
        )

    def _capture_for_describe(self) -> None:
        if self._describe_mode != "camera":
            return
        # Não captura se um widget de texto tiver foco
        focus = self.root.focus_get()
        if isinstance(focus, tk.Text):
            return
        frame = self.last_source_frame
        if frame is None:
            self._set_status("Sem quadro disponível para descrição.")
            return
        self._describe_mode = None
        # Recorta a região central (60% × 65%) — área da mira
        h, w = frame.shape[:2]
        rw, rh = int(w * 0.62), int(h * 0.65)
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        crop = frame[y0:y0 + rh, x0:x0 + rw]
        self._start_describe_spinner()
        self._set_status("Descrevendo imagem…")
        threading.Thread(target=self._describe_worker, args=(crop,), daemon=True).start()

    def _describe_screen_region(self) -> None:
        if self.source != self.SOURCE_SCREEN:
            self._toggle_source()
        if self.screen.virtual is None:
            self._set_status("Captura de tela indisponível.")
            return
        if self._screen_indicator is not None:
            try:
                self._screen_indicator.withdraw()
            except Exception:
                pass
        self._overlay_hide_root()
        self.root.after(250, self._open_describe_selector)

    def _open_describe_selector(self) -> None:
        def done(region: Region | None) -> None:
            self._hide_screen_indicator()
            self._overlay_show_root()
            if region and region.is_valid():
                frame = self.screen.grab_once(region)
                if frame is not None:
                    # Atualiza o live view para mostrar a mesma região descrita
                    self.screen.set_region(region)
                    self._start_describe_spinner()
                    self._set_status("Descrevendo imagem…")
                    threading.Thread(
                        target=self._describe_worker, args=(frame,), daemon=True
                    ).start()
                else:
                    self._set_status("Falha ao capturar a região.")
            else:
                self._set_status("Captura cancelada.")
            self._photo_size = (0, 0)

        sel = RegionSelector(self.root, self.screen.virtual, self._dpi_scale)
        sel.run(done)

    def _describe_worker(self, frame: np.ndarray) -> None:
        result = self.blip.describe(frame)
        if not self._closing:
            self.root.after(0, lambda: self._on_describe_done(result))

    def _on_describe_done(self, result: BLIPDescribeResult) -> None:
        self._stop_describe_spinner()
        if not result.ok:
            self._show_ocr_text(
                result.note or "Descrição falhou.", "",
                warn=True, panel_title="DESCRIÇÃO DA IMAGEM",
            )
            self._set_status("Descrição indisponível.")
            return
        meta = f"BLIP  ·  {result.elapsed_ms:.0f} ms"
        self._show_ocr_text(
            result.text, meta, panel_title="DESCRIÇÃO DA IMAGEM"
        )
        # Permite que TTS leia a descrição com L / "Ler em voz alta"
        self.last_ocr_text = result.text
        self._set_status(f"Descrição gerada em {result.elapsed_ms:.0f} ms  (em inglês)")

    def _draw_describe_reticle(self, frame: np.ndarray) -> np.ndarray:
        """Desenha mira centralizada com escurecimento externo."""
        out = frame.copy()
        h, w = out.shape[:2]
        rw, rh = int(w * 0.62), int(h * 0.65)
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        x1, y1 = x0 + rw, y0 + rh

        # Escurece a área externa
        dim = np.full_like(out, (14, 14, 16), dtype=np.uint8)
        cv2.addWeighted(out, 0.35, dim, 0.65, 0, out)
        # Restaura o interior
        out[y0:y1, x0:x1] = frame[y0:y1, x0:x1]

        # Cantos com brackets — cor C_ACCENT em BGR
        accent = (255, 197, 158)
        L, T = 22, 2
        for cx, cy, dx, dy in [
            (x0, y0, 1, 1), (x1, y0, -1, 1),
            (x0, y1, 1, -1), (x1, y1, -1, -1),
        ]:
            cv2.line(out, (cx, cy), (cx + dx * L, cy), accent, T + 1)
            cv2.line(out, (cx, cy), (cx, cy + dy * L), accent, T + 1)

        # Mira central
        cx, cy = w // 2, h // 2
        cv2.line(out, (cx - 12, cy), (cx + 12, cy), accent, 1)
        cv2.line(out, (cx, cy - 12), (cx, cy + 12), accent, 1)
        cv2.circle(out, (cx, cy), 3, accent, 1)

        # Texto de dica abaixo da mira
        hint = "ESPACO: capturar   ESC: cancelar"
        fs, thick, font = 0.42, 1, cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(hint, font, fs, thick)
        tx = max(0, (w - tw) // 2)
        ty = min(h - 6, y1 + th + 10)
        cv2.rectangle(out, (tx - 5, ty - th - 4), (tx + tw + 5, ty + 4),
                      (14, 14, 16), -1)
        cv2.putText(out, hint, (tx, ty), font, fs, accent, thick, cv2.LINE_AA)
        return out

    def _start_describe_spinner(self) -> None:
        self._describe_loading = True
        self._describe_spinner_frame = 0
        self._describe_start_time = time.time()
        self._animate_describe_spinner()

    def _stop_describe_spinner(self) -> None:
        self._describe_loading = False
        try:
            self._describe_spinner_label.configure(text="")
        except (AttributeError, tk.TclError):
            pass

    def _animate_describe_spinner(self) -> None:
        if not self._describe_loading or self._closing:
            return
        frames = ("◐", "◓", "◑", "◒")
        try:
            self._describe_spinner_label.configure(
                text=frames[self._describe_spinner_frame % 4]
            )
        except (AttributeError, tk.TclError):
            pass
        self._describe_spinner_frame += 1
        self.root.after(200, self._animate_describe_spinner)

    # ---------- Indicador de tela flutuante ----------

    def _show_screen_indicator(self) -> None:
        """Cria pequeno painel flutuante no canto superior direito enquanto
        a janela principal está minimizada no modo Tela."""
        if self._screen_indicator is not None:
            return
        self._reader_status_var = tk.StringVar(
            value="Clique em 'Ler em voz alta' para começar"
        )
        self._indicator_filter_var = tk.StringVar(
            value=f"Filtro: {FILTER_LABELS[self.settings.filter_mode]}"
        )
        ind = tk.Toplevel()
        ind.overrideredirect(True)
        ind.attributes("-topmost", True)
        try:
            ind.attributes("-alpha", 0.95)
        except tk.TclError:
            pass
        ind.configure(bg=C_BG)

        outer = tk.Frame(ind, bg=C_BORDER)
        outer.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        panel = tk.Frame(outer, bg=C_SURFACE)
        panel.pack(fill=tk.BOTH, expand=True)

        hdr = tk.Frame(panel, bg=C_SURFACE_HI)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="TextVision", bg=C_SURFACE_HI, fg=C_TEXT,
                 font=self.f_button, padx=12, pady=8).pack(side=tk.LEFT)
        tk.Label(hdr, text="● TELA", bg=C_SURFACE_HI, fg=C_ACCENT,
                 font=self.f_kbd, padx=10).pack(side=tk.RIGHT)

        status_frame = tk.Frame(panel, bg=C_SURFACE)
        status_frame.pack(fill=tk.X, padx=8, pady=(4, 2))
        tk.Label(
            status_frame, textvariable=self._reader_status_var,
            bg=C_SURFACE, fg=C_TEXT_DIM, font=self.f_status,
            anchor="w", justify=tk.LEFT, wraplength=268,
        ).pack(fill=tk.X)

        btns = tk.Frame(panel, bg=C_SURFACE)
        btns.pack(fill=tk.X, padx=8, pady=(4, 8))

        speak_btn = tk.Label(btns, text="Ler em voz alta  L", bg=C_ACCENT_DIM,
                             fg=C_TEXT, font=self.f_kbd, padx=8, pady=5, cursor="hand2")
        speak_btn.pack(fill=tk.X, pady=(0, 4))
        speak_btn.bind("<Button-1>", lambda e: self._open_window_picker())
        self._hover_swap(speak_btn, C_ACCENT_DIM, "#2a4a6e")

        desc_btn = tk.Label(btns, text="Descrever imagem  D", bg=C_SURFACE_HI,
                            fg=C_TEXT, font=self.f_kbd, padx=8, pady=5, cursor="hand2")
        desc_btn.pack(fill=tk.X, pady=(0, 8))
        desc_btn.bind("<Button-1>", lambda e: self._do_describe())
        self._hover_swap(desc_btn, C_SURFACE_HI, C_BORDER)

        tk.Frame(btns, bg=C_BORDER, height=1).pack(fill=tk.X, pady=(0, 8))

        filter_btn = tk.Label(btns, textvariable=self._indicator_filter_var,
                              bg=C_SURFACE_HI, fg=C_TEXT_DIM,
                              font=self.f_kbd, padx=8, pady=5, cursor="hand2")
        filter_btn.pack(fill=tk.X, pady=(0, 4))
        filter_btn.bind("<Button-1>", lambda e: self._cycle_filter())
        self._hover_swap(filter_btn, C_SURFACE_HI, C_BORDER)

        mag_btn = tk.Label(btns, text="Lupa do cursor  M", bg=C_SURFACE_HI,
                           fg=C_TEXT_DIM, font=self.f_kbd, padx=8, pady=5, cursor="hand2")
        mag_btn.pack(fill=tk.X, pady=(0, 4))
        mag_btn.bind("<Button-1>", lambda e: self._toggle_magnifier())
        self._hover_swap(mag_btn, C_SURFACE_HI, C_BORDER)

        cam_btn = tk.Label(btns, text="Câmera  S", bg=C_SURFACE_HI,
                           fg=C_TEXT, font=self.f_kbd, padx=8, pady=5, cursor="hand2")
        cam_btn.pack(fill=tk.X)
        cam_btn.bind("<Button-1>", lambda e: self._set_source(self.SOURCE_CAMERA))
        self._hover_swap(cam_btn, C_SURFACE_HI, C_BORDER)

        ind.update_idletasks()
        w = max(290, ind.winfo_reqwidth() + 4)
        h = ind.winfo_reqheight() + 4
        sw = self.root.winfo_screenwidth()
        ind.geometry(f"{w}x{h}+{sw - w - 20}+20")
        ind.focus_force()
        self._screen_indicator = ind

    # ---------- Window picker / screen reader ----------

    def _open_window_picker(self) -> None:
        """Mostra lista de janelas abertas para o usuário escolher qual ler."""
        windows = self._enumerate_windows()
        if not windows:
            self._update_reader_status("Nenhuma janela disponível.")
            return

        picker = tk.Toplevel()
        picker.overrideredirect(True)
        picker.attributes("-topmost", True)
        try:
            picker.attributes("-alpha", 0.97)
        except tk.TclError:
            pass
        picker.configure(bg=C_BG)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w, h = 460, min(420, 76 + len(windows) * 36 + 20)
        picker.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        outer = tk.Frame(picker, bg=C_BORDER)
        outer.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        inner = tk.Frame(outer, bg=C_BG)
        inner.pack(fill=tk.BOTH, expand=True)

        hdr = tk.Frame(inner, bg=C_SURFACE_HI)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Selecionar janela para leitura",
                 bg=C_SURFACE_HI, fg=C_TEXT, font=self.f_button,
                 padx=14, pady=10).pack(side=tk.LEFT)
        close_lbl = tk.Label(hdr, text="✕", bg=C_SURFACE_HI, fg=C_TEXT_MUTED,
                             font=self.f_button, padx=12, pady=10, cursor="hand2")
        close_lbl.pack(side=tk.RIGHT)
        close_lbl.bind("<Button-1>", lambda e: picker.destroy())
        self._hover_swap(close_lbl, C_SURFACE_HI, C_BORDER)

        tk.Label(inner, text="Clique na janela que deseja ler em voz alta:",
                 bg=C_BG, fg=C_TEXT_DIM, font=self.f_status,
                 padx=14, pady=6, anchor="w").pack(fill=tk.X)

        scroll_wrap = tk.Frame(inner, bg=C_BG)
        scroll_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        cvs = tk.Canvas(scroll_wrap, bg=C_BG, bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(scroll_wrap, orient=tk.VERTICAL, command=cvs.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cvs.configure(yscrollcommand=sb.set)
        list_frame = tk.Frame(cvs, bg=C_BG)
        _wid = cvs.create_window((0, 0), window=list_frame, anchor="nw")
        list_frame.bind("<Configure>", lambda e: cvs.configure(
            scrollregion=cvs.bbox("all")))
        cvs.bind("<Configure>", lambda e: cvs.itemconfig(_wid, width=e.width))
        picker.bind("<Escape>", lambda e: picker.destroy())

        def _select(hwnd, title):
            picker.destroy()
            self.root.after(150, lambda: self._start_window_reader(hwnd, title))

        for i, (hwnd, title) in enumerate(windows):
            nbg = C_SURFACE if i % 2 == 0 else C_BG
            row = tk.Frame(list_frame, bg=nbg, cursor="hand2")
            row.pack(fill=tk.X)
            lbl = tk.Label(row, text=title[:65], bg=nbg, fg=C_TEXT,
                           font=self.f_button, anchor="w", padx=12, pady=7)
            lbl.pack(fill=tk.X)

            def _bind(r, l, n, h_id, t):
                def _on_enter(_): r.configure(bg=C_SURFACE_HI); l.configure(bg=C_SURFACE_HI)
                def _on_leave(_): r.configure(bg=n); l.configure(bg=n)
                def _on_click(_): _select(h_id, t)
                for widget in (r, l):
                    widget.bind("<Enter>", _on_enter)
                    widget.bind("<Leave>", _on_leave)
                    widget.bind("<Button-1>", _on_click)

            _bind(row, lbl, nbg, hwnd, title)

    @staticmethod
    def _enumerate_windows() -> list[tuple[int, str]]:
        """Lista janelas visíveis com título, excluindo janelas de sistema."""
        import ctypes
        from ctypes import wintypes
        _u = ctypes.windll.user32
        GWL_EXSTYLE, GWL_STYLE = -20, -16
        WS_EX_TOOLWINDOW = 0x00000080
        WS_CAPTION = 0x00C00000
        windows: list[tuple[int, str]] = []
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def _cb(hwnd, _):
            if not _u.IsWindowVisible(hwnd):
                return True
            if _u.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
                return True
            if not (_u.GetWindowLongW(hwnd, GWL_STYLE) & WS_CAPTION):
                return True
            n = _u.GetWindowTextLengthW(hwnd)
            if n == 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            _u.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value.strip()
            if title and title not in ("TextVision", "Program Manager"):
                windows.append((int(hwnd), title))
            return True

        _u.EnumWindows(EnumWindowsProc(_cb), 0)
        return windows

    def _start_window_reader(self, hwnd: int, title: str) -> None:
        """Para qualquer leitura anterior e inicia a leitura da janela indicada."""
        self._stop_screen_reader()
        self._screen_reader_active = True
        self._screen_reader_last_text = ""
        self._update_reader_status(f"Lendo: {title[:35]}…")
        self._screen_reader_thread = threading.Thread(
            target=self._window_reader_loop, args=(hwnd,), daemon=True
        )
        self._screen_reader_thread.start()

    def _stop_screen_reader(self) -> None:
        self._screen_reader_active = False
        self.tts.cancel()
        if self._screen_reader_thread is not None:
            self._screen_reader_thread.join(timeout=2.0)
            self._screen_reader_thread = None

    def _window_reader_loop(self, hwnd: int) -> None:
        """Thread: OCR em loop na janela selecionada, fala texto novo via TTS."""
        import ctypes
        from ctypes import wintypes
        _u = ctypes.windll.user32
        try:
            _u.ShowWindow(hwnd, 9)  # SW_RESTORE
            _u.SetForegroundWindow(hwnd)
            time.sleep(0.4)
        except Exception:
            pass

        while self._screen_reader_active and not self._closing:
            try:
                rect = wintypes.RECT()
                _u.GetWindowRect(hwnd, ctypes.byref(rect))
                left, top = rect.left, rect.top
                width = rect.right - rect.left
                height = rect.bottom - rect.top
            except Exception:
                time.sleep(0.5)
                continue
            if width < 8 or height < 8:
                time.sleep(0.5)
                continue

            frame = self.screen.grab_once(Region(left, top, width, height))
            if frame is None:
                time.sleep(0.5)
                continue

            result = self.ocr.recognize(frame, psm_preference=3)
            if not self._screen_reader_active or self._closing:
                break

            text = result.text.strip() if result.text else ""
            if text and text != self._screen_reader_last_text:
                self._screen_reader_last_text = text
                self.last_ocr_text = text
                preview = text.replace("\n", " ")[:60]
                if not self._closing:
                    self.root.after(0, lambda p=preview: self._update_reader_status(p))
                self.tts.cancel()
                self.tts.speak(text)

            waited = 0.0
            while (self.tts.is_speaking and self._screen_reader_active
                   and not self._closing and waited < 120.0):
                time.sleep(0.2)
                waited += 0.2

            for _ in range(5):
                if not self._screen_reader_active or self._closing:
                    break
                time.sleep(0.2)

    def _update_reader_status(self, text: str) -> None:
        if self._reader_status_var is not None:
            try:
                self._reader_status_var.set(text)
            except tk.TclError:
                pass

    def _hide_screen_indicator(self) -> None:
        if self._screen_indicator is not None:
            try:
                self._screen_indicator.destroy()
            except Exception:
                pass
            self._screen_indicator = None

    def _overlay_hide_root(self) -> None:
        """Oculta a janela principal mantendo-a em estado 'normal'.

        No Windows, Toplevels criados com um pai iconic/withdrawn não
        aparecem na tela. Esta função mantém o root em 'normal' mas
        invisível (alpha=0 + posição fora da tela) para que overlays
        filhos (RegionSelector) possam ser exibidos corretamente.
        """
        if self.root.state() == "iconic":
            # Deiconify para sair do estado iconic — mas primeiro torna
            # invisível para evitar flash visual.
            try:
                self.root.attributes("-alpha", 0.0)
            except Exception:
                pass
            self.root.deiconify()
        else:
            try:
                self.root.attributes("-alpha", 0.0)
            except Exception:
                pass
        self.root.update_idletasks()
        # Move para fora da área visível — +99999 garante que está além
        # de qualquer monitor, inclusive configurações multi-monitor.
        self.root.geometry("+99999+0")
        self.root.update_idletasks()

    def _overlay_show_root(self) -> None:
        """Restaura a janela principal após um overlay de seleção."""
        # Restaura geometria original (salva antes de iconificar)
        geom = getattr(self, "_last_normal_geom", None)
        if geom:
            self.root.geometry(geom)
        else:
            self.root.geometry("")
        try:
            self.root.attributes("-alpha", 1.0)
        except Exception:
            pass
        self.root.deiconify()
        self.root.lift()

    # ---------- Spinner TTS ----------

    def _start_tts_spinner(self) -> None:
        self._tts_loading = True
        self._tts_spinner_frame = 0
        self._tts_speak_start = time.time()
        self._animate_tts_spinner()

    def _stop_tts_spinner(self) -> None:
        self._tts_loading = False
        try:
            self._tts_spinner_label.configure(text="")
        except (AttributeError, tk.TclError):
            pass

    def _animate_tts_spinner(self) -> None:
        if not self._tts_loading or self._closing:
            return
        frames = ("◐", "◓", "◑", "◒")
        try:
            self._tts_spinner_label.configure(
                text=frames[self._tts_spinner_frame % 4]
            )
        except (AttributeError, tk.TclError):
            pass
        self._tts_spinner_frame += 1
        elapsed = time.time() - self._tts_speak_start
        if elapsed > 1.0 and not self.tts.is_speaking:
            self._stop_tts_spinner()
            if not self._ocr_running:
                self._set_status("Leitura concluída.")
            return
        self.root.after(200, self._animate_tts_spinner)

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
                if self._describe_mode == "camera":
                    processed = self._draw_describe_reticle(processed)
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

    def _on_esc(self) -> None:
        """Esc cancela o modo descrever (câmera); se não estiver em modo
        descrever, encerra o aplicativo."""
        if self._describe_mode == "camera":
            self._describe_mode = None
            self._set_status("Modo de descrição cancelado.")
            return
        self._on_close()

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self._stop_screen_reader()
        except Exception:
            pass
        try:
            self._hide_screen_indicator()
        except Exception:
            pass
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
    import sys
    import ctypes

    if sys.platform == "win32":
        # Declara DPI-awareness antes de criar qualquer janela Tk para que
        # as coordenadas do Tkinter (pixels físicos) coincidam com as do mss.
        # Sem isso, em displays HiDPI o seletor de região captura a área errada.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # SYSTEM_DPI_AWARE
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    root = tk.Tk()

    if sys.platform == "win32":
        # Ajusta a escala de fontes do Tk para o DPI real do monitor,
        # compensando a perda do upscaling automático do SO.
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
            if dpi and dpi != 96:
                root.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass

    TextVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
