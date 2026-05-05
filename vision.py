"""Pipeline de processamento de imagem do TextVision.

Aplica zoom, CLAHE, ajuste de brilho/contraste e filtros de cor.
Otimizações:
    - CLAHE é instanciado uma vez (cv2.createCLAHE é caro).
    - Caminhos curtos quando o passo é no-op (zoom=1, sem CLAHE, etc.).
    - Filtro de cor evita roundtrips BGR↔gray quando possível.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


FILTER_MODES = (
    "normal",
    "grayscale",
    "high_contrast",
    "black_yellow",
    "yellow_black",
    "inverted",
)

FILTER_LABELS = {
    "normal": "Normal",
    "grayscale": "Escala de cinza",
    "high_contrast": "Alto contraste (P/B)",
    "black_yellow": "Preto sobre amarelo",
    "yellow_black": "Amarelo sobre preto",
    "inverted": "Cores invertidas",
}


@dataclass
class VisionSettings:
    zoom: float = 2.0
    brightness: int = 0           # -100..100
    contrast: float = 1.0         # 0.5..3.0
    clahe_enabled: bool = True
    filter_mode: str = "normal"
    stabilize: bool = True


# --- CLAHE singleton (criar é caro, reutilizar é barato) ---
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def apply_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(1, int(w / zoom))
    crop_h = max(1, int(h / zoom))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_brightness_contrast(frame: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    if brightness == 0 and abs(contrast - 1.0) < 1e-3:
        return frame
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """CLAHE no canal L do espaço LAB."""
    if frame.ndim == 2:
        return _CLAHE.apply(frame)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _binarize_for_filter(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )


def apply_filter(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normal":
        return frame
    if mode == "inverted":
        return cv2.bitwise_not(frame)
    if mode == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "high_contrast":
        bw = _binarize_for_filter(frame)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if mode in ("black_yellow", "yellow_black"):
        bw = _binarize_for_filter(frame)
        out = np.zeros((bw.shape[0], bw.shape[1], 3), dtype=np.uint8)
        yellow = (0, 255, 255)  # BGR
        if mode == "black_yellow":
            out[bw == 255] = yellow                 # texto preto, fundo amarelo
        else:
            out[bw == 0] = yellow                   # texto amarelo, fundo preto
        return out
    return frame


def process_frame(frame: np.ndarray, settings: VisionSettings) -> np.ndarray:
    """Pipeline de exibição: zoom → CLAHE → brilho/contraste → filtro."""
    out = apply_zoom(frame, settings.zoom)
    if settings.clahe_enabled:
        out = apply_clahe(out)
    out = apply_brightness_contrast(out, settings.brightness, settings.contrast)
    if settings.filter_mode != "normal":
        out = apply_filter(out, settings.filter_mode)
    return out


def process_for_ocr(frame: np.ndarray, settings: VisionSettings) -> np.ndarray:
    """Pipeline para OCR: aplica zoom, CLAHE e brilho/contraste mas
    NÃO aplica filtros de cor (Tesseract opera melhor sobre tons de cinza
    naturais, e o filtro de alto contraste já faria sua própria binarização
    pior do que a do OCR).
    """
    out = apply_zoom(frame, settings.zoom)
    if settings.clahe_enabled:
        out = apply_clahe(out)
    out = apply_brightness_contrast(out, settings.brightness, settings.contrast)
    return out


def cycle_filter(current: str, direction: int = 1) -> str:
    idx = FILTER_MODES.index(current) if current in FILTER_MODES else 0
    return FILTER_MODES[(idx + direction) % len(FILTER_MODES)]
