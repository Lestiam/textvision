"""OCR (Tesseract) e text-to-speech (SAPI5 via pyttsx3).

Pipeline de OCR otimizado:
    grayscale → upscale (se pequeno) → bilateralFilter → binarização
    (Otsu ou adaptativa, escolhida pela distribuição de luminância)
    → deskew opcional → tentativa multi-PSM com seleção da melhor
    confiança.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Caminhos comuns do binário Tesseract no Windows.
_WINDOWS_TESS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _find_local_tessdata() -> str | None:
    """Procura uma pasta `tessdata/` ao lado deste módulo com pelo menos
    um .traineddata. Útil quando o usuário não tem permissão de gravar
    na pasta padrão do Tesseract.
    """
    here = Path(__file__).resolve().parent
    candidate = here / "tessdata"
    if candidate.is_dir() and any(candidate.glob("*.traineddata")):
        return str(candidate)
    return None


@dataclass
class OCRResult:
    text: str = ""
    confidence: float = 0.0
    word_count: int = 0
    elapsed_ms: float = 0.0
    psm_used: int = 0
    language_used: str = ""
    low_confidence: bool = False
    note: str = ""              # mensagem amigável para a UI (ex.: erro, fallback)
    ok: bool = True             # False quando o engine não está utilizável


class OCREngine:
    """Wrapper sobre pytesseract com pré-processamento agressivo e
    seleção de PSM por confiança.
    """

    LOW_CONF_THRESHOLD = 55.0
    GOOD_CONF_THRESHOLD = 78.0
    PSM_PRESETS = (6, 3, 11)  # bloco uniforme, auto, texto esparso

    def __init__(self, requested_languages: str = "por+eng") -> None:
        self.requested_languages = requested_languages
        self._available: bool | None = None
        self._error: str | None = None
        self._available_langs: tuple[str, ...] = ()
        self._resolved_lang: str = ""
        self._tess_version: str = ""
        self._tessdata_dir: str | None = None  # caminho custom; None = padrão do tesseract

    # -------------------- bootstrap --------------------

    def _ensure(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pytesseract as pt
        except Exception as exc:
            self._available = False
            self._error = f"pytesseract não instalado: {exc}"
            return False

        # Tenta caminhos padrão se 'tesseract' não estiver no PATH.
        for path in _WINDOWS_TESS_PATHS:
            if os.path.exists(path):
                pt.pytesseract.tesseract_cmd = path
                break

        try:
            self._tess_version = str(pt.get_tesseract_version())
        except Exception as exc:
            self._available = False
            self._error = (
                "Tesseract OCR não encontrado. Instale o binário em "
                "https://github.com/UB-Mannheim/tesseract/wiki e marque o "
                "language pack 'Portuguese' no instalador."
            )
            return False

        # Descobrir idiomas disponíveis tanto na pasta padrão quanto no
        # tessdata local do projeto. Usamos TESSDATA_PREFIX (env var) em
        # vez de --tessdata-dir porque o último não tolera caminhos com
        # espaços no Windows (pytesseract não desescapa as aspas).
        try:
            default_langs = tuple(pt.get_languages(config=""))
        except Exception:
            default_langs = ()

        local_dir = _find_local_tessdata()
        local_langs: tuple[str, ...] = ()
        if local_dir:
            saved = os.environ.get("TESSDATA_PREFIX")
            os.environ["TESSDATA_PREFIX"] = local_dir
            try:
                local_langs = tuple(pt.get_languages(config=""))
            except Exception:
                local_langs = ()
            finally:
                if saved is None:
                    os.environ.pop("TESSDATA_PREFIX", None)
                else:
                    os.environ["TESSDATA_PREFIX"] = saved

        # Preferir o conjunto que cobre o pedido do usuário; em empate,
        # preferir o local (porque foi explicitamente provisionado).
        wanted = set((self.requested_languages or "").split("+"))
        default_cover = wanted & set(default_langs)
        local_cover = wanted & set(local_langs)
        if local_cover and len(local_cover) >= len(default_cover):
            self._tessdata_dir = local_dir
            self._available_langs = local_langs
            os.environ["TESSDATA_PREFIX"] = local_dir
        else:
            self._tessdata_dir = None
            self._available_langs = default_langs
            # remover qualquer TESSDATA_PREFIX prévio para usar o padrão
            os.environ.pop("TESSDATA_PREFIX", None)

        self._resolved_lang = self._resolve_lang(
            self.requested_languages, self._available_langs
        )
        self._available = True
        return True

    @staticmethod
    def _resolve_lang(requested: str, available: tuple[str, ...]) -> str:
        wanted = [p for p in (requested or "").split("+") if p]
        keep = [p for p in wanted if p in available]
        if keep:
            return "+".join(keep)
        if "eng" in available:
            return "eng"
        # último recurso: o primeiro idioma de texto disponível
        for lang in available:
            if lang != "osd":
                return lang
        return ""

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def available_languages(self) -> tuple[str, ...]:
        self._ensure()
        return self._available_langs

    @property
    def resolved_language(self) -> str:
        self._ensure()
        return self._resolved_lang

    @property
    def is_ready(self) -> bool:
        return self._ensure() and bool(self._resolved_lang)

    def set_language(self, requested: str) -> None:
        """Atualiza o idioma desejado; pode trocar a pasta tessdata se
        a outra cobrir melhor o que foi pedido.
        """
        self.requested_languages = requested
        if not self._available:
            return
        import pytesseract as pt

        # Limpar TESSDATA_PREFIX para enxergar o padrão.
        saved = os.environ.pop("TESSDATA_PREFIX", None)
        try:
            default_langs = tuple(pt.get_languages(config=""))
        except Exception:
            default_langs = ()

        local_dir = _find_local_tessdata()
        local_langs: tuple[str, ...] = ()
        if local_dir:
            os.environ["TESSDATA_PREFIX"] = local_dir
            try:
                local_langs = tuple(pt.get_languages(config=""))
            except Exception:
                local_langs = ()
            os.environ.pop("TESSDATA_PREFIX", None)

        wanted = set(requested.split("+"))
        default_cover = wanted & set(default_langs)
        local_cover = wanted & set(local_langs)
        if local_cover and len(local_cover) >= len(default_cover):
            self._tessdata_dir = local_dir
            self._available_langs = local_langs
            os.environ["TESSDATA_PREFIX"] = local_dir
        else:
            self._tessdata_dir = None
            self._available_langs = default_langs
            if saved is not None:
                # restaura se havia algo válido antes (provavelmente nada)
                os.environ["TESSDATA_PREFIX"] = saved
        self._resolved_lang = self._resolve_lang(requested, self._available_langs)

    def language_status(self) -> str:
        """Mensagem para a UI sobre o estado do idioma."""
        if not self._ensure():
            return self._error or "OCR indisponível."
        if not self._resolved_lang:
            return "Nenhum language pack do Tesseract encontrado."
        if self._resolved_lang != self.requested_languages:
            wanted = set((self.requested_languages or "").split("+"))
            missing = sorted(wanted - set(self._resolved_lang.split("+")))
            if missing:
                return (
                    f"Idioma(s) ausente(s): {'+'.join(missing)}. Usando "
                    f"'{self._resolved_lang}'. Para adicionar, reinstale o "
                    "Tesseract marcando o language pack ou copie o "
                    f"arquivo {missing[0]}.traineddata para a pasta tessdata."
                )
        return f"Idioma: {self._resolved_lang}"

    # -------------------- pré-processamento --------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _upscale_if_small(gray: np.ndarray, target: int = 1500) -> np.ndarray:
        h, w = gray.shape[:2]
        biggest = max(h, w)
        if biggest >= target:
            return gray
        scale = min(3.0, target / float(biggest))
        return cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        # Decide entre Otsu (cenas com bom contraste / iluminação uniforme)
        # e adaptativa (iluminação irregular) com base no desvio padrão.
        mean, std = cv2.meanStdDev(gray)
        if std[0, 0] >= 45:
            _, bw = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            bw = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 41, 15,
            )
        # Padronizar para texto escuro sobre fundo claro: se a maioria for
        # preta, inverter.
        if float(np.mean(bw)) < 127.0:
            bw = cv2.bitwise_not(bw)
        return bw

    @staticmethod
    def _deskew(bw: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(bw < 127))
        if coords.shape[0] < 200:
            return bw
        try:
            angle = cv2.minAreaRect(coords)[-1]
        except Exception:
            return bw
        if angle < -45:
            angle = 90 + angle
        # Apenas corrigir inclinações pequenas; rotações grandes vêm de
        # detecções espúrias.
        if abs(angle) < 0.5 or abs(angle) > 12:
            return bw
        h, w = bw.shape
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
        return cv2.warpAffine(
            bw, m, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)
        gray = self._upscale_if_small(gray)
        gray = cv2.bilateralFilter(gray, 5, 60, 60)
        bw = self._binarize(gray)
        bw = self._deskew(bw)
        return bw

    # -------------------- execução --------------------

    def _run_psm(self, prepared: np.ndarray, psm: int, lang: str):
        """Roda um único PSM e devolve (texto, conf_média, n_palavras)."""
        import pytesseract as pt
        config = f"--oem 1 --psm {psm}"
        try:
            data = pt.image_to_data(
                prepared, lang=lang, config=config,
                output_type=pt.Output.DICT,
            )
        except Exception as exc:
            self._error = f"image_to_data falhou: {exc}"
            return "", 0.0, 0
        words: list[str] = []
        confs: list[float] = []
        for i, word in enumerate(data.get("text", [])):
            word = (word or "").strip()
            if not word:
                continue
            try:
                c = float(data["conf"][i])
            except (ValueError, KeyError):
                c = -1.0
            if c < 0:
                continue
            words.append(word)
            confs.append(c)
        text = " ".join(words).strip()
        avg = sum(confs) / len(confs) if confs else 0.0
        return text, avg, len(words)

    def recognize(self, frame: np.ndarray, psm_preference: int | None = None) -> OCRResult:
        t0 = time.perf_counter()
        if not self._ensure():
            return OCRResult(ok=False, note=self._error or "OCR indisponível.")
        if not self._resolved_lang:
            return OCRResult(
                ok=False,
                note="Nenhum language pack do Tesseract encontrado.",
            )

        prepared = self._preprocess(frame)
        psms: list[int] = []
        if psm_preference is not None:
            psms.append(int(psm_preference))
        for p in self.PSM_PRESETS:
            if p not in psms:
                psms.append(p)

        best = OCRResult(ok=True, language_used=self._resolved_lang)
        for psm in psms:
            text, conf, n_words = self._run_psm(prepared, psm, self._resolved_lang)
            score = conf * (1.0 + 0.02 * min(n_words, 50))
            best_score = best.confidence * (1.0 + 0.02 * min(best.word_count, 50))
            if score > best_score:
                best = OCRResult(
                    text=text, confidence=conf, word_count=n_words,
                    psm_used=psm, language_used=self._resolved_lang, ok=True,
                )
            # parar cedo se já estiver bom
            if conf >= self.GOOD_CONF_THRESHOLD and n_words >= 3:
                break

        best.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        best.low_confidence = best.confidence < self.LOW_CONF_THRESHOLD
        if not best.text:
            best.note = "Nenhum texto detectado. Reaproxime, ajuste foco e tente novamente."
        elif best.low_confidence:
            best.note = "Confiança baixa — resultado pode conter erros."
        return best


# --------------------------------------------------------------------------
# Text-to-speech
# --------------------------------------------------------------------------

class TTSEngine:
    """Fila de fala em thread dedicada. Reinicia o motor a cada item para
    evitar travamentos conhecidos do SAPI5/pyttsx3 entre chamadas.
    """

    def __init__(self, rate: int = 175) -> None:
        self.rate = rate
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._available: bool | None = None
        self._error: str | None = None
        self._stop_flag = threading.Event()
        self._speaking_event = threading.Event()

    @property
    def is_speaking(self) -> bool:
        return self._speaking_event.is_set()

    def _ensure(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            self._available = True
        except Exception as exc:
            self._available = False
            self._error = str(exc)
        return self._available

    @property
    def error(self) -> str | None:
        return self._error

    def _run(self) -> None:
        import pyttsx3
        while not self._stop_flag.is_set():
            item = self._queue.get()
            if item is None:
                return
            self._speaking_event.set()
            try:
                # pyttsx3.init() cacheia a instância; limpar o cache antes de
                # cada chamada garante um engine SAPI5 novo, evitando o bug
                # onde runAndWait() não funciona na segunda fala do processo.
                try:
                    pyttsx3._activeEngines.clear()
                except Exception:
                    pass
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.say(item)
                engine.runAndWait()
                engine.stop()
            except Exception as exc:
                self._error = str(exc)
            finally:
                self._speaking_event.clear()

    def speak(self, text: str) -> bool:
        text = (text or "").strip()
        if not text or not self._ensure():
            return False
        if self._thread is None or not self._thread.is_alive():
            self._stop_flag.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self._queue.put(text)
        return True

    def cancel(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def shutdown(self) -> None:
        self._stop_flag.set()
        self._queue.put(None)
