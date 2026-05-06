"""Descrição de imagem com BLIP (Salesforce/blip-image-captioning-base).

O modelo (~900 MB) é baixado automaticamente pelo HuggingFace Hub na
primeira execução e armazenado em cache local (~/.cache/huggingface/).
Dependências: pip install transformers torch
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BLIPDescribeResult:
    text: str = ""
    elapsed_ms: float = 0.0
    ok: bool = True
    note: str = ""


class BLIPEngine:
    """Wrapper sobre o modelo BLIP para geração de legendas descritivas.

    Carregamento lazy: o modelo é baixado/carregado na primeira chamada
    a describe(). Chamadas seguintes reutilizam o modelo já em memória.
    """

    MODEL_ID = "Salesforce/blip-image-captioning-base"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._available: bool | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._available is True

    @property
    def error(self) -> str | None:
        return self._error

    def _ensure(self) -> bool:
        if self._available is not None:
            return self._available
        with self._lock:
            if self._available is not None:
                return self._available
            try:
                from transformers import (
                    BlipForConditionalGeneration,
                    BlipProcessor,
                )

                self._processor = BlipProcessor.from_pretrained(self.MODEL_ID)
                self._model = BlipForConditionalGeneration.from_pretrained(
                    self.MODEL_ID
                )
                self._model.eval()
                self._available = True
            except ImportError:
                self._available = False
                self._error = (
                    "Instale as dependências:\n"
                    "  pip install transformers torch\n"
                    "O modelo (~900 MB) é baixado automaticamente na 1ª execução."
                )
            except Exception as exc:
                self._available = False
                self._error = f"Erro ao carregar modelo BLIP: {exc}"
        return bool(self._available)

    def describe(self, frame: np.ndarray) -> BLIPDescribeResult:
        """Gera legenda descritiva para o frame BGR fornecido."""
        t0 = time.perf_counter()
        if not self._ensure():
            return BLIPDescribeResult(
                ok=False, note=self._error or "BLIP indisponível."
            )
        try:
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            inputs = self._processor(pil, return_tensors="pt")
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=80,
                    num_beams=4,
                    min_length=5,
                )
            text = self._processor.decode(out[0], skip_special_tokens=True)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return BLIPDescribeResult(text=text, elapsed_ms=elapsed, ok=True)
        except Exception as exc:
            return BLIPDescribeResult(ok=False, note=f"Erro na inferência: {exc}")
