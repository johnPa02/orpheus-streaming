import logging
from typing import Iterable, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from orpheus_tts import OrpheusModel

if TYPE_CHECKING:  # pragma: no cover
    from generator import Segment

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"
DEFAULT_SAMPLE_RATE = 24_000


def _bytes_to_tensor(chunk: bytes) -> torch.Tensor:
    if not chunk:
        return torch.empty(0, dtype=torch.float32)

    samples = np.frombuffer(chunk, dtype="<i2")
    if samples.size == 0:
        return torch.empty(0, dtype=torch.float32)

    float_samples = samples.astype(np.float32, copy=False) / 32768.0
    return torch.from_numpy(float_samples)


class OrpheusGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        voice: str = "tara",
        max_model_len: int = 2048,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.voice = voice or "tara"
        self.max_model_len = max_model_len
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._model = OrpheusModel(model_name=self.model_name, max_model_len=self.max_model_len)
        self._warned_stream_error = False

    def set_voice(self, voice: Optional[str]) -> None:
        if voice:
            self.voice = voice

    def warmup(self) -> None:
        try:
            stream = self._model.generate_speech(
                prompt="Warmup run.",
                voice=self.voice,
                max_tokens=128,
                temperature=0.6,
                top_p=0.9,
            )
            for chunk in stream:
                tensor = _bytes_to_tensor(chunk)
                if tensor.numel():
                    break
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Orpheus warmup failed: %s", exc)

    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List["Segment"],
        max_audio_length_ms: float,
        temperature: float,
        topk: int,
    ) -> Iterable[torch.Tensor]:
        max_tokens = self._estimate_max_tokens(max_audio_length_ms)
        try:
            stream = self._model.generate_speech(
                prompt=text,
                voice=self.voice,
                temperature=max(0.1, temperature),
                top_p=0.9,
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("Failed to start Orpheus stream: %s", exc)
            return

        for chunk in stream:
            tensor = _bytes_to_tensor(chunk)
            if tensor.numel() == 0:
                continue
            yield tensor

    def _estimate_max_tokens(self, max_audio_length_ms: float) -> int:
        if not max_audio_length_ms or max_audio_length_ms <= 0:
            return 2000
        seconds = max_audio_length_ms / 1000.0
        estimate = int(seconds * 400)
        return max(256, min(estimate, 2000))


def load_orpheus_generator(
    model_name: Optional[str] = None,
    voice: Optional[str] = None,
    max_model_len: int = 2048,
) -> OrpheusGenerator:
    generator = OrpheusGenerator(
        model_name=model_name or DEFAULT_MODEL_NAME,
        voice=voice or "tara",
        max_model_len=max_model_len,
    )
    generator.warmup()
    return generator
