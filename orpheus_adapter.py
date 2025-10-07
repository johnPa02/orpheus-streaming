import logging
from typing import Iterable, List, Optional, TYPE_CHECKING
import uuid
import time
import asyncio
import threading
import queue

import numpy as np
import torch
from orpheus_tts import OrpheusModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

if TYPE_CHECKING:  # pragma: no cover
    from generator import Segment

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"
DEFAULT_SAMPLE_RATE = 24_000


class CustomOrpheusModel(OrpheusModel):
    """Custom OrpheusModel that supports memory constraints to fix KV cache issues."""

    def __init__(self, model_name, dtype=torch.float16, max_model_len=2048, gpu_memory_utilization=0.85):
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._active_requests = set()  # Track active request IDs
        super().__init__(model_name, dtype)

    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

    def _generate_unique_request_id(self):
        """Generate a unique request ID that's guaranteed not to conflict."""
        while True:
            request_id = f"req-{uuid.uuid4().hex[:8]}-{int(time.time() * 1000) % 100000}"
            if request_id not in self._active_requests:
                self._active_requests.add(request_id)
                return request_id

    def _cleanup_request_id(self, request_id):
        """Clean up a request ID after completion."""
        self._active_requests.discard(request_id)

    def generate_tokens_sync(self, prompt, voice=None, request_id=None, temperature=0.6, top_p=0.8, max_tokens=1200,
                             stop_token_ids=[49158], repetition_penalty=1.3):
        """
        Override generate_tokens_sync to generate unique request IDs and handle cleanup.

        This fixes the 'Request req-001 already exists' error by generating
        unique request IDs for each request and properly cleaning them up.
        """
        # Generate unique request ID if not provided
        if request_id is None:
            request_id = self._generate_unique_request_id()
        else:
            # Even if provided, make sure it's tracked
            self._active_requests.add(request_id)

        try:
            # Format the prompt
            prompt_string = self._format_prompt(prompt, voice)
            logger.debug(f"Processing request {request_id}: {prompt}")

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty,
            )

            token_queue = queue.Queue()

            async def async_producer():
                try:
                    async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params,
                                                             request_id=request_id):
                        # Place each token text into the queue.
                        token_queue.put(result.outputs[0].text)
                    token_queue.put(None)  # Sentinel to indicate completion.
                except Exception as e:
                    logger.error(f"Error in async_producer for request {request_id}: {e}")
                    token_queue.put(None)  # Ensure we don't hang

            def run_async():
                try:
                    asyncio.run(async_producer())
                except Exception as e:
                    logger.error(f"Error in run_async for request {request_id}: {e}")
                    token_queue.put(None)

            thread = threading.Thread(target=run_async)
            thread.start()

            while True:
                token = token_queue.get()
                if token is None:
                    break
                yield token

            thread.join()

        finally:
            # Always clean up the request ID
            self._cleanup_request_id(request_id)


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
            gpu_memory_utilization: float = 0.85,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.voice = voice or "tara"
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sample_rate = DEFAULT_SAMPLE_RATE

        # Use our custom model with memory constraints
        self._model = CustomOrpheusModel(
            model_name=self.model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization
        )
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
            top_p: float,
    ) -> Iterable[torch.Tensor]:
        max_tokens = self._estimate_max_tokens(max_audio_length_ms)
        try:
            stream = self._model.generate_speech(
                prompt=text,
                voice=self.voice,
                temperature=max(0.1, temperature),
                top_p=top_p if top_p and top_p > 0 else 0.9,
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("Failed to start Orpheus stream: %s", exc)
            return []

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
        max_model_len: int = 1024,  # Reduced from 2048 to avoid KV cache issues
        gpu_memory_utilization: float = 0.85,
) -> OrpheusGenerator:
    """
    Load OrpheusGenerator with memory constraints to prevent KV cache issues.

    Args:
        model_name: Name of the model to load
        voice: Voice to use for generation
        max_model_len: Maximum sequence length (reduced to prevent memory issues)
        gpu_memory_utilization: GPU memory utilization ratio (default 0.85 instead of 0.90)

    Returns:
        OrpheusGenerator instance

    Raises:
        ValueError: If model initialization fails due to memory constraints
    """
    try:
        generator = OrpheusGenerator(
            model_name=model_name or DEFAULT_MODEL_NAME,
            voice=voice or "tara",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        generator.warmup()
        return generator
    except ValueError as e:
        if "KV cache" in str(e) or "max seq len" in str(e):
            logger.error(f"Memory constraint error: {e}")
            logger.info("Retrying with smaller max_model_len (512) and lower GPU memory utilization (0.75)")
            # Retry with more conservative settings
            try:
                generator = OrpheusGenerator(
                    model_name=model_name or DEFAULT_MODEL_NAME,
                    voice=voice or "tara",
                    max_model_len=512,
                    gpu_memory_utilization=0.75,
                )
                generator.warmup()
                return generator
            except Exception as retry_e:
                raise ValueError(
                    f"Failed to initialize OrpheusGenerator even with conservative settings. "
                    f"Original error: {e}. Retry error: {retry_e}"
                ) from retry_e
        else:
            raise
    except Exception as e:
        logger.error(f"Unexpected error during OrpheusGenerator initialization: {e}")
        raise
