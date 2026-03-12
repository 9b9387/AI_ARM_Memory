from __future__ import annotations

import math
import re
from typing import Protocol

import httpx
from loguru import logger

from arm_memory.config import ARMConfig


_STOP_WORDS = frozenset(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 我们 你们 他们 她们"
    .split()
)


def tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]{1,4}|[a-zA-Z0-9_]+", text.lower())
    results: list[str] = []
    for token in tokens:
        if token in _STOP_WORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            results.extend(char for char in token if char not in _STOP_WORDS)
        elif len(token) > 1:
            results.append(token)
    return results


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class EmbeddingProvider(Protocol):
    dimensions: int

    def tokenize(self, text: str) -> list[str]:
        ...

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    def healthcheck(self) -> bool:
        ...


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        timeout: int = 120,
        api_key: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = max(32, dimensions)
        self.timeout = timeout
        self.api_key = api_key

    def tokenize(self, text: str) -> list[str]:
        return tokenize_text(text)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {
            "input": texts,
        }
        if self.model:
            payload["model"] = self.model
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.post(f"{self.base_url}/embeddings", json=payload)
            response.raise_for_status()
            items = response.json().get("data") or []
        embeddings = [list(item.get("embedding") or []) for item in items]
        normalized = [_normalize_vector([float(value) for value in vector]) for vector in embeddings]
        for vector in normalized:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}"
                )
        return normalized

    def healthcheck(self) -> bool:
        try:
            self.embed("healthcheck")
            return True
        except Exception:
            logger.exception("OpenAI-compatible embedding provider healthcheck failed")
            return False


class MLXEmbeddingProvider:
    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
        max_length: int = 512,
        model_path: str = "",
        cache_dir: str = "",
    ):
        self.model_name = model_name
        self.dimensions = max(32, dimensions)
        self.max_length = max(16, max_length)
        self.model_path = (model_path or "").strip()
        self.cache_dir = (cache_dir or "").strip()
        self._model = None
        self._tokenizer = None

    def tokenize(self, text: str) -> list[str]:
        return tokenize_text(text)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model, tokenizer = self._load()
        inputs = self._tokenize_batch(tokenizer, texts)
        attention_mask = inputs.get("attention_mask")
        outputs = model(inputs["input_ids"], attention_mask=attention_mask)
        vectors = outputs.text_embeds.tolist()
        normalized = [_normalize_vector([float(value) for value in vector]) for vector in vectors]
        for vector in normalized:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}"
                )
        return normalized

    def healthcheck(self) -> bool:
        try:
            self.embed("healthcheck")
            return True
        except Exception:
            logger.exception("MLX embedding provider healthcheck failed")
            return False

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from mlx_embeddings.utils import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-embeddings is required for ARM_EMBEDDING_PROVIDER=mlx. "
                "Install with: pip install mlx-embeddings"
            ) from exc
        from pathlib import Path
        import os
        load_from: str
        if self.model_path:
            base = Path(self.model_path).expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
            subdir_name = self.model_name.replace("/", "_")
            local_model_dir = base / subdir_name
            if (local_model_dir / "config.json").exists():
                load_from = str(local_model_dir)
            else:
                logger.info(
                    "本地模型目录不存在或未就绪，正在下载到 {}: {}",
                    local_model_dir,
                    self.model_name,
                )
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=self.model_name,
                        local_dir=str(local_model_dir),
                        local_dir_use_symlinks=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"模型下载失败（{self.model_name} -> {local_model_dir}）: {exc}"
                    ) from exc
                load_from = str(local_model_dir)
        else:
            load_from = self.model_name
            if self.cache_dir:
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = self.cache_dir
                os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(self.cache_dir) / "hub")
        self._model, self._tokenizer = load(load_from)
        return self._model, self._tokenizer

    def _tokenize_batch(self, tokenizer, texts: list[str]):
        kwargs = {
            "return_tensors": "mlx",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
        }
        if callable(tokenizer):
            return tokenizer(texts, **kwargs)

        inner_tokenizer = getattr(tokenizer, "_tokenizer", None)
        if callable(inner_tokenizer):
            return inner_tokenizer(texts, **kwargs)

        raise RuntimeError(
            "Loaded MLX tokenizer is not callable and does not expose a callable _tokenizer"
        )


def build_embedding_provider(config: ARMConfig) -> EmbeddingProvider:
    provider_name = config.embedding_provider.strip().lower()
    try:
        if provider_name == "mlx":
            return MLXEmbeddingProvider(
                model_name=config.embedding_model,
                dimensions=config.vector_dimensions,
                max_length=config.embedding_max_length,
                model_path=config.embedding_model_path,
                cache_dir=config.embedding_cache_dir,
            )
        if provider_name in {"openai", "api", "cloud"}:
            return OpenAICompatibleEmbeddingProvider(
                base_url=config.embedding_base_url,
                model=config.embedding_model,
                dimensions=config.vector_dimensions,
                timeout=config.embedding_timeout,
                api_key=config.embedding_api_key,
            )
        raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
    except Exception:
        logger.exception(
            "Embedding provider initialisation failed; configure a valid provider (e.g. mlx or openai)."
        )
        raise
