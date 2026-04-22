"""
LLMProvider
===========
Provider-agnostic LLM client supporting OpenAI, Azure OpenAI, Ollama, and a
mock backend for offline testing.

Usage
-----
from ChaptersFinancial._platform.providers.llm import LLMProvider

provider = LLMProvider()                       # reads provider_config.yaml + env
result   = provider.complete_json(prompt, schema_dict)
embed    = provider.embed(["text one", "text two"])

Environment variables (take precedence over provider_config.yaml)
------------------------------------------------------------------
LLM_PROVIDER          openai | azure | ollama | openrouter | mock
OPENAI_API_KEY
OPENAI_MODEL
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION
OLLAMA_BASE_URL
OLLAMA_MODEL
OPENROUTER_API_KEY
OPENROUTER_MODEL
OPENROUTER_SITE_URL   (optional)
OPENROUTER_SITE_NAME  (optional)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import datetime
from pathlib import Path
from typing import Any

import yaml

# Strip <think>...</think> blocks produced by reasoning models (e.g. Qwen3)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Extract first JSON object or array from a response that may contain extra text
_JSON_EXTRACT_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "provider_config.yaml"


def _build_mock_stub(schema: dict) -> dict:
    """Build a minimal valid JSON stub that satisfies schema's required fields."""
    _type_defaults: dict = {
        "string": "",
        "number": 0.0,
        "integer": 0,
        "boolean": False,
        "array": [],
        "object": {},
        "null": None,
    }
    props = schema.get("properties", {})
    required = schema.get("required", list(props.keys()))
    stub: dict = {}
    for key in required:
        prop_def = props.get(key, {})
        ptype = prop_def.get("type", "string")
        if isinstance(ptype, list):
            ptype = next((t for t in ptype if t != "null"), "string")
        if ptype == "object":
            stub[key] = _build_mock_stub(prop_def)
        elif ptype == "array":
            stub[key] = []
        elif ptype == "string" and "enum" in prop_def:
            stub[key] = prop_def["enum"][0]
        else:
            stub[key] = _type_defaults.get(ptype, "")
    return stub


def _load_config() -> dict:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


class LLMProvider:
    """
    Thin, dependency-light wrapper so callers never import openai/httpx directly.
    Only the selected backend's library is imported at runtime.
    """

    def __init__(self, provider: str | None = None):
        cfg = _load_config()
        self._cfg = cfg["llm"]
        self._provider = (
            provider
            or os.getenv("LLM_PROVIDER")
            or self._cfg.get("default_provider", "openai")
        ).lower()

        # Cost/token tracking (accumulated for this instance)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0

        # On-disk response cache
        cache_dir = self._cfg.get("cache_dir", "data_fin/cache_llm")
        # Resolve relative to repo root (two levels up from this file)
        repo_root = Path(__file__).parent.parent.parent.parent
        self._cache_dir = repo_root / cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._max_retries = int(self._cfg.get("max_retries", 3))
        self._backoff_base = float(self._cfg.get("retry_backoff_base", 2.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete_json(
        self,
        prompt: str,
        schema: dict | None = None,
        *,
        system: str = "You are a precise financial data extraction assistant. "
                      "Always respond with valid JSON matching the provided schema.",
        temperature: float | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        """
        Complete a prompt and return a parsed JSON dict.
        Retries ONLY on JSON parse errors (reflexion loop).
        Network / timeout errors are raised immediately — retrying a timed-out
        inference just wastes 3× the time and can crash the Ollama process.
        Caches successful responses to disk.
        """
        cache_key = self._cache_key(prompt, schema, model or self._default_model())
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        temp = temperature if temperature is not None else float(
            self._cfg.get(self._provider, {}).get("temperature", 0.0)
        )
        mdl = model or self._default_model()
        req_timeout = timeout if timeout is not None else float(
            self._cfg.get(self._provider, {}).get("timeout", 300)
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                raw = self._call_chat(
                    prompt, system=system, temperature=temp, model=mdl,
                    timeout=req_timeout, schema=schema,
                )
                parsed = self._parse_json(raw)
                self._save_cache(cache_key, parsed)
                return parsed
            except (json.JSONDecodeError, ValueError) as exc:
                # Reflexion retry: only for malformed JSON responses
                last_error = exc
                prompt = (
                    f"{prompt}\n\n"
                    f"[RETRY {attempt + 1}] Your previous response was not valid JSON. "
                    f"Error: {exc}. Respond ONLY with a valid JSON object."
                )
                time.sleep(self._backoff_base ** attempt)
            except Exception as exc:
                # Network, timeout, auth errors — fail immediately (no retry)
                raise RuntimeError(
                    f"LLMProvider.complete_json: {type(exc).__name__}: {exc}"
                ) from exc

        raise RuntimeError(
            f"LLMProvider.complete_json failed after {self._max_retries} JSON retries: {last_error}"
        )

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """
        Return a list of embedding vectors (one per input text).
        Falls back to a zero-vector for mock/offline mode.
        """
        if self._provider == "mock":
            dim = int(self._cfg.get("openai", {}).get("embedding_dim", 1536))
            return [[0.0] * dim for _ in texts]

        if self._provider in ("openai", "azure"):
            return self._openai_embed(texts, model=model)

        if self._provider == "openrouter":
            # OpenRouter exposes an OpenAI-compatible embeddings endpoint;
            # use requests (with ssl_verify=False support) instead of OpenAI SDK
            return self._openrouter_embed(texts, model=model)

        if self._provider == "ollama":
            return self._ollama_embed(texts, model=model)

        raise ValueError(f"Unsupported provider for embed: {self._provider}")

    # ------------------------------------------------------------------
    # Cost / usage
    # ------------------------------------------------------------------
    def usage_summary(self) -> dict:
        return {
            "provider": self._provider,
            "total_calls": self._total_calls,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        }

    # ------------------------------------------------------------------
    # Internal: routing
    # ------------------------------------------------------------------
    def _call_chat(
        self, prompt: str, *, system: str, temperature: float, model: str,
        timeout: float = 300, schema: dict | None = None,
    ) -> str:
        if self._provider == "mock":
            return self._mock_response(prompt, schema=schema)
        if self._provider == "openai":
            return self._openai_chat(prompt, system=system, temperature=temperature, model=model)
        if self._provider == "azure":
            return self._azure_chat(prompt, system=system, temperature=temperature, model=model)
        if self._provider == "ollama":
            return self._ollama_chat(
                prompt, system=system, temperature=temperature, model=model, timeout=timeout
            )
        if self._provider == "openrouter":
            return self._openrouter_chat(
                prompt, system=system, temperature=temperature, model=model, timeout=timeout
            )
        raise ValueError(f"Unknown provider: {self._provider}")

    def _default_model(self) -> str:
        # Only look up the env var that matches the active provider to avoid
        # cross-contamination (e.g. OPENAI_MODEL overriding an Ollama run).
        provider_env = {
            "openai":      "OPENAI_MODEL",
            "azure":       "AZURE_OPENAI_DEPLOYMENT",
            "ollama":      "OLLAMA_MODEL",
            "openrouter":  "OPENROUTER_MODEL",
        }
        env_model = os.getenv(provider_env.get(self._provider, ""))
        return (
            env_model
            or self._cfg.get(self._provider, {}).get("model", "gpt-4o-mini")
        )

    # ------------------------------------------------------------------
    # Internal: OpenAI
    # ------------------------------------------------------------------
    def _openai_chat(self, prompt: str, *, system: str, temperature: float, model: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install openai>=1.12 to use the openai provider") from exc

        api_key = os.getenv("OPENAI_API_KEY") or self._cfg.get("openai", {}).get("api_key")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        if usage:
            self._total_prompt_tokens += usage.prompt_tokens
            self._total_completion_tokens += usage.completion_tokens
        self._total_calls += 1
        return response.choices[0].message.content or ""

    def _openai_embed(self, texts: list[str], *, model: str | None) -> list[list[float]]:
        try:
            from openai import OpenAI  # type: ignore
            import httpx as _httpx  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install openai>=1.12 to use the openai provider") from exc

        api_key = os.getenv("OPENAI_API_KEY") or self._cfg.get("openai", {}).get("api_key")
        embed_model = model or "text-embedding-3-small"
        ssl_verify_raw = os.getenv("OPENROUTER_SSL_VERIFY", "true").strip().lower()
        ssl_verify = ssl_verify_raw not in ("0", "false", "no")
        http_client = _httpx.Client(verify=ssl_verify) if not ssl_verify else None
        client = OpenAI(api_key=api_key, http_client=http_client)
        response = client.embeddings.create(input=texts, model=embed_model)
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Internal: Azure OpenAI
    # ------------------------------------------------------------------
    def _azure_chat(self, prompt: str, *, system: str, temperature: float, model: str) -> str:
        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install openai>=1.12 to use the azure provider") from exc

        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model)
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        if usage:
            self._total_prompt_tokens += usage.prompt_tokens
            self._total_completion_tokens += usage.completion_tokens
        self._total_calls += 1
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Internal: Ollama
    # ------------------------------------------------------------------
    def _ollama_chat(
        self, prompt: str, *, system: str, temperature: float, model: str,
        timeout: float = 300,
    ) -> str:
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install httpx to use the ollama provider") from exc

        base_url = os.getenv("OLLAMA_BASE_URL") or self._cfg.get("ollama", {}).get(
            "base_url", "http://localhost:11434"
        )
        mdl = os.getenv("OLLAMA_MODEL") or model

        # For Qwen3/Qwen3.5 models prepend /no_think to disable chain-of-thought
        # reasoning in the system prompt (saves tokens and avoids think-tag leakage)
        if "qwen3" in mdl.lower():
            system = "/no_think\n" + system

        # Cloud-routed models (e.g. deepseek-v3.2:cloud) cannot use Ollama's local
        # grammar-constrained JSON sampling – it forces Ollama to buffer and
        # resample locally, making every call extremely slow.  Skip format:json
        # for any model with a :cloud suffix; _parse_json handles extraction.
        is_cloud_model = mdl.lower().endswith(":cloud")

        # For cloud-routed models use streaming: Ollama forwards each token as
        # it arrives from the cloud, so the read timeout applies per-chunk
        # (milliseconds) rather than to the entire buffered response (minutes).
        # For local models keep stream=False for simplicity.
        payload = {
            "model": mdl,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "stream": is_cloud_model,
            "options": {"temperature": temperature},
        }
        if is_cloud_model:
            # Disable chain-of-thought for cloud models (DeepSeek, etc.)
            # Thinking phase streams many empty-content chunks that keep the
            # connection alive but delay the actual JSON response by 30-120s.
            payload["think"] = False
        else:
            payload["format"] = "json"

        # Per-chunk read timeout for streaming; full-response read timeout for local.
        http_timeout = httpx.Timeout(connect=5.0, read=timeout, write=30.0, pool=5.0)

        if is_cloud_model:
            content_parts: list[str] = []
            try:
                with httpx.stream(
                    "POST", f"{base_url}/api/chat", json=payload, timeout=http_timeout
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content_parts.append(chunk.get("message", {}).get("content", ""))
                        if chunk.get("done"):
                            break
            except (httpx.RemoteProtocolError, httpx.ReadError):
                # Cloud proxy dropped the connection mid-stream.
                # Use whatever chunks arrived — JSON is usually complete by then.
                if not content_parts:
                    raise
            self._total_calls += 1
            return "".join(content_parts)
        else:
            resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=http_timeout)
            resp.raise_for_status()
            self._total_calls += 1
            return resp.json()["message"]["content"]

    # ------------------------------------------------------------------
    # Internal: OpenRouter
    # ------------------------------------------------------------------
    def _openrouter_chat(
        self, prompt: str, *, system: str, temperature: float, model: str,
        timeout: float = 120,
    ) -> str:
        try:
            import requests  # type: ignore
            import urllib3   # type: ignore
        except ImportError as exc:
            raise ImportError("pip install requests to use the openrouter provider") from exc

        api_key = os.getenv("OPENROUTER_API_KEY") or self._cfg.get("openrouter", {}).get("api_key")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        base_url = (
            os.getenv("OPENROUTER_BASE_URL")
            or self._cfg.get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1")
        )
        mdl = os.getenv("OPENROUTER_MODEL") or model

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        site_url = os.getenv("OPENROUTER_SITE_URL") or self._cfg.get("openrouter", {}).get("site_url", "")
        site_name = os.getenv("OPENROUTER_SITE_NAME") or self._cfg.get("openrouter", {}).get("site_name", "")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        # Google AI Studio (gemma models) rejects the "system" role with
        # "Developer instruction is not enabled". Merge system into user instead.
        _is_google_model = mdl.startswith("google/") or "gemma" in mdl.lower()
        if _is_google_model and system:
            messages = [{"role": "user", "content": f"{system}\n\n{prompt}"}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ]

        payload = {
            "model": mdl,
            "messages": messages,
            "temperature": temperature,
        }
        # response_format=json_object is only supported by OpenAI-family models.
        # Free / open-weight models (e.g. llama, mistral, gemma) return plain text;
        # _parse_json handles JSON extraction from free-text responses.
        if mdl.startswith(("openai/", "anthropic/")) and not mdl.endswith(":free"):
            payload["response_format"] = {"type": "json_object"}

        # Respect OPENROUTER_SSL_VERIFY env var (set to 'false' behind corporate proxies
        # that inject a self-signed certificate into the TLS chain).
        ssl_verify_raw = os.getenv("OPENROUTER_SSL_VERIFY", "true").strip().lower()
        ssl_verify = ssl_verify_raw not in ("0", "false", "no")
        if not ssl_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        session = requests.Session()
        session.verify = ssl_verify

        # Retry on 429 (free-tier rate limit) with exponential back-off.
        # Free models on OpenRouter have a per-minute quota; start with 15 s.
        max_retries = 5
        base_wait = 15.0
        for attempt in range(max_retries):
            resp = session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            if resp.status_code == 429:
                wait = base_wait * (2 ** attempt)   # 15, 30, 60, 120, 240 s
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("X-Ratelimit-Reset-Requests")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text[:200]
                if attempt < max_retries - 1:
                    print(f"[OpenRouter] 429 rate-limited ({err_body}), waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})…")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"[OpenRouter] Rate limit exceeded after {max_retries} retries: {err_body}")
            if not resp.ok:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text[:400]
                raise RuntimeError(f"[OpenRouter] HTTP {resp.status_code}: {err_body}")
            break

        data = resp.json()

        usage = data.get("usage", {})
        self._total_prompt_tokens += usage.get("prompt_tokens", 0)
        self._total_completion_tokens += usage.get("completion_tokens", 0)
        self._total_calls += 1

        return data["choices"][0]["message"]["content"] or ""

    def _openrouter_embed(self, texts: list[str], *, model: str | None) -> list[list[float]]:
        """Embeddings via OpenRouter using requests (supports ssl_verify=False)."""
        try:
            import requests  # type: ignore
            import urllib3   # type: ignore
        except ImportError as exc:
            raise ImportError("pip install requests") from exc

        api_key = os.getenv("OPENROUTER_API_KEY") or self._cfg.get("openrouter", {}).get("api_key")
        base_url = (
            os.getenv("OPENROUTER_BASE_URL")
            or self._cfg.get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1")
        )
        embed_model = model or os.getenv("OPENROUTER_EMBED_MODEL") or "text-embedding-3-small"

        ssl_verify_raw = os.getenv("OPENROUTER_SSL_VERIFY", "true").strip().lower()
        ssl_verify = ssl_verify_raw not in ("0", "false", "no")
        if not ssl_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        session = requests.Session()
        session.verify = ssl_verify
        resp = session.post(
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"input": texts, "model": embed_model},
            timeout=60,
        )
        if not resp.ok:
            # Fall back to zero vectors if embedding endpoint unavailable
            dim = int(self._cfg.get("openai", {}).get("embedding_dim", 1536))
            return [[0.0] * dim for _ in texts]
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]

    def _ollama_embed(self, texts: list[str], *, model: str | None) -> list[list[float]]:
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install httpx to use the ollama provider") from exc

        base_url = os.getenv("OLLAMA_BASE_URL") or self._cfg.get("ollama", {}).get(
            "base_url", "http://localhost:11434"
        )
        mdl = model or os.getenv("OLLAMA_MODEL") or self._cfg.get("ollama", {}).get(
            "model", "llama3.1:latest"
        )
        results = []
        for text in texts:
            resp = httpx.post(
                f"{base_url}/api/embeddings",
                json={"model": mdl, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results

    # ------------------------------------------------------------------
    # Internal: Mock
    # ------------------------------------------------------------------
    def _mock_response(self, prompt: str, schema: dict | None = None) -> str:
        # Return a minimal valid JSON stub so tests don't need real API keys.
        # When a schema is provided build the minimal required-field stub.
        if schema:
            return json.dumps(_build_mock_stub(schema))

        fixture_dir = Path(
            self._cfg.get("mock", {}).get("fixture_dir", "data_fin/samples/mock_responses")
        )
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        fixture_file = fixture_dir / f"{key}.json"
        if fixture_file.exists():
            return fixture_file.read_text()
        return json.dumps({"mock": True, "entities": [], "events": [], "items": []})

    # ------------------------------------------------------------------
    # Internal: cache
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(prompt: str, schema: dict | None, model: str) -> str:
        payload = json.dumps({"prompt": prompt, "schema": schema, "model": model}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        self._cache_path(key).write_text(json.dumps(data, indent=2))

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        # 1. Strip <think>...</think> blocks (Qwen3 reasoning traces)
        raw = _THINK_TAG_RE.sub("", raw).strip()
        # 2. Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        # 3. Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 4. Extract the first JSON object/array embedded in any surrounding prose
        match = _JSON_EXTRACT_RE.search(raw)
        if match:
            return json.loads(match.group(1))
        raise json.JSONDecodeError("No JSON object found in response", raw, 0)
