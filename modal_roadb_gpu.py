"""
Modal GPU endpoint for Road B: The Other Screen.

This endpoint runs:
- Qwen3.5-9B GGUF
- llama-cpp-python
- Modal GPU

The Hugging Face Space calls this endpoint for model inference.

Expected POST body from the HF Space:

{
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "max_tokens": 850,
  "temperature": 0.78,
  "top_p": 0.92,
  "seed": -1,
  "token": "same value as ROADB_MODAL_TOKEN"
}
"""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
from typing import Any, Dict, List

import modal


APP_NAME = "roadb-qwen-llamacpp"
MODEL_REPO_ID = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "Qwen3.5-9B-Q4_K_M.gguf")

MODEL_DIR = "/models/qwen35-9b"
VOLUME_NAME = "roadb-qwen-cache"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "curl",
        "ca-certificates",
    )
    .run_commands(
        "python -m pip install --no-cache-dir --upgrade pip",
        "python -m pip install --no-cache-dir 'fastapi[standard]>=0.115.0' 'huggingface-hub>=0.27.0' 'hf-transfer>=0.1.9'",
        "python -m pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 'llama-cpp-python>=0.3.8'",
    )
)

app = modal.App(APP_NAME)

_LLM = None
_LOAD_LOCK = threading.Lock()


def _as_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except Exception:
        result = default

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)

    return result


def _as_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except Exception:
        result = default

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)

    return result


def normalize_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Qwen/llama.cpp chat templates expect one system message at the beginning.

    This merges all system messages into a single first system message and keeps
    only user/assistant turns afterward.
    """

    system_parts: List[str] = []
    body: List[Dict[str, str]] = []

    for message in messages:
        role = str(message.get("role", "user") or "user").strip().lower()
        content = str(message.get("content", "") or "").strip()

        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            body.append({"role": role, "content": content})
        else:
            body.append({"role": "user", "content": content})

    normalized: List[Dict[str, str]] = []

    if system_parts:
        normalized.append(
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            }
        )

    normalized.extend(body)
    return normalized


def _parse_json_string_literal(value: str) -> str:
    try:
        return json.loads('"' + value + '"')
    except Exception:
        return value.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")


def _extract_partial_json_fields(text: str) -> Dict[str, Any]:
    """Recover completed string fields from truncated JSON."""

    fields: Dict[str, Any] = {}

    for match in re.finditer(
        r'"([^"\\]+)"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    ):
        key = match.group(1).strip()
        value = _parse_json_string_literal(match.group(2)).strip()

        if key and value:
            fields[key] = value

    return fields


def extract_json(text: str) -> Dict[str, Any]:
    """Parse complete or partially truncated JSON from the model output."""

    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            value.setdefault("_raw", cleaned)
            return value
        return {"value": value, "_raw": cleaned}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            if isinstance(value, dict):
                value.setdefault("_raw", cleaned)
                return value
            return {"value": value, "_raw": cleaned}
        except Exception:
            pass

    partial = _extract_partial_json_fields(cleaned)
    if partial:
        partial["_raw"] = cleaned
        partial["_partial_json"] = True
        return partial

    return {
        "_raw": cleaned,
        "_parse_failed": True,
    }


def get_llm():
    """Load Qwen GGUF once per Modal container."""

    global _LLM

    if _LLM is not None:
        return _LLM

    with _LOAD_LOCK:
        if _LLM is not None:
            return _LLM

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        hf_token = os.environ.get("HF_TOKEN", "").strip() or None

        print(f"[roadb] Loading model {MODEL_REPO_ID}/{MODEL_FILENAME}")
        print(f"[roadb] HF_TOKEN present: {bool(hf_token)}")

        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=MODEL_DIR,
            token=hf_token,
        )

        print(f"[roadb] Model path: {model_path}")

        try:
            volume.commit()
            print("[roadb] Modal volume committed.")
        except Exception as exc:
            print(f"[roadb] Volume commit skipped/failed: {exc!r}")

        _LLM = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_batch=512,
            verbose=False,
        )

        print("[roadb] llama.cpp model loaded.")
        return _LLM


@app.function(
    image=image,
    gpu="L40S",
    timeout=1200,
    startup_timeout=1200,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=1200,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("roadb-modal-token")],
)
@modal.fastapi_endpoint(method="POST")
def generate(req: dict) -> Dict[str, Any]:
    """Generate a Road B response through Qwen GGUF + llama.cpp."""

    try:
        expected_token = os.environ.get("ROADB_MODAL_TOKEN", "").strip()

        if not expected_token:
            return {
                "ok": False,
                "error": "ROADB_MODAL_TOKEN is missing from Modal secret roadb-modal-token.",
            }

        received_token = str(req.get("token", "") or "").strip()

        if received_token != expected_token:
            return {
                "ok": False,
                "error": "Unauthorized Modal request. MODAL_QWEN_TOKEN does not match ROADB_MODAL_TOKEN.",
            }

        messages = req.get("messages", [])

        if not isinstance(messages, list) or not messages:
            return {
                "ok": False,
                "error": "Request must include a non-empty messages list.",
            }

        max_tokens = _as_int(req.get("max_tokens", 850), 850, minimum=64, maximum=1600)
        temperature = _as_float(req.get("temperature", 0.78), 0.78, minimum=0.0, maximum=2.0)
        top_p = _as_float(req.get("top_p", 0.92), 0.92, minimum=0.01, maximum=1.0)
        seed = _as_int(req.get("seed", -1), -1)

        safe_messages = normalize_chat_messages(messages)

        llm = get_llm()

        call_kwargs: Dict[str, Any] = {
            "messages": safe_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if seed >= 0:
            call_kwargs["seed"] = seed

        try:
            output = llm.create_chat_completion(**call_kwargs)
        except TypeError:
            call_kwargs.pop("seed", None)
            output = llm.create_chat_completion(**call_kwargs)

        content = output["choices"][0]["message"]["content"]
        parsed = extract_json(content)

        return {
            "ok": True,
            "model": f"{MODEL_REPO_ID}/{MODEL_FILENAME}",
            "runtime": "Modal GPU + llama.cpp",
            "raw": content,
            "parsed": parsed,
            "usage": output.get("usage", {}),
        }

    except Exception as exc:
        print("[roadb] Modal endpoint exception:")
        print(traceback.format_exc())

        return {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc()[-4000:],
        }
