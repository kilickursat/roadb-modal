"""Modal GPU endpoint for Road B.

Deploy this file with GitHub Actions. It exposes a POST endpoint that receives
chat messages from the Hugging Face Space, runs Qwen GGUF through llama.cpp on
Modal GPU, and returns parsed JSON fields.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import modal
from pydantic import BaseModel

APP_NAME = "roadb-qwen-llamacpp"
MODEL_REPO_ID = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILENAME = "Qwen3.5-9B-Q4_K_M.gguf"

volume = modal.Volume.from_name("roadb-qwen-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("build-essential", "cmake", "git", "curl")
    .pip_install(
        "huggingface-hub>=0.27.0",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.0",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
    .pip_install(
        "llama-cpp-python>=0.3.8",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
)

app = modal.App(APP_NAME)
_LLM = None


class GenerateRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 850
    temperature: float = 0.78
    top_p: float = 0.92
    seed: int = -1
    token: Optional[str] = None


def normalize_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
        normalized.append({"role": "system", "content": "\n\n".join(system_parts)})
    normalized.extend(body)
    return normalized


def _decode_json_string(value: str) -> str:
    try:
        return json.loads('"' + value + '"')
    except Exception:
        return value.replace('\\"', '"').replace("\\n", "\n")


def extract_json(text: str) -> Dict[str, Any]:
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
        except Exception:
            pass

    fields: Dict[str, Any] = {}
    for match in re.finditer(
        r'"([^"\\]+)"\s*:\s*"((?:\\.|[^"\\])*)"',
        cleaned,
        flags=re.DOTALL,
    ):
        key = match.group(1).strip()
        val = _decode_json_string(match.group(2)).strip()
        if key and val:
            fields[key] = val

    if fields:
        fields["_raw"] = cleaned
        fields["_partial_json"] = True
        return fields

    return {"_raw": cleaned, "_parse_failed": True}


def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir="/models",
    )

    try:
        volume.commit()
    except Exception:
        pass

    _LLM = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1,
        n_batch=512,
        verbose=False,
    )
    return _LLM


@app.function(
    image=image,
    gpu="L40S",
    timeout=900,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("roadb-modal-token")],
)
@modal.fastapi_endpoint(method="POST")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    expected = os.environ.get("ROADB_MODAL_TOKEN", "").strip()
    if expected and req.token != expected:
        return {"ok": False, "error": "Unauthorized Road B Modal request."}

    llm = get_llm()
    kwargs: Dict[str, Any] = {
        "messages": normalize_chat_messages(req.messages),
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
    }
    if req.seed >= 0:
        kwargs["seed"] = req.seed

    try:
        out = llm.create_chat_completion(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        out = llm.create_chat_completion(**kwargs)

    content = out["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    return {
        "ok": True,
        "runtime": "Modal GPU + llama.cpp",
        "model": f"{MODEL_REPO_ID}/{MODEL_FILENAME}",
        "raw": content,
        "parsed": parsed,
    }
