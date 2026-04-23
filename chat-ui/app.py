import json
import os
from typing import Any

import httpx
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm-gptoss:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local-test-key")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")


def vllm_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
    }


def extract_text_field(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        text = value.get("text")
        if text:
            return str(text).strip()
        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def extract_message_parts(raw: dict[str, Any]) -> dict[str, str]:
    choices = raw.get("choices") or []
    if not choices:
        return {"content": "", "reasoning": "", "answer": ""}

    message = (choices[0] or {}).get("message") or {}

    content = extract_text_field(message.get("content"))
    reasoning = extract_text_field(message.get("reasoning"))

    answer = content or reasoning or ""
    return {
        "content": content,
        "reasoning": reasoning,
        "answer": answer,
    }


@app.get("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME})


@app.get("/api/models")
def models():
    try:
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{VLLM_BASE_URL}/models", headers=vllm_headers())
            resp.raise_for_status()
            return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/chat")
def chat():
    data: dict[str, Any] = request.get_json(silent=True) or {}

    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers=vllm_headers(),
                json=payload,
            )
            resp.raise_for_status()

            raw = resp.json()
            parsed = extract_message_parts(raw)

            return jsonify({
                "ok": True,
                "model": MODEL_NAME,
                "answer": parsed["answer"],
                "content": parsed["content"],
                "reasoning": parsed["reasoning"],
                "raw": raw,
            })

    except httpx.HTTPStatusError as e:
        detail = e.response.text
        return jsonify({
            "ok": False,
            "error": f"HTTP {e.response.status_code}",
            "detail": detail,
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.post("/api/chat/stream")
def chat_stream():
    data: dict[str, Any] = request.get_json(silent=True) or {}

    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    @stream_with_context
    def generate():
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)

        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream(
                    "POST",
                    f"{VLLM_BASE_URL}/chat/completions",
                    headers=vllm_headers(),
                    json=payload,
                ) as resp:
                    resp.raise_for_status()

                    for line in resp.iter_lines():
                        if not line:
                            continue

                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="ignore")

                        if not line.startswith("data: "):
                            continue

                        data_str = line[len("data: "):].strip()

                        if data_str == "[DONE]":
                            yield "event: done\ndata: {}\n\n"
                            return

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue

                        delta = (choices[0] or {}).get("delta") or {}
                        finish_reason = (choices[0] or {}).get("finish_reason")

                        content_piece = extract_text_field(delta.get("content"))
                        reasoning_piece = extract_text_field(delta.get("reasoning"))

                        if content_piece:
                            yield f"event: content\ndata: {json.dumps({'text': content_piece}, ensure_ascii=False)}\n\n"

                        if reasoning_piece:
                            yield f"event: reasoning\ndata: {json.dumps({'text': reasoning_piece}, ensure_ascii=False)}\n\n"

                        if finish_reason is not None:
                            yield f"event: meta\ndata: {json.dumps({'finish_reason': finish_reason}, ensure_ascii=False)}\n\n"

                    yield "event: done\ndata: {}\n\n"

        except httpx.HTTPStatusError as e:
            detail = e.response.text
            yield f"event: error\ndata: {json.dumps({'error': f'HTTP {e.response.status_code}', 'detail': detail}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream")