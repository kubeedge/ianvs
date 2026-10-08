# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Local evidence persistence and bounded HTTP communication."""

import hashlib
import ipaddress
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


def save_json(path, value):
    """Replace evidence atomically, including after an interrupted write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_loopback_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def http_json(url, payload=None, timeout=30, headers=None):
    if not is_loopback_url(url):
        raise ValueError("Benchmark HTTP communication is restricted to loopback")
    class NoRedirect(HTTPRedirectHandler):
        """Never forward local observations/credentials to a redirect target."""
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    # A system HTTP proxy must never receive model logs or local repair requests.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def stream_tokens(url, payload, timeout=30):
    """Measure first generated token at the client, not the response headers."""
    if not is_loopback_url(url):
        raise ValueError("Inference endpoint must be loopback")
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.monotonic()
    tokens, first, result = [], None, None
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=timeout) as response:
        for line in response:
            if time.monotonic() - start > timeout:
                raise TimeoutError("Streaming inference exceeded request budget")
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            if "token_ids" in event:
                if event["token_ids"] and first is None:
                    first = time.monotonic() - start
                tokens.extend(event["token_ids"])
            if event.get("done"):
                result = event
                break
    if result is None or first is None or not tokens:
        raise RuntimeError("Inference stream ended without tokens and completion")
    return {**result, "token_ids": tokens, "ttft_seconds": first,
            "client_seconds": time.monotonic() - start}
