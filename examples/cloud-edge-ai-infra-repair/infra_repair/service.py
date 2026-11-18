# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Loopback-only real model service used by the WSL process backend."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import time

from .io_utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.4)
    parser.add_argument("--host", default="127.0.1.1")
    args = parser.parse_args()
    import torch  # Heavy dependencies are confined to the service process.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(args.model):
        raise FileNotFoundError("Configured model directory does not exist: " + args.model)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    if not 0 < args.gpu_memory_fraction <= 0.4:
        raise ValueError("GPU allocation budget must be positive and at most 40 percent")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=getattr(torch, args.dtype),
    ).to(args.device).eval()

    class Handler(BaseHTTPRequestHandler):
        """A bounded inference endpoint; no administrative API."""

        def reply(self, code, body):
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):  # pylint: disable=invalid-name
            self.reply(200 if self.path == "/health" else 404,
                       {"ready": True, "pid": os.getpid()})

        def do_POST(self):  # pylint: disable=invalid-name
            if self.path not in ("/generate", "/generate-stream"):
                self.reply(404, {"error": "unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("request size outside bounds")
                body = json.loads(self.rfile.read(length))
                prompt = body["prompt"]
                if not isinstance(prompt, str) or len(prompt) > 2048:
                    raise ValueError("invalid prompt")
                encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
                start = time.monotonic()
                streamer = None
                if self.path == "/generate-stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    handler = self
                    class Streamer:
                        """Transformers sends the prompt first; never count it as output."""
                        prompt_pending = True
                        def put(self, values):
                            if self.prompt_pending:
                                self.prompt_pending = False
                                return
                            event = {"token_ids": values.reshape(-1).tolist()}
                            handler.wfile.write((json.dumps(event) + "\n").encode())
                            handler.wfile.flush()
                        def end(self):
                            return
                    streamer = Streamer()
                with torch.inference_mode():
                    generated = model.generate(**encoded, do_sample=False, max_new_tokens=16,
                                               pad_token_id=tokenizer.eos_token_id, streamer=streamer)
                tokens = generated[0, encoded["input_ids"].shape[1]:].tolist()
                if streamer is not None:
                    event = {"done": True, "text": tokenizer.decode(tokens),
                             "server_seconds": time.monotonic() - start}
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
                    return
                self.reply(200, {"token_ids": tokens, "text": tokenizer.decode(tokens),
                                 "server_seconds": time.monotonic() - start})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})

        def log_message(self, format, *values):  # pylint: disable=redefined-builtin
            return

    # WSL mirrored networking can redirect 127.0.0.1 to Windows loopback0.
    server = HTTPServer((args.host, 0), Handler)
    save_json(args.ready, {"port": server.server_port, "pid": os.getpid()})
    print("Model loaded; HTTP service ready", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
