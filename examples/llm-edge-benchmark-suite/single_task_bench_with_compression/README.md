# LLM edge benchmark with compression

This example quantizes a Qwen model on CPU and runs inference on three bundled
test prompts. Run all commands from the Ianvs repository root.

## Install the Python dependencies

The commands below use Python 3.11 on Ubuntu x86-64. Install a C/C++
compiler and CMake to build the llama.cpp Python bindings for CPU inference.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt resources/third_party/sedna-0.6.0.1-py3-none-any.whl
python -m pip install -e .
CMAKE_ARGS="-DGGML_NATIVE=OFF -DLLAMA_CURL=OFF" CMAKE_BUILD_PARALLEL_LEVEL=4 \
  python -m pip install \
  --no-binary=llama-cpp-python \
  -r examples/llm-edge-benchmark-suite/single_task_bench_with_compression/requirements.txt
```

## Download the FP16 model and quantizer

Use the FP16 GGUF as the input to Q8_0 quantization. The download is about
1.27 GB, and quantization writes a separate GGUF alongside it.

```bash
mkdir -p models/qwen llama.cpp
curl --fail --location \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf \
  --output models/qwen/qwen2.5-0.5b-instruct-fp16.gguf
curl --fail --location \
  https://github.com/ggml-org/llama.cpp/releases/download/b12110/llama-b12110-bin-ubuntu-x64.tar.gz \
  --output /tmp/ianvs-llama-cpp.tar.gz
tar --extract --gzip --file /tmp/ianvs-llama-cpp.tar.gz \
  --no-same-owner --strip-components=1 --directory llama.cpp
```

On other platforms, install a compatible build of
[llama.cpp](https://github.com/ggml-org/llama.cpp/releases) and set
`llama_quantize_path` in `testalgorithms/algorithm.yaml` to the executable path.

## Run the benchmark

The train and test JSONL files are included under `dataset/`. The training
step uses the pretrained weights as-is.

```bash
ianvs -f examples/llm-edge-benchmark-suite/single_task_bench_with_compression/benchmarkingjob.yaml
```

The log shows the generated Q8_0 GGUF path loaded for inference. Ianvs prints
a ranking table and writes results under `workspace/benchmarkingjob/`.
The table reports request latency in milliseconds, throughput in requests per
second, and process memory usage in bytes. The `prefill_latency` column reports
the time in milliseconds from the start of a request to the first streamed chunk
containing generated text. It includes prompt processing and the wait for that text.

The bundled prompts are for checking that the pipeline runs.
