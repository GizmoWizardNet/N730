# N730

N730 is an experimental AI inference runtime built to run modern Large Language Models on extremely low-end hardware such as the NVIDIA GT 730.

Instead of loading an entire model into VRAM at once, N730 streams quantized transformer layers dynamically during inference, allowing models far larger than the GPU's memory capacity to run on legacy hardware.

The project combines:
- Layer streaming
- Dynamic mixed-precision quantization
- Asynchronous prefetch scheduling
- Native AVX2/C++ acceleration
- Runtime dequantization
- KV-cache-based autoregressive inference

N730 is currently capable of:
- Converting HuggingFace transformer models into the `.n730` format
- Streaming 198+ transformer layers from disk in real time
- Running autoregressive transformer inference
- Dynamically dequantizing INT2 / INT4 / INT8 / FP16 weights
- Executing inference on hardware as old as the GT 730

---

## Why?

Modern AI systems assume access to:
- Massive VRAM
- High-end GPUs
- Expensive hardware

N730 explores a different idea:

> What if transformer models could be virtualized like memory?

Instead of treating VRAM as the hard limit, N730 treats disk, RAM, and GPU memory as a streaming hierarchy — continuously moving layers through the GPU only when needed.

The goal is not to outperform modern inference engines.

The goal is to make AI inference possible on hardware that should theoretically be incapable of running it.

---

## Architecture

N730 currently consists of:

### `converter.py`
Converts HuggingFace transformer models into the `.n730` streaming format.

Features:
- Layer sensitivity profiling
- Mixed-precision quantization
- Big-endian packed layer storage
- O(1) seek-table-based layer access

---

### `scheduler.py`
The streaming runtime responsible for:
- Layer prefetching
- Disk scheduling
- RAM staging
- Runtime dequantization
- Async pipeline management

---

### `n730core.cpp`
Native AVX2-accelerated runtime core.

Handles:
- INT2 / INT4 / INT8 dequantization
- Persistent model file handles
- Zero-copy layer reads
- Streaming layer unpacking

Compiled as:
- `n730core.dll` (Windows)
- `n730core.so` (Linux)

---

### `inference.py`
The actual transformer inference engine.

Implements:
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- RMSNorm
- KV cache
- Top-p sampling
- Streaming autoregressive generation

---

## Current Status

N730 is currently in active experimental development.

Working:
- Native C++ runtime
- Streaming scheduler
- Quantized layer loading
- KV cache
- Autoregressive token generation
- HuggingFace tokenizer integration

In progress:
- Numerical correctness validation
- CUDA backend for GT 730
- Optimized transformer kernels
- Better scheduler overlap
- Full GPU inference path

---

## Example

```bash
python inference.py \
  --model deepseek-r1-1.5b.n730 \
  --prompt "What is 2+2?"
```

Project Goals
 - Make AI inference accessible on low-end hardware
 - Explore streamed transformer execution
 - Research memory-virtualized inference systems
 - Build a fully open-source experimental runtime
 
## Disclaimer

N730 is an experimental research project.

Performance, correctness, and stability are still under active development.

This project is not affiliated with NVIDIA, DeepSeek, HuggingFace, or any model provider.

## License: MIT License
