"""
N730 Inference Engine
=====================
Project Bombakla — Phase 4

The part that actually generates text.

Wires together:
  - HuggingFace tokenizer  (encode prompt → token ids)
  - N730Scheduler          (stream layers from .n730 file)
  - Transformer forward()  (attention + MLP per layer)
  - Sampler                (logits → next token)

The model never fully lives in memory.
Each forward pass streams all 198 layers through VRAM one at a time.
KV cache lives in RAM between tokens.

Usage:
    python inference.py --model deepseek-r1-1.5b.n730 --prompt "What is 2+2?"
    python inference.py --model deepseek-r1-1.5b.n730 --prompt "Explain gravity" --max-tokens 200
    python inference.py --model deepseek-r1-1.5b.n730 --interactive
"""

import argparse
import json
import math
import struct
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

# ─── Try loading the scheduler (C++ accelerated if available) ─────────────────
try:
    from scheduler_cpp import N730Scheduler
    SCHEDULER_BACKEND = "C++"
except ImportError:
    from scheduler import N730Scheduler
    SCHEDULER_BACKEND = "Python"


# ─── Transformer math ─────────────────────────────────────────────────────────
# Pure numpy — runs on CPU, feeds the GT 730 VRAM one layer at a time.
# When the CUDA kernel (Phase 5) is ready, these functions get replaced
# with GPU calls. The interface stays identical.

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray],
               eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x    = (x - mean) / np.sqrt(var + eps)
    x    = x * weight
    if bias is not None:
        x = x + bias
    return x

def rope_freqs(seq_len: int, head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Rotary position embedding frequencies."""
    theta = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos   = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(pos, theta)
    return np.concatenate([freqs, freqs], axis=-1)

def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

def apply_rope(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    cos = np.cos(freqs)[np.newaxis, :, np.newaxis, :]
    sin = np.sin(freqs)[np.newaxis, :, np.newaxis, :]
    # x shape: (1, seq_len, n_heads, head_dim)
    return x * cos + rotate_half(x) * sin


# ─── KV Cache ────────────────────────────────────────────────────────────────

class KVCache:
    """
    Key-value cache for attention — lives in RAM between tokens.
    Growing cache: each new token appends K/V for all layers.
    Max context window before eviction.
    """
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 max_seq: int = 512):
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.max_seq  = max_seq
        # k_cache[layer] = (seq_len, n_heads, head_dim)
        self.k_cache = [None] * n_layers
        self.v_cache = [None] * n_layers
        self.seq_len = 0

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray):
        """Append new K/V for this layer."""
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = np.concatenate([self.k_cache[layer_idx], k], axis=0)
            self.v_cache[layer_idx] = np.concatenate([self.v_cache[layer_idx], v], axis=0)

        # Update global seq len from first layer
        if layer_idx == 0:
            self.seq_len = self.k_cache[0].shape[0]

    def get(self, layer_idx: int):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
        self.seq_len = 0

    @property
    def ram_mb(self) -> float:
        total = 0
        for k in self.k_cache:
            if k is not None:
                total += k.nbytes
        for v in self.v_cache:
            if v is not None:
                total += v.nbytes
        return total / (1024**2)


# ─── Model config ─────────────────────────────────────────────────────────────

class ModelConfig:
    """
    Architecture config for DeepSeek-R1-Distill-Qwen-1.5B.
    Parsed from the model's config.json if available, otherwise hardcoded.
    """
    def __init__(self, config_path: Optional[str] = None):
        # Defaults for deepseek-r1-distill-qwen-1.5b
        self.hidden_size       = 1536
        self.num_hidden_layers = 28
        self.num_attention_heads = 12
        self.num_kv_heads      = 2      # GQA: 2 KV heads, 12 query heads
        self.head_dim          = self.hidden_size // self.num_attention_heads  # 128
        self.intermediate_size = 8960
        self.vocab_size        = 151936
        self.max_position_embeddings = 131072
        self.rms_norm_eps      = 1e-6
        self.rope_theta        = 10000.0

        if config_path and Path(config_path).exists():
            self._load(config_path)

    def _load(self, path: str):
        with open(path) as f:
            cfg = json.load(f)
        self.hidden_size         = cfg.get("hidden_size", self.hidden_size)
        self.num_hidden_layers   = cfg.get("num_hidden_layers", self.num_hidden_layers)
        self.num_attention_heads = cfg.get("num_attention_heads", self.num_attention_heads)
        self.num_kv_heads        = cfg.get("num_key_value_heads", self.num_kv_heads)
        self.intermediate_size   = cfg.get("intermediate_size", self.intermediate_size)
        self.vocab_size          = cfg.get("vocab_size", self.vocab_size)
        self.rms_norm_eps        = cfg.get("rms_norm_eps", self.rms_norm_eps)
        self.rope_theta          = cfg.get("rope_theta", self.rope_theta)
        self.head_dim            = self.hidden_size // self.num_attention_heads
        print(f"  Config loaded from {path}")


# ─── Layer weight bundle ──────────────────────────────────────────────────────

class TransformerLayerWeights:
    """
    Groups all weight matrices for one transformer layer.
    The scheduler streams these in; we unpack them here.
    """
    __slots__ = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "input_norm", "post_attn_norm",
    ]
    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, None)


# ─── Transformer forward pass ─────────────────────────────────────────────────

class N730Transformer:
    """
    Streaming transformer that runs one full forward pass per token.
    Layers are streamed from .n730 via the scheduler — never fully in memory.
    """

    def __init__(self, model_path: str, config: ModelConfig, prefetch: int = 6):
        self.config     = config
        self.model_path = model_path
        self.prefetch   = prefetch

        # Scheduler instance (persistent — file handle stays open)
        print(f"  Loading scheduler ({SCHEDULER_BACKEND} backend)...")
        self.scheduler = N730Scheduler(model_path, prefetch=prefetch)

        # Build layer name → seek table index map
        self._build_layer_map()

        # Static buffers we load once at startup
        self.embed_weight  = None   # vocab_size × hidden_size
        self.lm_head_weight = None  # hidden_size × vocab_size
        self._load_static_weights()

        # KV cache
        self.kv_cache = KVCache(
            n_layers  = config.num_hidden_layers,
            n_heads   = config.num_kv_heads,
            head_dim  = config.head_dim,
            max_seq   = 2048,
        )

        # RoPE frequencies (precomputed)
        self.rope_freqs = rope_freqs(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
        )

    def _build_layer_map(self):
        """Map layer names to their index in the seek table for O(1) lookup."""
        self._name_to_idx = {}
        for i, entry in enumerate(self.scheduler._seek_table):
            self._name_to_idx[entry["layer_name"]] = i

    def _load_static_weights(self):
        """
        Load embedding table and LM head once — they're needed every token.
        Everything else streams per-layer.
        """
        print(f"  Loading static weights (embed + lm_head)...")
        t0 = time.perf_counter()

        embed_idx = self._name_to_idx.get("model.embed_tokens.weight")
        lmhead_idx = self._name_to_idx.get("lm_head.weight")

        if embed_idx is not None:
            buf = self.scheduler.get_layer(embed_idx)
            self.embed_weight = buf.weights   # (vocab_size, hidden_size)
            print(f"    embed_tokens: {self.embed_weight.shape}  [{buf.precision}]")

        if lmhead_idx is not None:
            buf = self.scheduler.get_layer(lmhead_idx)
            self.lm_head_weight = buf.weights  # (vocab_size, hidden_size)
            print(f"    lm_head:      {self.lm_head_weight.shape}  [{buf.precision}]")

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"    Done in {elapsed:.0f}ms")

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Token ids → embedding vectors."""
        if self.embed_weight is None:
            # Fallback: random embeddings (for testing without real weights)
            rng = np.random.default_rng(42)
            return rng.normal(0, 0.02, (len(token_ids), self.config.hidden_size)).astype(np.float32)
        return self.embed_weight[token_ids]  # (seq_len, hidden_size)

    def _rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.config.rms_norm_eps)
        return (x / rms) * weight

    def _attention(self, x: np.ndarray, layer_idx: int,
                   weights: TransformerLayerWeights) -> np.ndarray:
        """
        Grouped Query Attention (GQA) forward pass.
        x: (seq_len, hidden_size)
        Returns: (seq_len, hidden_size)
        """
        cfg   = self.config
        seq   = x.shape[0]
        d     = cfg.head_dim

        # Infer heads from actual weight shapes (handles synthetic + real models)
        if weights.q_proj is not None:
            h = max(1, weights.q_proj.shape[0] // d)
        else:
            h = cfg.num_attention_heads
        if weights.k_proj is not None:
            kv_h = max(1, weights.k_proj.shape[0] // d)
        else:
            kv_h = cfg.num_kv_heads
        groups = max(1, h // kv_h)

        # Project Q, K, V
        if weights.q_proj is not None:
            q = x @ weights.q_proj.T  # (seq, h*d)
        else:
            q = np.zeros((seq, h * d), dtype=np.float32)

        if weights.k_proj is not None:
            k = x @ weights.k_proj.T  # (seq, kv_h*d)
        else:
            k = np.zeros((seq, kv_h * d), dtype=np.float32)

        if weights.v_proj is not None:
            v = x @ weights.v_proj.T  # (seq, kv_h*d)
        else:
            v = np.zeros((seq, kv_h * d), dtype=np.float32)

        # Reshape for multi-head: (seq, heads, head_dim)
        # Guard: re-infer head counts from actual projected sizes
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        d_actual = d if q_dim % d == 0 else q_dim
        h_actual = q_dim // d_actual
        kv_h_actual = k_dim // d_actual
        groups = max(1, h_actual // kv_h_actual)
        h, kv_h, d = h_actual, kv_h_actual, d_actual

        q = q.reshape(seq, h,    d)
        k = k.reshape(seq, kv_h, d)
        v = v.reshape(seq, kv_h, d)

        # Apply RoPE
        pos_start = self.kv_cache.seq_len
        freqs = self.rope_freqs[pos_start:pos_start + seq]
        # freqs: (seq, head_dim) — apply to each head
        cos_f = np.cos(freqs)  # (seq, d)
        sin_f = np.sin(freqs)  # (seq, d)

        def apply_rope_to(x_heads):
            # x_heads: (seq, n_heads, head_dim)
            half = d // 2
            x1, x2 = x_heads[..., :half], x_heads[..., half:]
            rot    = np.concatenate([-x2, x1], axis=-1)
            c = cos_f[:, np.newaxis, :]  # (seq, 1, d)
            s = sin_f[:, np.newaxis, :]
            return x_heads * c + rot * s

        q = apply_rope_to(q)
        k = apply_rope_to(k)

        # Update KV cache
        self.kv_cache.update(layer_idx, k, v)
        k_full, v_full = self.kv_cache.get(layer_idx)
        total_seq = k_full.shape[0]

        # Expand KV heads to match Q heads (GQA)
        # k_full: (total_seq, kv_h, d) → (total_seq, h, d)
        k_exp = np.repeat(k_full, groups, axis=1)
        v_exp = np.repeat(v_full, groups, axis=1)

        # Attention scores: (seq, h, total_seq)
        # q: (seq, h, d), k_exp: (total_seq, h, d)
        scale = 1.0 / math.sqrt(d)
        # Efficient batched dot: for each head
        attn = np.einsum("shd,thd->sht", q, k_exp) * scale  # (seq, h, total_seq)

        # Causal mask (only for prefill; single token needs no mask)
        if seq > 1:
            mask = np.triu(np.full((seq, total_seq), -1e9), k=total_seq - seq + 1)
            attn += mask[:, np.newaxis, :]

        attn = softmax(attn, axis=-1)  # (seq, h, total_seq)

        # Weighted sum: (seq, h, d)
        out = np.einsum("sht,thd->shd", attn, v_exp)

        # Merge heads: (seq, h*d)
        out = out.reshape(seq, h * d)

        # Output projection
        if weights.o_proj is not None:
            out = out @ weights.o_proj.T
        return out

    def _mlp(self, x: np.ndarray, weights: TransformerLayerWeights) -> np.ndarray:
        """SwiGLU MLP: gate_proj * silu(up_proj) → down_proj."""
        if weights.gate_proj is None:
            return x

        gate = x @ weights.gate_proj.T   # (seq, intermediate)
        up   = x @ weights.up_proj.T     # (seq, intermediate)

        # SiLU activation on gate
        gate = gate * (1.0 / (1.0 + np.exp(-gate)))  # silu

        hidden = gate * up
        return hidden @ weights.down_proj.T

    def forward_one_token(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Full transformer forward pass for one token (or prompt prefill).
        Streams all 28 transformer layers from .n730.
        Returns logits: (vocab_size,)
        """
        cfg = self.config
        x = self.embed(token_ids)   # (seq_len, hidden_size)
        seq_len = x.shape[0]

        # Collect weights for each transformer layer from the stream
        # We need 7 weight matrices per layer: q,k,v,o,gate,up,down
        # Plus 2 norm weights (not in .n730 yet — use identity fallback)
        layer_weights_store = {}

        # Build expected weight names for all transformer layers
        needed = set()
        for i in range(cfg.num_hidden_layers):
            for w in ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]:
                needed.add(f"model.layers.{i}.self_attn.{w}.weight"
                           if "proj" in w and w != "gate_proj" and w != "up_proj" and w != "down_proj"
                           else f"model.layers.{i}.mlp.{w}.weight"
                           if w in ("gate_proj", "up_proj", "down_proj")
                           else f"model.layers.{i}.self_attn.{w}.weight")

        # Fix: build correct names
        needed = set()
        for i in range(cfg.num_hidden_layers):
            needed.add(f"model.layers.{i}.self_attn.q_proj.weight")
            needed.add(f"model.layers.{i}.self_attn.k_proj.weight")
            needed.add(f"model.layers.{i}.self_attn.v_proj.weight")
            needed.add(f"model.layers.{i}.self_attn.o_proj.weight")
            needed.add(f"model.layers.{i}.mlp.gate_proj.weight")
            needed.add(f"model.layers.{i}.mlp.up_proj.weight")
            needed.add(f"model.layers.{i}.mlp.down_proj.weight")

        # Stream all layers, collect the ones we need
        t_stream_start = time.perf_counter()
        for layer_buf in self.scheduler.stream():
            name = layer_buf.layer_name
            if name in needed:
                layer_weights_store[name] = layer_buf.weights

        stream_ms = (time.perf_counter() - t_stream_start) * 1000

        # Now run the transformer forward pass with collected weights
        for i in range(cfg.num_hidden_layers):
            w = TransformerLayerWeights()
            w.q_proj    = layer_weights_store.get(f"model.layers.{i}.self_attn.q_proj.weight")
            w.k_proj    = layer_weights_store.get(f"model.layers.{i}.self_attn.k_proj.weight")
            w.v_proj    = layer_weights_store.get(f"model.layers.{i}.self_attn.v_proj.weight")
            w.o_proj    = layer_weights_store.get(f"model.layers.{i}.self_attn.o_proj.weight")
            w.gate_proj = layer_weights_store.get(f"model.layers.{i}.mlp.gate_proj.weight")
            w.up_proj   = layer_weights_store.get(f"model.layers.{i}.mlp.up_proj.weight")
            w.down_proj = layer_weights_store.get(f"model.layers.{i}.mlp.down_proj.weight")

            residual = x

            # Input norm (identity fallback — norm weights match actual hidden dim)
            actual_hidden = x.shape[-1]
            norm_w = np.ones(actual_hidden, dtype=np.float32)
            x_normed = self._rms_norm(x, norm_w)

            # Attention
            attn_out = self._attention(x_normed, i, w)
            x = residual + attn_out

            residual = x
            x_normed = self._rms_norm(x, norm_w)

            # MLP
            mlp_out = self._mlp(x_normed, w)
            x = residual + mlp_out

        # Final norm
        norm_w = np.ones(x.shape[-1], dtype=np.float32)
        x = self._rms_norm(x, norm_w)

        # LM head: project last token position to vocab
        last = x[-1]   # (hidden_size,)
        if self.lm_head_weight is not None:
            logits = last @ self.lm_head_weight.T   # (vocab_size,)
        else:
            # Fallback: random logits (proves pipeline works before real weights)
            rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
            logits = rng.normal(0, 1, cfg.vocab_size).astype(np.float32)

        return logits, stream_ms


# ─── Sampling ────────────────────────────────────────────────────────────────

def sample_token(logits: np.ndarray, temperature: float = 0.7,
                 top_p: float = 0.9) -> int:
    """Top-p (nucleus) sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Sort descending
    sorted_idx  = np.argsort(probs)[::-1]
    sorted_prob = probs[sorted_idx]
    cumsum      = np.cumsum(sorted_prob)

    # Keep tokens up to top_p mass
    cutoff = np.searchsorted(cumsum, top_p) + 1
    sorted_idx  = sorted_idx[:cutoff]
    sorted_prob = sorted_prob[:cutoff]
    sorted_prob /= sorted_prob.sum()

    return int(np.random.choice(sorted_idx, p=sorted_prob))


# ─── Tokenizer wrapper ────────────────────────────────────────────────────────

class N730Tokenizer:
    """Thin wrapper around HuggingFace tokenizer."""

    def __init__(self, model_id: str = "deepseek-ai/deepseek-r1-distill-qwen-1.5b",
                 local_path: Optional[str] = None):
        try:
            from transformers import AutoTokenizer
            path = local_path or model_id
            print(f"  Loading tokenizer from {path}...")
            self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.eos_id = self.tok.eos_token_id
            print(f"  Vocab size: {self.tok.vocab_size}  EOS: {self.eos_id}")
        except Exception as e:
            print(f"  Tokenizer load failed: {e}")
            print(f"  Falling back to character-level tokenizer (demo mode)")
            self.tok = None
            self.eos_id = 0

    def encode(self, text: str) -> np.ndarray:
        if self.tok:
            return np.array(self.tok.encode(text), dtype=np.int32)
        return np.array([ord(c) % 1000 for c in text], dtype=np.int32)

    def decode(self, token_id: int) -> str:
        if self.tok:
            return self.tok.decode([token_id], skip_special_tokens=True)
        return chr(token_id % 128) if token_id > 31 else " "

    def decode_batch(self, token_ids: list) -> str:
        if self.tok:
            return self.tok.decode(token_ids, skip_special_tokens=True)
        return "".join(self.decode(t) for t in token_ids)


# ─── Main generation loop ─────────────────────────────────────────────────────

class N730Generator:
    """
    End-to-end text generator.
    prompt → tokens → transformer (streamed) → sample → decode → print
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None, prefetch: int = 6):
        print(f"\n╔══════════════════════════════════════════╗")
        print(f"║   N730 Inference Engine  •  Bombakla      ║")
        print(f"╚══════════════════════════════════════════╝\n")

        self.config = ModelConfig(config_path)
        print(f"  Architecture : {self.config.num_hidden_layers}L  "
              f"h={self.config.hidden_size}  "
              f"heads={self.config.num_attention_heads}(q)/{self.config.num_kv_heads}(kv)")

        self.transformer = N730Transformer(model_path, self.config, prefetch=prefetch)
        self.tokenizer   = N730Tokenizer(
            model_id="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
            local_path=tokenizer_path,
        )

    def generate(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stream: bool = True) -> str:
        """Generate text token by token."""
        self.transformer.kv_cache.clear()

        print(f"\n  Prompt: {prompt!r}")
        print(f"  Generating (max {max_tokens} tokens, T={temperature})...\n")
        print(f"  {'─'*60}")
        if stream:
            print(f"  ", end="", flush=True)

        token_ids = self.tokenizer.encode(prompt)
        generated = []
        total_tokens = 0
        t_start = time.perf_counter()

        # Prefill: process full prompt as one pass
        logits, stream_ms = self.transformer.forward_one_token(token_ids)
        next_token = sample_token(logits, temperature, top_p)
        generated.append(next_token)
        total_tokens += 1

        decoded = self.tokenizer.decode(next_token)
        if stream:
            print(decoded, end="", flush=True)

        prefill_ms = (time.perf_counter() - t_start) * 1000
        t_gen = time.perf_counter()

        # Autoregressive decode: one token at a time
        for _ in range(max_tokens - 1):
            if next_token == self.tokenizer.eos_id:
                break

            token_arr = np.array([next_token], dtype=np.int32)
            logits, stream_ms = self.transformer.forward_one_token(token_arr)
            next_token = sample_token(logits, temperature, top_p)
            generated.append(next_token)
            total_tokens += 1

            decoded = self.tokenizer.decode(next_token)
            if stream:
                print(decoded, end="", flush=True)

        elapsed = time.perf_counter() - t_start
        gen_elapsed = time.perf_counter() - t_gen
        tokens_per_sec = max(total_tokens - 1, 1) / max(gen_elapsed, 0.001)

        full_text = self.tokenizer.decode_batch(generated)

        print(f"\n  {'─'*60}")
        print(f"\n  Tokens generated : {total_tokens}")
        print(f"  Prefill time     : {prefill_ms:.0f}ms")
        print(f"  Speed            : {tokens_per_sec:.2f} tokens/sec")
        print(f"  KV cache RAM     : {self.transformer.kv_cache.ram_mb:.1f} MB")
        print(f"  Total time       : {elapsed:.1f}s\n")

        return full_text


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="N730 Inference Engine — Project Bombakla")
    parser.add_argument("--model",       type=str, required=True, help="Path to .n730 model file")
    parser.add_argument("--config",      type=str, default=None,  help="Path to config.json (optional)")
    parser.add_argument("--tokenizer",   type=str, default=None,  help="Local tokenizer path (default: HF hub)")
    parser.add_argument("--prompt",      type=str, default="Hello, I am",  help="Input prompt")
    parser.add_argument("--max-tokens",  type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p",       type=float, default=0.9)
    parser.add_argument("--prefetch",    type=int, default=6)
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--no-stream",   action="store_true",  help="Don't print tokens as they generate")
    args = parser.parse_args()

    gen = N730Generator(
        model_path     = args.model,
        config_path    = args.config,
        tokenizer_path = args.tokenizer,
        prefetch       = args.prefetch,
    )

    if args.interactive:
        print("\n  Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("  You: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                gen.generate(
                    prompt,
                    max_tokens  = args.max_tokens,
                    temperature = args.temperature,
                    top_p       = args.top_p,
                    stream      = not args.no_stream,
                )
            except KeyboardInterrupt:
                print("\n  Interrupted.")
                break
    else:
        gen.generate(
            args.prompt,
            max_tokens  = args.max_tokens,
            temperature = args.temperature,
            top_p       = args.top_p,
            stream      = not args.no_stream,
        )


if __name__ == "__main__":
    main()