# PhantomRAM — Roadmap

The shipped v0.3 prefetcher is a small, self-contained piece of what
PhantomRAM is trying to become. This file is the honest version of
"what I'd like to do next" — no guarantees on timing. Contributors
welcome.

---

## The core thesis

> Many workloads with deterministic access patterns are bottlenecked
> on DRAM capacity, not bandwidth or latency. If you can predict what
> memory they'll need a few tens of milliseconds ahead, you can serve
> that prediction from NVMe and make the DRAM/NVMe price gap disappear
> for those workloads.

LLM inference is the best first target because the access pattern is
*trivially* deterministic — layer 1, layer 2, layer 3 — and the compute
phase per layer (50–200 ms on CPU, 10–30 ms on GPU for 70B) gives
prefetch a real window to work in.

Game texture streaming is the second target. Camera motion is predictable
on sub-second timescales, and Vulkan sparse resources give us a clean
hook to swap texture tiles in and out of VRAM.

---

## What's shipped (v0.3)

- `libphantom_preload.so` — transparent `LD_PRELOAD` shim for GGUF models.
- GGUF parser with layer-aware readahead.
- `mincore()` frontier detection + background prefetch thread.
- Measured 1.91x model load, 2.27x prompt eval on Llama 3.1 70B.
- Unit tests and a synthetic benchmark.

## Near-term (the next push)

These are the things I think would move the needle without needing
anything exotic.

### 1. Format-agnostic mode
GGUF detection is hardcoded. A sequential-access heuristic (e.g., detect
sustained forward reads over a large file-backed mapping) would let
PhantomRAM help for safetensors, PyTorch `.pt` files, and any other
large-file workload.

### 2. Multi-NVMe striping
With two or more NVMe drives, PhantomRAM could issue parallel `readahead()`
calls across drives, multiplying effective bandwidth. This directly
attacks the steady-state token-generation bottleneck, the one place
v0.3 doesn't help.

### 3. Eviction-aware prefetching
When the model is larger than RAM, prefetching pages that will be
evicted before they're read is worse than useless. A smarter policy
would skip layers too far ahead given the observed eviction rate.

### 4. `madvise(MADV_SEQUENTIAL)` + `MADV_POPULATE_READ`
Lower-hanging fruit than it sounds. Kernel-level readahead is more
aggressive under `MADV_SEQUENTIAL`, and `MADV_POPULATE_READ` (5.14+)
can fault pages in synchronously without going through userspace. A
hybrid approach may beat the pure `readahead()` strategy.

### 5. Integration with llama.cpp directly
Instead of `LD_PRELOAD`, upstream a `--phantom-ram` flag that calls the
same hooks. Easier for users, easier to maintain, gives us a chance to
fix the two llama.cpp issues we hit:
- Default context window allocates a huge KV cache (40 GB for 70B).
- `CPU_REPACK` allocates 31 GB of anonymous memory you can't mmap around.

## Medium-term (interesting problems)

### v1 — Kernel module
A `phantom_ram.ko` LKM could register as a real page-fault handler
rather than hinting via `readahead()`. Benefits:
- Zero-copy DMA from NVMe into the page cache via `io_uring`.
- eBPF hook for custom prefetch policies.
- Activation recomputation for ML training (skip paging out weights
  the gradient will need again).

Estimated 3–5x speedup over v0.3 on steady-state token generation.
Requires actual kernel expertise, not just userspace syscalls.

### Game texture streaming
Same core idea, different workload. A Vulkan implicit layer that:
1. Intercepts `vkCreateImage` / `vkBindImageMemory` for sparse resources.
2. Uses camera pose + motion vectors to predict which tiles will be
   visible in the next few frames.
3. Streams tiles NVMe → system RAM → VRAM ahead of the draw call.

Target: "Cyberpunk at 4K Ultra on an 8 GB GPU without stuttering."
This was the original second use case and is still the one most likely
to turn into a useful consumer tool. **Not yet validated with numbers.**

### v2 — CXL FPGA memory expansion
An open-hardware PCIe Gen4 FPGA board with SODIMM slots. Attach 256–512 GB
of recycled DDR4 to any server for ~$200 in parts. Not software: this
is the "fix the oligopoly by routing around it" piece. Completely
separate from the software effort but shares the thesis.

### v3 — Distributed KV cache
LLM serving bottleneck is often KV cache, not weights. A network-attached
KV pool over RDMA/DPDK would let small-memory machines handle long
contexts by offloading KV to a pooled RAM tier.

---

## What got cut (and why)

This project had a v0.2 that used `userfaultfd` to intercept faults and
serve from a private buffer pool. I shipped that, measured it, and
deleted it. The postmortem:

- `userfaultfd` does work, but only for `MAP_ANONYMOUS` regions. For
  file-backed GGUF maps, you have to create a *second* mapping that
  shadows llama.cpp's original, which doubles memory and makes 70B
  models OOM on 28 GB machines.
- The second buffer pool was also redundant: the kernel already has a
  page cache that does eviction correctly. Reinventing it was pure
  complexity with no benefit.
- v0.3 is 10% of the code of v0.2 and beats it on every metric.

The lesson I'm taking forward: **the kernel is usually smarter than I
am. Hint, don't replace.** Future phases that do add kernel-side code
(v1 LKM) need to justify it with something `readahead()` fundamentally
can't do.

---

## How to help

Open an issue or a PR. The areas that most need help:

| Area                               | Needed skills               |
|------------------------------------|-----------------------------|
| Multi-NVMe striping                | Linux I/O, `io_uring`       |
| Vulkan sparse resources (gaming)   | Vulkan, graphics drivers    |
| v1 kernel module                   | Linux kernel dev            |
| Benchmarks on different hardware   | Any: you just need machines |
| Upstream to llama.cpp              | C++, llama.cpp familiarity  |

If you have Llama 70B and an NVMe, the single most useful thing you
can do right now is reproduce the v0.3 benchmark and report your
numbers.
