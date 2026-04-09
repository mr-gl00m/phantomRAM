# PhantomRAM

**Actually download more RAM.**

A transparent page cache prefetcher that makes NVMe behave like RAM. Zero extra memory. Drop-in `LD_PRELOAD`. Measured **1.91x faster model load** and **2.27x faster prompt eval** on Llama 3.1 70B (Q4_K_M) in 28 GB of RAM.

---

## Why this exists

The "you can't download more RAM" joke always bugged me. Then RAM prices skyrocketed and the joke stopped being funny. Suddenly something that used to be basic consumer hardware was priced like a luxury good. So I built the thing the joke said couldn't exist: a prefetcher that uses the kernel's own page cache as a buffer pool, warming it ahead of your application's reads so NVMe latency stops mattering for anything that fits in RAM.

It's not a RAM *replacement*. It's a way to make the RAM you already have behave like more RAM by eliminating the cold-cache tax on everything you load. Thus: PhantomRAM.

---

## What it does

When you `mmap()` a large GGUF model file, the Linux kernel loads pages
on demand, the first time your inference engine touches a page, it stalls
on an NVMe read. Kernel readahead helps a little (128 KB at a time by
default), but it doesn't know what a "transformer layer" is.

PhantomRAM sits in front of `mmap()` as an `LD_PRELOAD` shim and:

1. **Detects GGUF model files** by magic number when they're mapped.
2. **Parses the GGUF header** to build a layer-to-byte-range map
   (80 layers and 724 tensors for Llama 70B).
3. **Fires an immediate 1 GB burst of `readahead()`** so model loading
   overlaps with llama.cpp's metadata processing.
4. **Runs a background thread** that polls page residency via `mincore()`
   and pushes `readahead()` ahead of the application's access frontier,
   layer by layer.
5. **Suppresses `mlock()`** on the mapped region so the kernel can freely
   evict cold pages under memory pressure.

This is important, man. The kernel's page cache *is* the buffer pool. PhantomRAM never allocates memory of its own: it just makes the kernel's existing cache smarter about the access pattern.

---

## Measured results (v0.3, 2026-04-08)

Full data, methodology, and failure modes in
[`docs/benchmark-results-v0.3.md`](docs/benchmark-results-v0.3.md).

### Llama 3.1 70B Instruct Q4_K_M (39.6 GB model, 28 GB RAM)

Real inference via `llama-simple`, page cache dropped before each run.

| Metric             | Baseline       | PhantomRAM     | Improvement      |
|--------------------|----------------|----------------|------------------|
| Wall time          | 18m 13s        | 16m 46s        | **8% faster**    |
| **Model load**     | 432.8 s        | 226.9 s        | **1.91x faster** |
| **Prompt eval**    | 194.5 s        | 85.8 s         | **2.27x faster** |
| Prompt tok/s       | 0.07           | 0.15           | **2.14x**        |
| Token generation   | ~753 s (4 tok) | ~753 s (4 tok) | Same             |
| Extra memory       | 0              | 0              | **Zero**         |

### Llama 3.1 8B Instruct Q4_K_M (4.6 GB, fits in RAM)

Synthetic mmap sequential read benchmark.

| Metric     | Baseline  | PhantomRAM | Improvement   |
|------------|-----------|------------|---------------|
| Throughput | 0.19 GB/s | 0.39 GB/s  | **2.02x**     |
| Total time | 23.98 s   | 11.84 s    | **2.02x**     |

---

## Where it *doesn't* help (yet)

Being upfront about the limits so nobody shows up expecting magic:

- **Steady-state token generation on larger-than-RAM models.** Every
  generated token touches all 40 GB of weights through 28 GB of RAM.
  The kernel is already evicting and paging optimally for the sequential
  access pattern; NVMe bandwidth is the bottleneck, not prefetch
  intelligence. PhantomRAM adds no meaningful improvement here.
- **Synthetic sequential reads larger than RAM.** The benchmark touches
  one byte per page and moves on, creating artificially high eviction
  pressure. PhantomRAM's prefetcher competes with the reader for NVMe
  bandwidth and actually slows things down ~10% in this case. Real
  inference is fine because the compute phase gives prefetch a window
  to work in.
- **Anything that isn't GGUF.** The shim is GGUF-aware and ignores other
  mmaps. A format-agnostic mode is on the roadmap and is the path to
  the gaming use case.

**TL;DR: PhantomRAM wins big on cold-start and first-token latency. It doesn't change the steady-state tok/s ceiling imposed by your NVMe's bandwidth.** See the roadmap for how we plan to attack that.

---

## Quick start

### Requirements
- Linux 5.7+ (or WSL2 Ubuntu 22.04+)
- `gcc`, `make`, `pthread` (all standard)
- A GGUF model file (llama.cpp-compatible)

### Build
```bash
git clone https://github.com/MR-GL00M/phantomram.git
cd phantomram
make
```

Output: `build/libphantom_preload.so`, `build/test_pcache`, `build/mmap_bench`.

### Run the tests
```bash
make test
```

Expected: `6/6 passed`.

### Use it with llama.cpp
`scripts/setup.sh` will clone and build llama.cpp alongside PhantomRAM
if you don't already have it. It produces `llama.cpp/build/bin/llama-simple`.
```bash
./scripts/setup.sh    # one-time: builds phantomram + llama.cpp

LLAMA_SIMPLE=./llama.cpp/build/bin/llama-simple
MODEL=/path/to/model.gguf

# Baseline (no PhantomRAM) — drop page cache for a fair cold start
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
"$LLAMA_SIMPLE" -m "$MODEL" -n 16 -c 256 --no-warmup --no-repack

# With PhantomRAM
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
LD_PRELOAD=./build/libphantom_preload.so PHANTOM_VERBOSE=1 \
    "$LLAMA_SIMPLE" -m "$MODEL" -n 16 -c 256 --no-warmup --no-repack
```

The `-c 256 --no-warmup --no-repack` flags aren't cosmetic,
see [`docs/benchmark-results-v0.3.md`](docs/benchmark-results-v0.3.md#issues-encountered)
for why each one matters when running 70B on a 28 GB machine.

### Run the synthetic benchmark
```bash
# Baseline
./build/mmap_bench /path/to/model.gguf

# With PhantomRAM
LD_PRELOAD=./build/libphantom_preload.so PHANTOM_VERBOSE=1 \
    ./build/mmap_bench /path/to/model.gguf
```

### Reproducing the 70B result
See [`docs/benchmark-results-v0.3.md`](docs/benchmark-results-v0.3.md#reproduction)
for the full reproduction recipe, including the drop-caches / swapoff
discipline needed to avoid contamination between runs, and the exact
llama.cpp flags required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  llama.cpp (unmodified)                                      │
│    mmap(model.gguf) ─── normal file-backed mapping ───────┐  │
└────────────────────────────────────────────────────────────┼─┘
                                                              │
              LD_PRELOAD intercepts mmap()                    │
                         │                                    │
                         ▼                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  libphantom_preload.so (this project)                       │
│                                                             │
│    ┌──────────────┐  ┌────────────────┐  ┌──────────────┐   │
│    │ GGUF parser  │  │  1 GB initial  │  │  Background  │   │
│    │              │  │     burst      │  │  prefetcher  │   │
│    │ 80 layers,   │  │ readahead() on │  │              │   │
│    │ 724 tensors, │  │ mmap — runs    │  │ Polls        │   │
│    │ byte ranges  │  │ during         │  │ mincore(),   │   │
│    │              │  │ metadata load  │  │ advances     │   │
│    └──────┬───────┘  └────────────────┘  │ frontier,    │   │
│           │                               │ layer-aware │   │
│           └───────── layer map ──────────►│ readahead() │   │
│                                           └──────┬──────┘   │
└──────────────────────────────────────────────────┼──────────┘
                                                    │
                         readahead() syscalls       │
                                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Linux kernel page cache (the buffer pool — not ours)       │
│    Pages get filled from NVMe ahead of llama.cpp's reads.   │
└─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  NVMe SSD                                                   │
└─────────────────────────────────────────────────────────────┘
```

**Key design decision:** v0.2 of this project used `userfaultfd` to
intercept page faults and serve from a private buffer pool. That
doubled memory because llama.cpp's own mmap was still live alongside
our shadow mapping, and it OOM'd on anything larger than 28 GB. v0.3
deletes all that and lets the kernel do what it's already good at: the prefetcher only hints.

---

## Project layout

```
.
├── README.md                  You are here
├── ROADMAP.md                 Future directions (multi-NVMe, game textures, etc.)
├── LICENSE                    MIT
├── CONTRIBUTING.md            How to poke at it
├── Makefile                   Single make command builds everything
├── src/
│   ├── phantom_prefetch_cache.{h,c}   Core prefetcher engine
│   ├── phantom_preload.c              LD_PRELOAD mmap/mlock/munmap hooks
│   └── gguf_loader.{h,c}              Minimal GGUF header parser
├── tests/
│   └── test_pcache.c                  6 unit tests for the prefetcher
├── benchmarks/
│   └── mmap_bench.c                   Synthetic sequential-read benchmark
├── docs/
│   └── benchmark-results-v0.3.md      Full measured data + failure modes
└── scripts/
    └── setup.sh                       Clones + builds llama.cpp for reproduction
```

---

## Environment variables

| Variable          | Default | Description                                          |
|-------------------|---------|------------------------------------------------------|
| `PHANTOM_VERBOSE` | `0`     | Set to `1` for per-region and shutdown stats on stderr |

Tuning knobs like readahead window, initial burst, and poll interval
are currently compile-time constants in
`src/phantom_prefetch_cache.h` (`PCACHE_READAHEAD_PAGES`,
`PCACHE_INITIAL_BURST`, `PCACHE_POLL_INTERVAL_US`). Making them
runtime-configurable is on the roadmap.

---

## Why "PhantomRAM"

The application thinks it has 40 GB of memory and acts accordingly.
Physically, it has 28 GB of RAM plus an NVMe. The kernel and
PhantomRAM conspire to make the difference invisible during the
phases of inference where there's compute slack to hide I/O behind.
The extra RAM isn't really there... it's a goddamn phantom.

---

## Support

SIGIL is free and public domain (CC0).

If you find this useful, consider supporting development:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/mr_gl00m)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub_Sponsors-EA4AAA?style=for-the-badge&logo=github&logoColor=white)](https://ko-fi.com/mr_gl00m)

**Crypto:**
- BTC: `bc1qnedeq3dr2dmlwgmw2mr5mtpxh45uhl395prr0d`
- ETH: `0x1bCbBa9854dA4Fc1Cb95997D5f42006055282e3c`
- SOL: `3Wm8wS93UpG2CrZsMWHSspJh7M5gQ6NXBbgLHDFXmAdQ`

---

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project is very much in
the "poke at it" phase. Kernel people, llama.cpp people, and anyone
who has a theory about why steady-state eviction pressure is so
hard to beat, come argue with me.
