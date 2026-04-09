# Contributing to PhantomRAM

This project is in the "poke at it" phase. The v0.3 prefetcher works
and is measured, but there's a lot of low-hanging fruit on the
[roadmap](ROADMAP.md) and I'd rather have more eyes on it than fewer.

## Getting set up

Requirements: Linux 5.7+ (or WSL2 Ubuntu 22.04+), `gcc`, `make`,
`pthread`. No external libraries... That's intentional.

```bash
git clone https://github.com/MR-GL00M/phantomram.git
cd phantomram
make
make test
```

If `make test` reports `6/6 passed`, you're good.

For end-to-end reproduction of the 70B benchmark, see
`scripts/setup.sh` — it clones llama.cpp at a pinned commit and builds
it for you.

## Where the code lives

- `src/phantom_prefetch_cache.{h,c}` — the prefetcher engine. If you're
  touching prefetch policy, frontier detection, or eviction-awareness,
  this is the file.
- `src/phantom_preload.c` — the `LD_PRELOAD` shim. `mmap`/`munmap`/`mlock`
  interception, GGUF detection, layer map handoff.
- `src/gguf_loader.{h,c}` — minimal GGUF header parser. Tensor offsets,
  sizes, layer IDs.
- `tests/test_pcache.c` — unit tests. Add new ones for any non-trivial
  change to the engine.
- `benchmarks/mmap_bench.c` — the synthetic benchmark used in
  `docs/benchmark-results-v0.3.md`.

## What's useful right now

If you have a few hours:

1. **Reproduce the benchmark on different hardware.** The v0.3 numbers
   are from a single machine (i7-14700KF, 28 GB WSL2 RAM, NVMe SSD).
   Different CPU/NVMe combinations will tell us how much of the win is
   hardware-dependent. Open an issue with your results.
2. **Try format-agnostic prefetching.** `src/phantom_preload.c` hardcodes
   GGUF detection. A sequential-access heuristic that kicks in for any
   large file-backed mmap would unlock safetensors, `.pt` files, etc.
3. **Eviction-aware prefetch.** When the model is larger than RAM,
   prefetching too far ahead is actively harmful. The engine has the
   data it needs via `mincore()`; it just needs a smarter policy.

If you want a bigger project, see `ROADMAP.md`.

## Code style

- C: Linux kernel style — `snake_case`, 4-space indent, braces on the
  same line as functions. Don't introduce external dependencies
  without a very good reason.
- Keep diagnostic output going to stderr, gated on `PHANTOM_VERBOSE`.
- Add a test for anything non-trivial. `test_pcache.c` is small on
  purpose; keep it that way.

## Submitting changes

1. Open an issue first for anything larger than a bug fix. I'd rather
   spend 10 minutes agreeing on the approach than have you spend 10
   hours on a PR I can't merge.
2. Run `make clean && make test` before you push.
3. Include benchmark numbers for any change that could affect
   performance. Use the same drop-caches discipline as
   `docs/benchmark-results-v0.3.md`.

## License

MIT. By contributing you agree your contributions are licensed under MIT.
