/*
 * PhantomRAM — Synthetic mmap sequential read benchmark.
 *
 * Simulates LLM inference by mmap'ing a large file and reading through
 * it sequentially in "layer" chunks, measuring page fault throughput.
 *
 * Run WITHOUT PhantomRAM (baseline):
 *   ./mmap_bench /path/to/model.gguf
 *
 * Run WITH PhantomRAM:
 *   LD_PRELOAD=../v0-userfaultfd/libphantom_preload.so ./mmap_bench /path/to/model.gguf
 *
 * The benchmark:
 *   1. Opens and mmap's the file (MAP_PRIVATE, file-backed)
 *   2. Drops all page cache for the file (posix_fadvise DONTNEED)
 *   3. Reads through the file sequentially in 2MB chunks, touching every page
 *   4. Reports throughput and timing per "layer" (configurable chunk size)
 *
 * This isolates page fault / readahead performance from inference compute.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* -------------------------------------------------------------------------- */
/* Configuration                                                               */
/* -------------------------------------------------------------------------- */

#define PAGE_SIZE       (4096)           /* System page size */
#define CHUNK_SIZE      (2 * 1024 * 1024) /* 2MB — matches PhantomRAM granularity */
#define DEFAULT_LAYER_MB 512             /* Default ~512MB per "layer" */
#define TOUCH_STRIDE    PAGE_SIZE        /* Touch every 4KB page */
#define DEFAULT_COMPUTE_MS 0             /* Simulated compute delay between layers */

/* -------------------------------------------------------------------------- */
/* Timing helpers                                                              */
/* -------------------------------------------------------------------------- */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* -------------------------------------------------------------------------- */
/* Page residency check via mincore                                            */
/* -------------------------------------------------------------------------- */

static size_t count_resident_pages(void *addr, size_t length)
{
    size_t num_pages = (length + PAGE_SIZE - 1) / PAGE_SIZE;
    unsigned char *vec = malloc(num_pages);
    if (!vec) return 0;

    if (mincore(addr, length, vec) != 0) {
        free(vec);
        return 0;
    }

    size_t resident = 0;
    for (size_t i = 0; i < num_pages; i++) {
        if (vec[i] & 1) resident++;
    }
    free(vec);
    return resident;
}

/* -------------------------------------------------------------------------- */
/* Main benchmark                                                              */
/* -------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file> [read_size_gb] [compute_ms] [layer_mb]\n", argv[0]);
        fprintf(stderr, "  file:         Large file to benchmark (e.g. model.gguf)\n");
        fprintf(stderr, "  read_size_gb: How many GB to read (default: entire file)\n");
        fprintf(stderr, "  compute_ms:   Simulated compute delay between layers (default: 0)\n");
        fprintf(stderr, "                Real 70B inference: ~50-200ms per layer\n");
        fprintf(stderr, "  layer_mb:     Size of each simulated layer in MB (default: 512)\n");
        return 1;
    }

    const char *filepath = argv[1];
    double read_limit_gb = 0;
    if (argc >= 3) read_limit_gb = atof(argv[2]);
    int compute_ms = DEFAULT_COMPUTE_MS;
    if (argc >= 4) compute_ms = atoi(argv[3]);
    int layer_mb = DEFAULT_LAYER_MB;
    if (argc >= 5) layer_mb = atoi(argv[4]);
    size_t layer_size = (size_t)layer_mb * 1024 * 1024;

    /* Open file */
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    size_t file_size = st.st_size;
    size_t read_size = file_size;
    if (read_limit_gb > 0) {
        size_t limit = (size_t)(read_limit_gb * 1024 * 1024 * 1024);
        if (limit < read_size) read_size = limit;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "  mmap_bench: PhantomRAM synthetic benchmark\n");
    fprintf(stderr, "  ============================================\n");
    fprintf(stderr, "  File:       %s\n", filepath);
    fprintf(stderr, "  File size:  %.2f GB\n", file_size / (1024.0 * 1024 * 1024));
    fprintf(stderr, "  Read size:  %.2f GB\n", read_size / (1024.0 * 1024 * 1024));
    fprintf(stderr, "  Chunk:      %d MB\n", CHUNK_SIZE / (1024 * 1024));
    fprintf(stderr, "  Layer sim:  %d MB\n", layer_mb);
    fprintf(stderr, "  Compute:    %d ms between layers%s\n", compute_ms,
            compute_ms ? " (simulating inference)" : " (pure I/O)");
    fprintf(stderr, "\n");

    /* mmap the file */
    void *mapped = mmap(NULL, read_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    fprintf(stderr, "  mmap'd at %p\n", mapped);

    /* Drop page cache — but only if PhantomRAM isn't active.
     * When PhantomRAM is running, we rely on the system-level
     * drop_caches (done before launch) for a fair cold start.
     * posix_fadvise(DONTNEED) after mmap would nuke PhantomRAM's
     * initial burst, giving an unfairly pessimistic result. */
    if (!getenv("LD_PRELOAD")) {
        fprintf(stderr, "  Dropping page cache (posix_fadvise DONTNEED)...\n");
        posix_fadvise(fd, 0, read_size, POSIX_FADV_DONTNEED);
        usleep(500000);
    } else {
        fprintf(stderr, "  Skipping posix_fadvise (PhantomRAM active — using system drop_caches)\n");
    }

    /* Check initial residency */
    size_t total_pages = read_size / PAGE_SIZE;
    size_t resident_before = count_resident_pages(mapped, read_size);
    fprintf(stderr, "  Resident pages before read: %zu / %zu (%.1f%%)\n",
            resident_before, total_pages,
            total_pages ? 100.0 * resident_before / total_pages : 0);
    fprintf(stderr, "\n");

    /* If PhantomRAM is active, give it a moment to start prefetching */
    if (getenv("LD_PRELOAD")) {
        fprintf(stderr, "  LD_PRELOAD detected — waiting 2s for prefetcher startup...\n");
        sleep(2);
        size_t resident_after_wait = count_resident_pages(mapped, read_size);
        fprintf(stderr, "  Resident pages after prefetcher startup: %zu / %zu (%.1f%%)\n",
                resident_after_wait, total_pages,
                total_pages ? 100.0 * resident_after_wait / total_pages : 0);
        fprintf(stderr, "\n");
    }

    /* ---- Sequential read benchmark ---- */
    fprintf(stderr, "  Starting sequential read...\n");
    fprintf(stderr, "  %-8s  %-10s  %-10s  %-10s\n",
            "Layer", "Size(MB)", "Time(s)", "Speed(GB/s)");
    fprintf(stderr, "  %-8s  %-10s  %-10s  %-10s\n",
            "-----", "--------", "-------", "----------");

    volatile uint64_t sink = 0;  /* Prevent optimization */
    size_t offset = 0;
    int layer_num = 0;
    double total_start = now_sec();

    while (offset < read_size) {
        size_t layer_end = offset + layer_size;
        if (layer_end > read_size) layer_end = read_size;
        size_t layer_bytes = layer_end - offset;

        double layer_start = now_sec();

        /* Touch every page in this "layer" — forces page faults */
        const uint8_t *base = (const uint8_t *)mapped + offset;
        for (size_t pos = 0; pos < layer_bytes; pos += TOUCH_STRIDE) {
            sink += base[pos];  /* Read one byte per page */
        }

        double layer_elapsed = now_sec() - layer_start;
        double layer_gb = layer_bytes / (1024.0 * 1024 * 1024);
        double speed = layer_elapsed > 0 ? layer_gb / layer_elapsed : 0;

        fprintf(stderr, "  %-8d  %-10.0f  %-10.3f  %-10.2f\n",
                layer_num, layer_bytes / (1024.0 * 1024),
                layer_elapsed, speed);

        /* Simulate compute phase — this is where PhantomRAM prefetches
         * the next layer while the "GPU/CPU" is busy with matmuls */
        if (compute_ms > 0) {
            usleep(compute_ms * 1000);
        }

        offset = layer_end;
        layer_num++;
    }

    double total_elapsed = now_sec() - total_start;
    double total_gb = read_size / (1024.0 * 1024 * 1024);

    fprintf(stderr, "\n");
    fprintf(stderr, "  ============================================\n");
    fprintf(stderr, "  Total read:     %.2f GB\n", total_gb);
    fprintf(stderr, "  Total time:     %.3f s\n", total_elapsed);
    fprintf(stderr, "  Avg throughput: %.2f GB/s\n",
            total_elapsed > 0 ? total_gb / total_elapsed : 0);
    fprintf(stderr, "\n");

    /* Final residency */
    size_t resident_after = count_resident_pages(mapped, read_size);
    fprintf(stderr, "  Resident pages after read: %zu / %zu (%.1f%%)\n",
            resident_after, total_pages,
            total_pages ? 100.0 * resident_after / total_pages : 0);

    /* Prevent sink from being optimized away */
    if (sink == 0xDEADBEEF) printf("unlikely\n");

    munmap(mapped, read_size);
    close(fd);

    fprintf(stderr, "\n  Done.\n\n");
    return 0;
}
