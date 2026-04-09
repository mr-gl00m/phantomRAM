/*
 * PhantomRAM — test_pcache.c
 *
 * Tests for the v0.3 transparent page cache prefetcher.
 * Validates: init/destroy, region registration, prefetch, mincore residency.
 */

#define _GNU_SOURCE
#include "phantom_prefetch_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>

/* --------------------------------------------------------------------------
 * Test helpers
 * -------------------------------------------------------------------------- */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                                  \
    do {                                                            \
        tests_run++;                                                \
        printf("  [%d] %-40s ", tests_run, name);                   \
        fflush(stdout);                                             \
    } while (0)

#define PASS()                                                      \
    do { tests_passed++; printf("PASS\n"); } while (0)

#define FAIL(msg)                                                   \
    do { printf("FAIL: %s\n", msg); return; } while (0)

#define ASSERT(cond, msg)                                           \
    do { if (!(cond)) { FAIL(msg); } } while (0)

/*
 * Create a temporary file with predictable content.
 * Each 2MB page starts with its page number (uint32_t).
 */
static int create_test_file(size_t size_mb)
{
    char path[] = "/tmp/phantom_pcache_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    unlink(path);  /* auto-delete on close */

    size_t size = size_mb * 1024 * 1024;
    if (ftruncate(fd, size) < 0) {
        close(fd);
        return -1;
    }

    /* Write page markers at the start of each 2MB chunk */
    size_t num_pages = size / PCACHE_PAGE_SIZE;
    for (size_t i = 0; i < num_pages; i++) {
        uint32_t marker = (uint32_t)i;
        if (pwrite(fd, &marker, sizeof(marker),
                   i * PCACHE_PAGE_SIZE) != sizeof(marker)) {
            close(fd);
            return -1;
        }
    }

    return fd;
}

/* --------------------------------------------------------------------------
 * Tests
 * -------------------------------------------------------------------------- */

static void test_init_destroy(void)
{
    TEST("init_destroy");

    phantom_pcache_t *ctx = NULL;
    phantom_err_t err = phantom_pcache_init(&ctx, 0);
    ASSERT(err == PHANTOM_OK, "init failed");
    ASSERT(ctx != NULL, "ctx is NULL");

    phantom_pcache_destroy(ctx);
    PASS();
}

static void test_register_region(void)
{
    TEST("register_region");

    phantom_pcache_t *ctx = NULL;
    phantom_pcache_init(&ctx, 0);

    /* Create a 16MB test file */
    int fd = create_test_file(16);
    ASSERT(fd >= 0, "create_test_file failed");

    /* mmap it (kernel file-backed mapping) */
    size_t size = 16 * 1024 * 1024;
    void *addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT(addr != MAP_FAILED, "mmap failed");

    /* Register with pcache */
    phantom_err_t err = phantom_pcache_register(ctx, fd, addr, size, 0);
    ASSERT(err == PHANTOM_OK, "register failed");

    munmap(addr, size);
    close(fd);
    phantom_pcache_destroy(ctx);
    PASS();
}

static void test_prefetch_residency(void)
{
    TEST("prefetch_residency");

    phantom_pcache_t *ctx = NULL;
    phantom_pcache_init(&ctx, 0);

    /* Create a 32MB test file */
    int fd = create_test_file(32);
    ASSERT(fd >= 0, "create_test_file failed");

    size_t size = 32 * 1024 * 1024;
    void *addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT(addr != MAP_FAILED, "mmap failed");

    /* Drop caches for this mapping */
    madvise(addr, size, MADV_DONTNEED);

    /* Verify pages are NOT resident */
    unsigned char vec[16] = {0};
    mincore(addr, 16 * sysconf(_SC_PAGESIZE), vec);
    int initially_resident = 0;
    for (int i = 0; i < 16; i++)
        if (vec[i] & 1) initially_resident++;

    /* Register and prefetch */
    phantom_pcache_register(ctx, fd, addr, size, 0);

    /* Explicit prefetch of first 8MB */
    phantom_pcache_prefetch_range(ctx, fd, 0, 8 * 1024 * 1024);

    /* Give readahead time to complete */
    usleep(100000);  /* 100ms */

    /* Check residency — first pages should now be resident */
    memset(vec, 0, sizeof(vec));
    mincore(addr, 16 * sysconf(_SC_PAGESIZE), vec);
    int now_resident = 0;
    for (int i = 0; i < 16; i++)
        if (vec[i] & 1) now_resident++;

    ASSERT(now_resident > initially_resident,
           "prefetch didn't increase page residency");

    munmap(addr, size);
    close(fd);
    phantom_pcache_destroy(ctx);
    PASS();
}

static void test_auto_prefetch(void)
{
    TEST("auto_prefetch");

    phantom_pcache_t *ctx = NULL;
    phantom_pcache_init(&ctx, 0);

    /* Create a 64MB test file */
    int fd = create_test_file(64);
    ASSERT(fd >= 0, "create_test_file failed");

    size_t size = 64 * 1024 * 1024;
    void *addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT(addr != MAP_FAILED, "mmap failed");

    /* Evict from both page tables AND page cache */
    madvise(addr, size, MADV_DONTNEED);
    posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED);

    phantom_pcache_register(ctx, fd, addr, size, 0);
    phantom_pcache_start(ctx);

    /* Simulate sequential access: touch first few pages to create a frontier */
    volatile uint32_t sink = 0;
    for (int i = 0; i < 4; i++) {
        sink += *(volatile uint32_t *)((char *)addr + i * PCACHE_PAGE_SIZE);
    }
    (void)sink;

    /* Give the prefetch thread time to detect frontier and prefetch ahead */
    usleep(200000);  /* 200ms */

    /* Get stats */
    phantom_pcache_stats_t stats;
    phantom_pcache_get_stats(ctx, &stats);

    printf("[prefetched=%lu, resident=%lu, frontier=%lu] ",
           (unsigned long)stats.pages_prefetched,
           (unsigned long)stats.pages_already_resident,
           (unsigned long)stats.frontier_advances);

    /* The prefetcher should have either:
     * 1. Issued prefetches (pages were cold), OR
     * 2. Found pages already resident (kernel cached aggressively)
     * Either way, the auto-prefetch machinery ran. */
    ASSERT(stats.pages_prefetched > 0 || stats.pages_already_resident > 0,
           "auto-prefetch didn't run at all");

    munmap(addr, size);
    close(fd);
    phantom_pcache_destroy(ctx);
    PASS();
}

static void test_stats(void)
{
    TEST("stats");

    phantom_pcache_t *ctx = NULL;
    phantom_pcache_init(&ctx, 0);

    phantom_pcache_stats_t stats;
    phantom_err_t err = phantom_pcache_get_stats(ctx, &stats);
    ASSERT(err == PHANTOM_OK, "get_stats failed");
    ASSERT(stats.pages_prefetched == 0, "initial stats not zero");

    phantom_pcache_destroy(ctx);
    PASS();
}

static void test_data_correctness(void)
{
    TEST("data_correctness");

    /* Verify that prefetched data is correct when read through mmap */
    int fd = create_test_file(16);
    ASSERT(fd >= 0, "create_test_file failed");

    size_t size = 16 * 1024 * 1024;
    void *addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT(addr != MAP_FAILED, "mmap failed");

    phantom_pcache_t *ctx = NULL;
    phantom_pcache_init(&ctx, 0);
    phantom_pcache_register(ctx, fd, addr, size, 0);
    phantom_pcache_prefetch_range(ctx, fd, 0, size);
    usleep(100000);

    /* Read through mmap and verify page markers */
    size_t num_pages = size / PCACHE_PAGE_SIZE;
    for (size_t i = 0; i < num_pages; i++) {
        uint32_t marker = *(uint32_t *)((char *)addr + i * PCACHE_PAGE_SIZE);
        ASSERT(marker == (uint32_t)i, "page marker mismatch");
    }

    munmap(addr, size);
    close(fd);
    phantom_pcache_destroy(ctx);
    PASS();
}

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */

int main(void)
{
    printf("\n=== PhantomRAM v0.3 — Prefetch Cache Tests ===\n\n");

    test_init_destroy();
    test_register_region();
    test_prefetch_residency();
    test_auto_prefetch();
    test_stats();
    test_data_correctness();

    printf("\n  Results: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
