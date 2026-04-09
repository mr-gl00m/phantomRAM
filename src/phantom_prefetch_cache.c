/*
 * PhantomRAM — phantom_prefetch_cache.c
 *
 * Transparent page cache prefetcher (v0.3).
 *
 * Monitors file-backed mmap residency via mincore() and prefetches
 * ahead of the access frontier using readahead(). For GGUF models,
 * uses layer-aware prefetching to warm entire layers before they're
 * needed by the inference engine.
 *
 * Zero extra memory — the kernel page cache IS the buffer pool.
 *
 * Build: Linux 5.7+. Link with -lpthread.
 */

#define _GNU_SOURCE
#include "phantom_prefetch_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>

#include <sys/mman.h>

/* --------------------------------------------------------------------------
 * Internal context
 * -------------------------------------------------------------------------- */

struct phantom_pcache {
    /* Registered file-backed mmap regions */
    pcache_region_t     regions[PCACHE_MAX_REGIONS];
    int                 num_regions;
    pthread_mutex_t     regions_lock;

    /* Layer map (optional, from GGUF parser) */
    pcache_layer_map_t  layer_map;
    bool                has_layer_map;

    /* Auto-prefetch thread */
    pthread_t           prefetch_thread;
    volatile bool       running;

    /* Per-region frontier tracking: index of highest page we've observed
     * as resident. Prefetch issues readahead starting from here. */
    size_t             *frontier;    /* Array [PCACHE_MAX_REGIONS] */

    /* Stats */
    phantom_pcache_stats_t stats;
    pthread_mutex_t        stats_lock;

    int                 verbose;
};

/* --------------------------------------------------------------------------
 * Time utility
 * -------------------------------------------------------------------------- */

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* --------------------------------------------------------------------------
 * Residency check via mincore()
 * --------------------------------------------------------------------------
 * mincore() returns a byte per page indicating whether the page is
 * currently in the page cache. We use this to detect where the
 * application is currently reading (the "frontier") and prefetch
 * ahead of it.
 */

/*
 * Find the highest resident page index in a region.
 * Uses batched mincore() — checks multiple 2MB chunks per syscall
 * by querying a large range and sampling every PCACHE_PAGE_SIZE bytes.
 */
static size_t find_frontier(pcache_region_t *region, size_t start_page)
{
    if (!region->mmap_addr || !region->active)
        return start_page;

    long sys_page = sysconf(_SC_PAGESIZE);
    size_t pages_per_chunk = PCACHE_PAGE_SIZE / sys_page;
    size_t frontier = start_page;

    /* Scan a wide window to find how far the application has read */
    size_t scan_end = start_page + PCACHE_SCAN_WINDOW;
    if (scan_end > region->num_pages) scan_end = region->num_pages;
    size_t scan_count = scan_end - start_page;
    if (scan_count == 0) return start_page;

    /* Batch mincore: query all system pages in one call, then sample
     * every pages_per_chunk'th byte to check each 2MB chunk. */
    size_t total_sys_pages = scan_count * pages_per_chunk;
    unsigned char *vec = malloc(total_sys_pages);
    if (!vec) {
        /* Fallback: per-chunk mincore */
        for (size_t i = start_page; i < scan_end; i++) {
            unsigned char v = 0;
            void *addr = (char *)region->mmap_addr + i * PCACHE_PAGE_SIZE;
            if (mincore(addr, sys_page, &v) == 0 && (v & 1))
                frontier = i;
        }
        return frontier;
    }

    void *scan_addr = (char *)region->mmap_addr + start_page * PCACHE_PAGE_SIZE;
    size_t scan_bytes = scan_count * PCACHE_PAGE_SIZE;

    /* Clamp to region boundary */
    size_t max_bytes = region->size - start_page * PCACHE_PAGE_SIZE;
    if (scan_bytes > max_bytes) scan_bytes = max_bytes;
    size_t actual_sys_pages = scan_bytes / sys_page;

    if (mincore(scan_addr, scan_bytes, vec) == 0) {
        /* Sample every pages_per_chunk'th entry */
        for (size_t i = 0; i < scan_count && i * pages_per_chunk < actual_sys_pages; i++) {
            if (vec[i * pages_per_chunk] & 1)
                frontier = start_page + i;
        }
    }

    free(vec);
    return frontier;
}

/* --------------------------------------------------------------------------
 * Prefetch via readahead()
 * --------------------------------------------------------------------------
 * readahead() is a Linux syscall that populates the page cache
 * asynchronously. It returns immediately and the kernel schedules
 * the I/O in the background. Perfect for prefetching.
 */

static void prefetch_pages(phantom_pcache_t *ctx, pcache_region_t *region,
                           size_t start_page, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        size_t page_idx = start_page + i;
        if (page_idx >= region->num_pages) break;

        uint64_t offset = region->file_offset + page_idx * PCACHE_PAGE_SIZE;
        size_t len = PCACHE_PAGE_SIZE;

        /* Clamp last page to region boundary */
        if (page_idx == region->num_pages - 1) {
            size_t remaining = region->size - page_idx * PCACHE_PAGE_SIZE;
            if (remaining < len) len = remaining;
        }

        /* Check if already resident (avoid redundant I/O).
         * mincore may fail with ENOMEM if page was recently unmapped. */
        unsigned char vec = 0;
        void *chunk_addr = (char *)region->mmap_addr + page_idx * PCACHE_PAGE_SIZE;
        if (mincore(chunk_addr, sysconf(_SC_PAGESIZE), &vec) != 0)
            continue;  /* Unmapped region — skip */
        if (vec & 1) {
            pthread_mutex_lock(&ctx->stats_lock);
            ctx->stats.pages_already_resident++;
            pthread_mutex_unlock(&ctx->stats_lock);
            continue;
        }

        /* Fire async readahead */
        int ret = readahead(region->fd, offset, len);
        if (ret == 0) {
            pthread_mutex_lock(&ctx->stats_lock);
            ctx->stats.pages_prefetched++;
            ctx->stats.bytes_prefetched += len;
            pthread_mutex_unlock(&ctx->stats_lock);
        }
    }
}

/* --------------------------------------------------------------------------
 * Layer-aware prefetch
 * --------------------------------------------------------------------------
 * When we have a GGUF layer map, we can prefetch entire layers at once.
 * Given the current access frontier, figure out which layer is being
 * read, and prefetch the next `lookahead` layers.
 */

static void prefetch_layers_ahead(phantom_pcache_t *ctx,
                                  pcache_region_t *region,
                                  size_t frontier_page)
{
    if (!ctx->has_layer_map) return;

    pcache_layer_map_t *lm = &ctx->layer_map;
    uint64_t frontier_offset = region->file_offset +
                               frontier_page * PCACHE_PAGE_SIZE;

    /* Find which layer the frontier is in */
    int current_layer = -1;
    for (int i = 0; i < lm->num_layers; i++) {
        uint64_t layer_start = lm->layers[i].file_offset;
        uint64_t layer_end = layer_start + lm->layers[i].size;
        if (frontier_offset >= layer_start && frontier_offset < layer_end) {
            current_layer = i;
            break;
        }
    }
    if (current_layer < 0) return;

    /* Prefetch next `lookahead` layers */
    int lookahead = lm->lookahead > 0 ? lm->lookahead : 2;
    for (int i = 1; i <= lookahead; i++) {
        int target = current_layer + i;
        if (target >= lm->num_layers) break;

        uint64_t layer_offset = lm->layers[target].file_offset;
        size_t layer_size = lm->layers[target].size;

        /* Convert to region-relative page indices */
        if (layer_offset < region->file_offset) continue;
        uint64_t rel_offset = layer_offset - region->file_offset;
        size_t start_page = rel_offset / PCACHE_PAGE_SIZE;
        size_t num_pages = (layer_size + PCACHE_PAGE_SIZE - 1) / PCACHE_PAGE_SIZE;

        prefetch_pages(ctx, region, start_page, num_pages);

        pthread_mutex_lock(&ctx->stats_lock);
        ctx->stats.layer_prefetches++;
        pthread_mutex_unlock(&ctx->stats_lock);
    }
}

/* --------------------------------------------------------------------------
 * Auto-prefetch thread
 * --------------------------------------------------------------------------
 * Polls residency every PCACHE_POLL_INTERVAL_US microseconds.
 * When the frontier advances, prefetches PCACHE_READAHEAD_PAGES ahead.
 * Also triggers layer-aware prefetch when a GGUF layer map is set.
 */

static void *prefetch_thread_fn(void *arg)
{
    phantom_pcache_t *ctx = (phantom_pcache_t *)arg;

    struct timespec sleep_ts = {
        .tv_sec = 0,
        .tv_nsec = PCACHE_POLL_INTERVAL_US * 1000,
    };

    /* Track how far ahead we've already prefetched per region */
    size_t prefetch_cursor[PCACHE_MAX_REGIONS];
    memset(prefetch_cursor, 0, sizeof(prefetch_cursor));

    while (ctx->running) {
        nanosleep(&sleep_ts, NULL);

        pthread_mutex_lock(&ctx->regions_lock);
        for (int r = 0; r < ctx->num_regions; r++) {
            pcache_region_t *region = &ctx->regions[r];
            if (!region->active) continue;

            size_t old_frontier = ctx->frontier[r];
            size_t new_frontier = find_frontier(region, old_frontier);

            if (new_frontier > old_frontier) {
                /* Frontier advanced — the app is actively reading */
                ctx->frontier[r] = new_frontier;

                pthread_mutex_lock(&ctx->stats_lock);
                ctx->stats.frontier_advances++;
                pthread_mutex_unlock(&ctx->stats_lock);

                /* Layer-aware prefetch if available */
                prefetch_layers_ahead(ctx, region, new_frontier);
            }

            /* Continuously push the prefetch cursor ahead of the frontier.
             * Don't wait for frontier to advance — keep feeding readahead
             * so the kernel always has I/O in flight. */
            size_t target = ctx->frontier[r] + PCACHE_READAHEAD_PAGES;
            if (target > region->num_pages) target = region->num_pages;

            if (prefetch_cursor[r] < target) {
                size_t start = prefetch_cursor[r];
                size_t count = target - start;
                /* Cap per-iteration to avoid blocking too long */
                if (count > 128) count = 128;
                prefetch_pages(ctx, region, start, count);
                prefetch_cursor[r] = start + count;
            }
        }
        pthread_mutex_unlock(&ctx->regions_lock);
    }

    return NULL;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

phantom_err_t phantom_pcache_init(phantom_pcache_t **out, int verbose)
{
    phantom_pcache_t *ctx = calloc(1, sizeof(phantom_pcache_t));
    if (!ctx) return PHANTOM_ERR_OOM;

    ctx->verbose = verbose;
    ctx->running = false;
    ctx->has_layer_map = false;

    ctx->frontier = calloc(PCACHE_MAX_REGIONS, sizeof(size_t));
    if (!ctx->frontier) {
        free(ctx);
        return PHANTOM_ERR_OOM;
    }

    pthread_mutex_init(&ctx->regions_lock, NULL);
    pthread_mutex_init(&ctx->stats_lock, NULL);

    if (verbose) {
        fprintf(stderr,
            "\n"
            "  phantom-pcache: initialized (transparent prefetcher)\n"
            "    Mode:         page cache warming (zero extra memory)\n"
            "    Readahead:    %d pages (%d MB) ahead\n"
            "    Poll rate:    %d us\n"
            "\n",
            PCACHE_READAHEAD_PAGES,
            PCACHE_READAHEAD_PAGES * (PCACHE_PAGE_SIZE / (1024 * 1024)),
            PCACHE_POLL_INTERVAL_US);
    }

    *out = ctx;
    return PHANTOM_OK;
}

phantom_err_t phantom_pcache_register(phantom_pcache_t *ctx,
                                      int fd,
                                      void *mmap_addr,
                                      size_t size,
                                      uint64_t file_offset)
{
    pthread_mutex_lock(&ctx->regions_lock);

    if (ctx->num_regions >= PCACHE_MAX_REGIONS) {
        pthread_mutex_unlock(&ctx->regions_lock);
        return PHANTOM_ERR_INVALID;
    }

    int idx = ctx->num_regions;
    pcache_region_t *region = &ctx->regions[idx];

    /* Dup the fd so we own our own copy */
    region->fd = dup(fd);
    if (region->fd < 0) {
        pthread_mutex_unlock(&ctx->regions_lock);
        return PHANTOM_ERR_IO;
    }

    /* Clear O_NONBLOCK if set */
    int fl = fcntl(region->fd, F_GETFL);
    if (fl >= 0 && (fl & O_NONBLOCK))
        fcntl(region->fd, F_SETFL, fl & ~O_NONBLOCK);

    region->mmap_addr = mmap_addr;
    region->size = size;
    region->file_offset = file_offset;
    region->num_pages = (size + PCACHE_PAGE_SIZE - 1) / PCACHE_PAGE_SIZE;
    region->active = true;

    ctx->frontier[idx] = 0;
    ctx->num_regions++;

    pthread_mutex_unlock(&ctx->regions_lock);

    if (ctx->verbose) {
        fprintf(stderr,
            "  phantom-pcache: registered region %d — %zu MB at %p "
            "(file offset %lu, %zu pages)\n",
            idx, size / (1024 * 1024), mmap_addr,
            (unsigned long)file_offset, region->num_pages);
    }

    /* Aggressive initial burst: prefetch the first PCACHE_INITIAL_BURST
     * pages (1GB) immediately. Model loading is sequential from the start,
     * so warming this much of the page cache gives us a head start over
     * the kernel's default readahead (~256KB). */
    size_t burst = PCACHE_INITIAL_BURST < region->num_pages
                       ? PCACHE_INITIAL_BURST : region->num_pages;
    prefetch_pages(ctx, region, 0, burst);

    if (ctx->verbose) {
        fprintf(stderr,
            "  phantom-pcache: initial burst — %zu pages (%zu MB) prefetched\n",
            burst, burst * PCACHE_PAGE_SIZE / (1024 * 1024));
    }

    return PHANTOM_OK;
}

phantom_err_t phantom_pcache_set_layer_map(phantom_pcache_t *ctx,
                                           const pcache_layer_map_t *map)
{
    if (!map || map->num_layers <= 0) return PHANTOM_ERR_INVALID;

    ctx->layer_map.num_layers = map->num_layers;
    ctx->layer_map.lookahead = map->lookahead > 0 ? map->lookahead : 2;
    ctx->layer_map.layers = malloc(map->num_layers * sizeof(pcache_layer_t));
    if (!ctx->layer_map.layers) return PHANTOM_ERR_OOM;

    memcpy(ctx->layer_map.layers, map->layers,
           map->num_layers * sizeof(pcache_layer_t));
    ctx->has_layer_map = true;

    if (ctx->verbose) {
        fprintf(stderr,
            "  phantom-pcache: layer map set — %d layers, lookahead=%d\n",
            map->num_layers, ctx->layer_map.lookahead);
    }

    return PHANTOM_OK;
}

phantom_err_t phantom_pcache_prefetch_range(phantom_pcache_t *ctx,
                                            int fd,
                                            uint64_t offset,
                                            size_t length)
{
    /* Find region by fd */
    pthread_mutex_lock(&ctx->regions_lock);
    pcache_region_t *region = NULL;
    for (int i = 0; i < ctx->num_regions; i++) {
        if (ctx->regions[i].active && ctx->regions[i].fd == fd) {
            region = &ctx->regions[i];
            break;
        }
    }
    pthread_mutex_unlock(&ctx->regions_lock);

    if (!region) {
        /* No matching region — use readahead directly */
        readahead(fd, offset, length);
        return PHANTOM_OK;
    }

    /* Convert to page range */
    uint64_t rel_offset = (offset >= region->file_offset)
        ? offset - region->file_offset : 0;
    size_t start_page = rel_offset / PCACHE_PAGE_SIZE;
    size_t num_pages = (length + PCACHE_PAGE_SIZE - 1) / PCACHE_PAGE_SIZE;

    prefetch_pages(ctx, region, start_page, num_pages);
    return PHANTOM_OK;
}

phantom_err_t phantom_pcache_start(phantom_pcache_t *ctx)
{
    if (ctx->running) return PHANTOM_OK;
    ctx->running = true;

    int ret = pthread_create(&ctx->prefetch_thread, NULL,
                             prefetch_thread_fn, ctx);
    if (ret != 0) {
        ctx->running = false;
        return PHANTOM_ERR_IO;
    }

    if (ctx->verbose) {
        fprintf(stderr, "  phantom-pcache: auto-prefetch thread started\n");
    }

    return PHANTOM_OK;
}

phantom_err_t phantom_pcache_get_stats(phantom_pcache_t *ctx,
                                       phantom_pcache_stats_t *out)
{
    pthread_mutex_lock(&ctx->stats_lock);
    *out = ctx->stats;
    pthread_mutex_unlock(&ctx->stats_lock);
    return PHANTOM_OK;
}

void phantom_pcache_notify_trim(phantom_pcache_t *ctx,
                                void *old_addr, void *new_addr,
                                size_t new_size)
{
    pthread_mutex_lock(&ctx->regions_lock);
    for (int i = 0; i < ctx->num_regions; i++) {
        if (ctx->regions[i].mmap_addr == old_addr) {
            size_t trim_bytes = (size_t)((char *)new_addr - (char *)old_addr);
            ctx->regions[i].mmap_addr = new_addr;
            ctx->regions[i].size = new_size;
            ctx->regions[i].file_offset += trim_bytes;
            ctx->regions[i].num_pages = (new_size + PCACHE_PAGE_SIZE - 1)
                                        / PCACHE_PAGE_SIZE;
            /* Reset frontier since addresses changed */
            ctx->frontier[i] = 0;

            if (ctx->verbose) {
                fprintf(stderr,
                    "  phantom-pcache: region %d trimmed — "
                    "now %zu MB at %p (offset +%zu)\n",
                    i, new_size / (1024 * 1024), new_addr, trim_bytes);
            }
            break;
        }
    }
    pthread_mutex_unlock(&ctx->regions_lock);
}

void phantom_pcache_destroy(phantom_pcache_t *ctx)
{
    if (!ctx) return;

    /* Stop prefetch thread */
    ctx->running = false;
    pthread_join(ctx->prefetch_thread, NULL);

    /* Print final stats */
    fprintf(stderr,
        "\n"
        "  phantom-pcache: shutting down\n"
        "    Pages prefetched:       %lu\n"
        "    Pages already resident: %lu\n"
        "    Bytes prefetched:       %.1f GB\n"
        "    Frontier advances:      %lu\n"
        "    Layer prefetches:       %lu\n"
        "\n",
        (unsigned long)ctx->stats.pages_prefetched,
        (unsigned long)ctx->stats.pages_already_resident,
        (double)ctx->stats.bytes_prefetched / (1024.0 * 1024.0 * 1024.0),
        (unsigned long)ctx->stats.frontier_advances,
        (unsigned long)ctx->stats.layer_prefetches);

    /* Close duped fds */
    for (int i = 0; i < ctx->num_regions; i++) {
        if (ctx->regions[i].active && ctx->regions[i].fd >= 0)
            close(ctx->regions[i].fd);
    }

    /* Free layer map */
    if (ctx->has_layer_map)
        free(ctx->layer_map.layers);

    free(ctx->frontier);
    pthread_mutex_destroy(&ctx->regions_lock);
    pthread_mutex_destroy(&ctx->stats_lock);
    free(ctx);
}
