/*
 * PhantomRAM — phantom_prefetch_cache.h
 *
 * Transparent page cache prefetcher (v0.3).
 *
 * Instead of intercepting page faults via userfaultfd (which doubles
 * memory by creating a second mapping), this module lets the kernel
 * handle file-backed mmap naturally and warms the page cache ahead
 * of access using readahead + io_uring.
 *
 * Zero extra memory. Same NVMe acceleration.
 */

#ifndef PHANTOM_PREFETCH_CACHE_H
#define PHANTOM_PREFETCH_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Error codes (shared with phantom_core.h)
 * -------------------------------------------------------------------------- */

#ifndef PHANTOM_ERR_DEFINED
#define PHANTOM_ERR_DEFINED
typedef enum {
    PHANTOM_OK         =  0,
    PHANTOM_ERR_MMAP   = -1,
    PHANTOM_ERR_UFFD   = -2,
    PHANTOM_ERR_URING  = -3,
    PHANTOM_ERR_OOM    = -4,
    PHANTOM_ERR_IO     = -5,
    PHANTOM_ERR_INVALID = -6,
} phantom_err_t;
#endif

/* --------------------------------------------------------------------------
 * Configuration
 * -------------------------------------------------------------------------- */

#define PCACHE_MAX_REGIONS      64
#define PCACHE_PAGE_SIZE        (2 * 1024 * 1024)   /* 2MB prefetch granularity */
#define PCACHE_READAHEAD_PAGES  256                  /* Pages to prefetch ahead (512MB) */
#define PCACHE_SCAN_WINDOW      512                  /* Pages to scan for frontier */
#define PCACHE_INITIAL_BURST    512                  /* Pages to prefetch on registration (1GB) */
#define PCACHE_IO_DEPTH         128                  /* io_uring queue depth */
#define PCACHE_POLL_INTERVAL_US 1000                 /* Residency poll interval (1ms) */

/* --------------------------------------------------------------------------
 * Stats
 * -------------------------------------------------------------------------- */

typedef struct {
    uint64_t pages_prefetched;       /* readahead() calls issued */
    uint64_t pages_already_resident; /* Skipped (already in page cache) */
    uint64_t bytes_prefetched;       /* Total bytes submitted for readahead */
    uint64_t frontier_advances;      /* Times the access frontier moved */
    uint64_t layer_prefetches;       /* Whole-layer prefetches triggered */
    double   avg_lead_pages;         /* Average pages ahead of access */
} phantom_pcache_stats_t;

/* --------------------------------------------------------------------------
 * Registered region
 * -------------------------------------------------------------------------- */

typedef struct {
    int         fd;             /* File descriptor (duped, owned by us) */
    void       *mmap_addr;     /* Where the kernel mmap'd this region */
    size_t      size;           /* Size in bytes */
    uint64_t    file_offset;    /* Offset within the file */
    size_t      num_pages;      /* size / PCACHE_PAGE_SIZE */
    bool        active;
} pcache_region_t;

/* --------------------------------------------------------------------------
 * Layer info (from GGUF parser)
 * -------------------------------------------------------------------------- */

typedef struct {
    uint64_t    file_offset;    /* Absolute byte offset in file */
    size_t      size;           /* Bytes for this layer's weights */
} pcache_layer_t;

typedef struct {
    pcache_layer_t *layers;
    int             num_layers;
    int             lookahead;      /* Layers to prefetch ahead (default 2) */
} pcache_layer_map_t;

/* --------------------------------------------------------------------------
 * Opaque context
 * -------------------------------------------------------------------------- */

typedef struct phantom_pcache phantom_pcache_t;

/*
 * Initialize the page cache prefetcher.
 * Lightweight — no buffer pool, no userfaultfd.
 */
phantom_err_t phantom_pcache_init(phantom_pcache_t **out, int verbose);

/*
 * Register a file-backed mmap region for prefetching.
 * The fd is duped internally (caller keeps ownership of their fd).
 */
phantom_err_t phantom_pcache_register(phantom_pcache_t *ctx,
                                      int fd,
                                      void *mmap_addr,
                                      size_t size,
                                      uint64_t file_offset);

/*
 * Set the GGUF layer map for smart layer-aware prefetching.
 * If not set, falls back to sequential-only prefetch.
 */
phantom_err_t phantom_pcache_set_layer_map(phantom_pcache_t *ctx,
                                           const pcache_layer_map_t *map);

/*
 * Prefetch a specific byte range (async, non-blocking).
 * Uses readahead() to populate the kernel page cache.
 */
phantom_err_t phantom_pcache_prefetch_range(phantom_pcache_t *ctx,
                                            int fd,
                                            uint64_t offset,
                                            size_t length);

/*
 * Start the background auto-prefetch thread.
 * Monitors page residency via mincore() and prefetches ahead of access.
 */
phantom_err_t phantom_pcache_start(phantom_pcache_t *ctx);

/*
 * Get current stats snapshot.
 */
phantom_err_t phantom_pcache_get_stats(phantom_pcache_t *ctx,
                                       phantom_pcache_stats_t *out);

/*
 * Notify that a region was partially unmapped from the start.
 * Adjusts the tracked mmap_addr and size to reflect the trim.
 */
void phantom_pcache_notify_trim(phantom_pcache_t *ctx,
                                void *old_addr, void *new_addr,
                                size_t new_size);

/*
 * Tear down. Stops prefetch thread, closes duped fds.
 */
void phantom_pcache_destroy(phantom_pcache_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PHANTOM_PREFETCH_CACHE_H */
