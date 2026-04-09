/*
 * PhantomRAM — LD_PRELOAD shim for transparent LLM integration (v0.3).
 *
 * v0.3 pivot: Instead of intercepting mmap with userfaultfd (which
 * doubles memory), we let the kernel handle file-backed mmap naturally
 * and run a transparent page cache prefetcher alongside it.
 *
 * The kernel's page cache IS our buffer pool. PhantomRAM just warms
 * it ahead of access using readahead() with GGUF layer awareness.
 *
 * Usage:
 *   LD_PRELOAD=./libphantom_preload.so ./llama-cli -m model.gguf ...
 *
 * Environment variables:
 *   PHANTOM_VERBOSE          Set to 1 for debug output (default: 0)
 *
 * Build:
 *   gcc -shared -fPIC -o libphantom_preload.so phantom_preload.c \
 *       phantom_prefetch_cache.c gguf_loader.c -lpthread -ldl
 */

#define _GNU_SOURCE
#include "phantom_prefetch_cache.h"
#include "gguf_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>

#include <sys/mman.h>
#include <sys/syscall.h>

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

#define MIN_INTERCEPT_SIZE  (64ULL * 1024 * 1024)  /* 64 MB */
#define MAX_PHANTOM_MAPS    64

/* ------------------------------------------------------------------ */
/* Tracked phantom regions                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    void       *addr;
    size_t      length;
    int         fd;         /* Duped fd for this region */
} phantom_map_entry_t;

static phantom_pcache_t    *g_pcache       = NULL;
static phantom_map_entry_t  g_maps[MAX_PHANTOM_MAPS];
static int                  g_num_maps     = 0;
static pthread_mutex_t      g_lock         = PTHREAD_MUTEX_INITIALIZER;
static bool                 g_initialized  = false;
static bool                 g_init_failed  = false;
static int                  g_verbose      = 0;

/* ------------------------------------------------------------------ */
/* Real libc function pointers                                         */
/* ------------------------------------------------------------------ */

static void *(*real_mmap)(void *, size_t, int, int, int, off_t)   = NULL;
static int   (*real_munmap)(void *, size_t)                        = NULL;
static int   (*real_mlock)(const void *, size_t)                   = NULL;

/*
 * Resolve real libc functions via dlsym(RTLD_NEXT, ...).
 * Guard against recursion: dlsym may itself call mmap internally,
 * so we use a flag and fall back to raw syscall if re-entered.
 */
static volatile int g_in_dlsym = 0;

static void resolve_real_functions(void)
{
    if (real_mmap) return;

    g_in_dlsym = 1;
    real_mmap   = dlsym(RTLD_NEXT, "mmap");
    real_munmap = dlsym(RTLD_NEXT, "munmap");
    real_mlock  = dlsym(RTLD_NEXT, "mlock");
    g_in_dlsym = 0;
}

/* Raw syscall fallback for mmap during early init / dlsym recursion */
static void *raw_mmap(void *addr, size_t length, int prot,
                       int flags, int fd, off_t offset)
{
    return (void *)syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
}

/* ------------------------------------------------------------------ */
/* GGUF detection                                                      */
/* ------------------------------------------------------------------ */

static bool is_gguf_fd(int fd)
{
    if (fd < 0) return false;

    uint32_t magic = 0;
    ssize_t n = pread(fd, &magic, sizeof(magic), 0);
    return (n == sizeof(magic) && magic == GGUF_MAGIC);
}

/* ------------------------------------------------------------------ */
/* Region tracking                                                     */
/* ------------------------------------------------------------------ */

static bool is_phantom_addr(const void *addr)
{
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_num_maps; i++) {
        const char *base = (const char *)g_maps[i].addr;
        if ((const char *)addr >= base &&
            (const char *)addr <  base + g_maps[i].length) {
            pthread_mutex_unlock(&g_lock);
            return true;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return false;
}

static void track_region(void *addr, size_t length, int fd)
{
    pthread_mutex_lock(&g_lock);
    if (g_num_maps < MAX_PHANTOM_MAPS) {
        g_maps[g_num_maps++] = (phantom_map_entry_t){
            .addr   = addr,
            .length = length,
            .fd     = fd,
        };
    }
    pthread_mutex_unlock(&g_lock);
}

static bool __attribute__((unused)) untrack_region(void *addr)
{
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_num_maps; i++) {
        if (g_maps[i].addr == addr) {
            g_maps[i] = g_maps[--g_num_maps];
            pthread_mutex_unlock(&g_lock);
            return true;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return false;
}

/* ------------------------------------------------------------------ */
/* Lazy PhantomRAM initialization                                      */
/* ------------------------------------------------------------------ */

static void lazy_init(void)
{
    if (g_initialized || g_init_failed) return;

    const char *verbose_str = getenv("PHANTOM_VERBOSE");
    g_verbose = verbose_str ? atoi(verbose_str) : 0;

    fprintf(stderr,
        "\n"
        "  phantom-preload: initializing (v0.3 prefetch mode)\n"
        "    Mode:           transparent page cache prefetcher\n"
        "    Extra memory:   0 (kernel page cache only)\n"
        "\n");

    phantom_err_t err = phantom_pcache_init(&g_pcache, g_verbose);
    if (err != PHANTOM_OK) {
        fprintf(stderr, "  phantom-preload: pcache init FAILED (err=%d) "
                "- falling through to standard mmap\n", err);
        g_pcache = NULL;
        g_init_failed = true;
    }

    g_initialized = true;
}

/* ------------------------------------------------------------------ */
/* GGUF layer map setup                                                */
/* ------------------------------------------------------------------ */

static void setup_layer_map(int fd)
{
    if (!g_pcache) return;

    phantom_gguf_model_t model = {0};
    if (phantom_gguf_parse(fd, &model) != PHANTOM_OK) {
        if (g_verbose)
            fprintf(stderr, "  phantom-preload: GGUF parse failed, "
                    "using sequential-only prefetch\n");
        return;
    }

    if (g_verbose)
        phantom_gguf_print_summary(&model);

    /* Build pcache layer map from GGUF tensor metadata */
    pcache_layer_map_t lmap = {0};
    lmap.num_layers = model.num_layers;
    lmap.lookahead = 2;
    lmap.layers = calloc(model.num_layers, sizeof(pcache_layer_t));
    if (!lmap.layers) {
        phantom_gguf_free(&model);
        return;
    }

    /* For each layer, find the offset range of its tensors */
    for (int layer = 0; layer < model.num_layers; layer++) {
        uint64_t min_offset = UINT64_MAX;
        uint64_t max_end = 0;

        for (uint64_t t = 0; t < model.n_tensors; t++) {
            if (model.tensors[t].layer_id == layer) {
                uint64_t abs_offset = model.data_offset + model.tensors[t].offset;
                uint64_t end = abs_offset + model.tensors[t].size;
                if (abs_offset < min_offset) min_offset = abs_offset;
                if (end > max_end) max_end = end;
            }
        }

        if (min_offset < UINT64_MAX) {
            lmap.layers[layer].file_offset = min_offset;
            lmap.layers[layer].size = max_end - min_offset;
        }
    }

    phantom_pcache_set_layer_map(g_pcache, &lmap);
    free(lmap.layers);
    phantom_gguf_free(&model);
}

/* ------------------------------------------------------------------ */
/* Intercepted: mmap                                                   */
/* ------------------------------------------------------------------ */

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
    /* During dlsym resolution or before real_mmap is set, use raw syscall */
    if (g_in_dlsym || !real_mmap) {
        if (!real_mmap && !g_in_dlsym)
            resolve_real_functions();
        if (!real_mmap)
            return raw_mmap(addr, length, prot, flags, fd, offset);
    }

    /* Only intercept file-backed, large, GGUF mappings */
    if (fd >= 0 && length >= MIN_INTERCEPT_SIZE && is_gguf_fd(fd)) {
        lazy_init();

        /* Let the REAL mmap proceed — kernel handles the file-backed mapping */
        void *result = real_mmap(addr, length, prot, flags, fd, offset);
        if (result == MAP_FAILED)
            return result;

        if (g_pcache) {
            fprintf(stderr,
                "  phantom-preload: observed GGUF mmap — "
                "%zu MB at %p (fd=%d, offset=%ld)\n",
                length / (1024 * 1024), result, fd, (long)offset);

            /* Register this region for prefetching */
            phantom_pcache_register(g_pcache, fd, result, length,
                                    (uint64_t)offset);
            track_region(result, length, fd);

            /* Parse GGUF and set up layer-aware prefetch */
            setup_layer_map(fd);

            /* Start auto-prefetch if not already running */
            phantom_pcache_start(g_pcache);
        }

        return result;
    }

    return real_mmap(addr, length, prot, flags, fd, offset);
}

/* Also intercept mmap64 (same on 64-bit, but some libc versions export both) */
void *mmap64(void *addr, size_t length, int prot, int flags,
             int fd, off_t offset)
    __attribute__((alias("mmap")));

/* ------------------------------------------------------------------ */
/* Intercepted: mlock / mlock2                                         */
/* ------------------------------------------------------------------ */

int mlock(const void *addr, size_t len)
{
    if (!real_mlock) resolve_real_functions();

    /* Suppress mlock on phantom regions — let the kernel manage
     * page eviction freely. mlock would pin pages and prevent
     * the kernel from evicting them under memory pressure. */
    if (is_phantom_addr(addr)) {
        if (g_verbose)
            fprintf(stderr,
                "  phantom-preload: suppressed mlock at %p (%zu MB) "
                "- kernel manages eviction\n",
                addr, len / (1024 * 1024));
        return 0;  /* Pretend success */
    }

    return real_mlock(addr, len);
}

/* ------------------------------------------------------------------ */
/* Intercepted: munmap                                                 */
/* ------------------------------------------------------------------ */

int munmap(void *addr, size_t length)
{
    if (!real_munmap) resolve_real_functions();

    /* Check if this overlaps a phantom region */
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_num_maps; i++) {
        if (g_maps[i].addr == addr && length == g_maps[i].length) {
            /* Full unmap of phantom region — remove tracking */
            g_maps[i] = g_maps[--g_num_maps];
            pthread_mutex_unlock(&g_lock);
            if (g_verbose)
                fprintf(stderr,
                    "  phantom-preload: full munmap at %p (%zu MB)\n",
                    addr, length / (1024 * 1024));
            return real_munmap(addr, length);
        }
        if (g_maps[i].addr == addr && length < g_maps[i].length) {
            /* Partial unmap from the start (metadata trim).
             * Adjust tracked region to reflect the remaining portion. */
            void *old_addr = g_maps[i].addr;
            g_maps[i].addr = (char *)addr + length;
            g_maps[i].length -= length;
            void *new_addr = g_maps[i].addr;
            size_t new_length = g_maps[i].length;
            pthread_mutex_unlock(&g_lock);

            /* Notify pcache so it doesn't mincore() unmapped pages */
            if (g_pcache)
                phantom_pcache_notify_trim(g_pcache, old_addr,
                                           new_addr, new_length);

            if (g_verbose)
                fprintf(stderr,
                    "  phantom-preload: partial munmap at %p (%zu MB) "
                    "- region trimmed to %p (%zu MB)\n",
                    addr, length / (1024 * 1024),
                    new_addr, new_length / (1024 * 1024));
            return real_munmap(addr, length);
        }
    }
    pthread_mutex_unlock(&g_lock);

    return real_munmap(addr, length);
}

/* ------------------------------------------------------------------ */
/* Cleanup on process exit                                             */
/* ------------------------------------------------------------------ */

__attribute__((destructor))
static void phantom_preload_cleanup(void)
{
    if (g_pcache) {
        phantom_pcache_destroy(g_pcache);
        g_pcache = NULL;
    }
}
