/*
 * PhantomRAM — gguf_loader.h
 *
 * Minimal GGUF model file parser. Extracts tensor metadata (name, offset,
 * size, layer_id) so the prefetcher can build a layer-to-byte-range map.
 *
 * GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

#ifndef PHANTOM_GGUF_LOADER_H
#define PHANTOM_GGUF_LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes — shared with phantom_prefetch_cache.h via guard. */
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

/* GGUF magic: "GGUF" in little-endian = 0x46554747 */
#define GGUF_MAGIC 0x46554747

/* Tensor descriptor extracted from GGUF metadata */
typedef struct {
    char        name[256];
    uint64_t    offset;         /* Byte offset in file (from data section start) */
    size_t      size;           /* Total size in bytes */
    int         layer_id;       /* Parsed from name (e.g., "blk.42.attn_q" -> 42) */
    int         n_dims;
    uint64_t    dims[4];
    uint32_t    type;           /* GGML type enum */
} phantom_tensor_info_t;

/* Parsed model metadata */
typedef struct {
    uint32_t                version;
    uint64_t                n_tensors;
    uint64_t                n_kv;           /* Number of key-value metadata entries */
    uint64_t                data_offset;    /* Byte offset where tensor data begins */
    int                     num_layers;     /* Inferred from tensor names */
    phantom_tensor_info_t  *tensors;
} phantom_gguf_model_t;

/*
 * Parse a GGUF file and extract tensor metadata.
 * Does NOT load tensor data — only reads headers.
 */
phantom_err_t phantom_gguf_parse(int fd, phantom_gguf_model_t *out);

/* Free parsed model metadata. */
void phantom_gguf_free(phantom_gguf_model_t *model);

/* Print model summary to stdout. */
void phantom_gguf_print_summary(const phantom_gguf_model_t *model);

#ifdef __cplusplus
}
#endif

#endif /* PHANTOM_GGUF_LOADER_H */
