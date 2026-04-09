/*
 * PhantomRAM — gguf_loader.c
 *
 * Minimal GGUF parser. Reads header + tensor info to build a layer map.
 * We only parse enough to know where each tensor lives on disk.
 */

#define _GNU_SOURCE
#include "gguf_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>

/* GGUF value types */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* GGML type sizes (bytes per element, approximate for quantized) */
static const float ggml_type_size[] = {
    [0]  = 4.0,    /* F32 */
    [1]  = 2.0,    /* F16 */
    [2]  = 0.625,  /* Q4_0: 4.5 bits avg -> 18 bytes per 32 elements */
    [3]  = 0.6875, /* Q4_1 */
    [6]  = 0.5625, /* Q5_0 */
    [7]  = 0.625,  /* Q5_1 */
    [8]  = 1.0625, /* Q8_0 */
    [9]  = 1.0625, /* Q8_1 */
    [10] = 0.5625, /* Q2_K */
    [11] = 0.6875, /* Q3_K */
    [12] = 0.625,  /* Q4_K */
    [13] = 0.6875, /* Q5_K */
    [14] = 1.0625, /* Q6_K */
};
#define GGML_TYPE_COUNT 15

/* --------------------------------------------------------------------------
 * Reading helpers
 * -------------------------------------------------------------------------- */

static int read_exact(int fd, void *buf, size_t len)
{
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, (char *)buf + total, len - total);
        if (n <= 0) return -1;
        total += n;
    }
    return 0;
}

static int read_u32(int fd, uint32_t *out)
{
    return read_exact(fd, out, 4);
}

static int read_u64(int fd, uint64_t *out)
{
    return read_exact(fd, out, 8);
}

/* Read a GGUF string: uint64 length + bytes (NOT null-terminated on disk) */
static int read_gguf_string(int fd, char *buf, size_t buf_size)
{
    uint64_t len;
    if (read_u64(fd, &len) != 0) return -1;
    if (len >= buf_size) {
        /* Too long — skip it */
        lseek(fd, len, SEEK_CUR);
        buf[0] = '\0';
        return 0;
    }
    if (read_exact(fd, buf, len) != 0) return -1;
    buf[len] = '\0';
    return 0;
}

/* Skip a GGUF value based on its type */
static int skip_gguf_value(int fd, uint32_t type)
{
    switch (type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:
        return lseek(fd, 1, SEEK_CUR) < 0 ? -1 : 0;
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
        return lseek(fd, 2, SEEK_CUR) < 0 ? -1 : 0;
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
        return lseek(fd, 4, SEEK_CUR) < 0 ? -1 : 0;
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
        return lseek(fd, 8, SEEK_CUR) < 0 ? -1 : 0;
    case GGUF_TYPE_STRING: {
        uint64_t len;
        if (read_u64(fd, &len) != 0) return -1;
        return lseek(fd, len, SEEK_CUR) < 0 ? -1 : 0;
    }
    case GGUF_TYPE_ARRAY: {
        uint32_t elem_type;
        uint64_t count;
        if (read_u32(fd, &elem_type) != 0) return -1;
        if (read_u64(fd, &count) != 0) return -1;
        for (uint64_t i = 0; i < count; i++) {
            if (skip_gguf_value(fd, elem_type) != 0) return -1;
        }
        return 0;
    }
    default:
        return -1;
    }
}

/* --------------------------------------------------------------------------
 * Layer ID extraction from tensor name
 * --------------------------------------------------------------------------
 * Names look like: "blk.42.attn_q.weight" or "layers.17.feed_forward.w1"
 * We extract the first number after "blk." or "layers."
 */

static int extract_layer_id(const char *name)
{
    /* Look for "blk." or "layers." pattern */
    const char *p = strstr(name, "blk.");
    if (p) p += 4;
    else {
        p = strstr(name, "layers.");
        if (p) p += 7;
    }

    if (!p) return -1; /* Not a layer tensor (e.g., "token_embd", "output_norm") */

    int id = 0;
    while (*p && isdigit((unsigned char)*p)) {
        id = id * 10 + (*p - '0');
        p++;
    }
    return id;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

phantom_err_t phantom_gguf_parse(int fd, phantom_gguf_model_t *out)
{
    memset(out, 0, sizeof(*out));
    lseek(fd, 0, SEEK_SET);

    /* Read magic */
    uint32_t magic;
    if (read_u32(fd, &magic) != 0 || magic != GGUF_MAGIC) {
        fprintf(stderr, "phantom-gguf: bad magic (got 0x%08x, expected 0x%08x)\n",
                magic, GGUF_MAGIC);
        return PHANTOM_ERR_INVALID;
    }

    /* Version */
    if (read_u32(fd, &out->version) != 0) return PHANTOM_ERR_IO;
    if (out->version < 2 || out->version > 3) {
        fprintf(stderr, "phantom-gguf: unsupported version %u\n", out->version);
        return PHANTOM_ERR_INVALID;
    }

    /* Counts */
    if (read_u64(fd, &out->n_tensors) != 0) return PHANTOM_ERR_IO;
    if (read_u64(fd, &out->n_kv) != 0) return PHANTOM_ERR_IO;

    /* Skip KV metadata — we don't need it for layer mapping */
    for (uint64_t i = 0; i < out->n_kv; i++) {
        char key[256];
        if (read_gguf_string(fd, key, sizeof(key)) != 0) return PHANTOM_ERR_IO;

        uint32_t value_type;
        if (read_u32(fd, &value_type) != 0) return PHANTOM_ERR_IO;
        if (skip_gguf_value(fd, value_type) != 0) return PHANTOM_ERR_IO;
    }

    /* Read tensor info */
    out->tensors = calloc(out->n_tensors, sizeof(phantom_tensor_info_t));
    if (!out->tensors) return PHANTOM_ERR_OOM;

    int max_layer = -1;

    for (uint64_t i = 0; i < out->n_tensors; i++) {
        phantom_tensor_info_t *t = &out->tensors[i];

        /* Name */
        if (read_gguf_string(fd, t->name, sizeof(t->name)) != 0)
            return PHANTOM_ERR_IO;

        /* Dimensions */
        uint32_t n_dims;
        if (read_u32(fd, &n_dims) != 0) return PHANTOM_ERR_IO;
        t->n_dims = n_dims;

        uint64_t n_elements = 1;
        for (int d = 0; d < t->n_dims; d++) {
            if (read_u64(fd, &t->dims[d]) != 0) return PHANTOM_ERR_IO;
            n_elements *= t->dims[d];
        }

        /* Type */
        if (read_u32(fd, &t->type) != 0) return PHANTOM_ERR_IO;

        /* Offset (relative to data section start) */
        if (read_u64(fd, &t->offset) != 0) return PHANTOM_ERR_IO;

        /* Compute size */
        if (t->type < GGML_TYPE_COUNT) {
            t->size = (size_t)(n_elements * ggml_type_size[t->type]);
        } else {
            t->size = n_elements * 2; /* Fallback: assume fp16 */
        }

        /* Extract layer ID */
        t->layer_id = extract_layer_id(t->name);
        if (t->layer_id > max_layer) max_layer = t->layer_id;
    }

    /* Record where the data section starts (current file position, aligned) */
    off_t pos = lseek(fd, 0, SEEK_CUR);
    /* GGUF aligns data to 32 bytes */
    out->data_offset = (pos + 31) & ~31ULL;
    out->num_layers = max_layer + 1;

    return PHANTOM_OK;
}

void phantom_gguf_free(phantom_gguf_model_t *model)
{
    free(model->tensors);
    memset(model, 0, sizeof(*model));
}

void phantom_gguf_print_summary(const phantom_gguf_model_t *model)
{
    printf("phantom-gguf: GGUF v%u — %lu tensors, %d layers\n",
           model->version, (unsigned long)model->n_tensors, model->num_layers);
    printf("  data offset: 0x%lx\n", (unsigned long)model->data_offset);

    /* Per-layer size summary */
    size_t *layer_sizes = calloc(model->num_layers, sizeof(size_t));
    size_t non_layer_size = 0;

    for (uint64_t i = 0; i < model->n_tensors; i++) {
        const phantom_tensor_info_t *t = &model->tensors[i];
        if (t->layer_id >= 0 && t->layer_id < model->num_layers) {
            layer_sizes[t->layer_id] += t->size;
        } else {
            non_layer_size += t->size;
        }
    }

    printf("  non-layer tensors (embed/norm/output): %.1f MB\n",
           non_layer_size / (1024.0 * 1024.0));

    if (model->num_layers > 0) {
        printf("  layer 0 size: %.1f MB\n",
               layer_sizes[0] / (1024.0 * 1024.0));
        if (model->num_layers > 1) {
            printf("  layer %d size: %.1f MB\n",
                   model->num_layers - 1,
                   layer_sizes[model->num_layers - 1] / (1024.0 * 1024.0));
        }

        size_t total = non_layer_size;
        for (int i = 0; i < model->num_layers; i++) total += layer_sizes[i];
        printf("  total model size: %.1f GB\n", total / (1024.0 * 1024.0 * 1024.0));
    }

    free(layer_sizes);
}
