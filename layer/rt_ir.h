// rt_ir.h — Minimal Ray Tracing Intermediate Representation
// A flat instruction stream representing ray tracing intent.
// NOT Vulkan, NOT SPIR-V, NOT CUDA — pure RT semantics.
//
// Pipeline:  SPIR-V → [IR Builder] → IRProgram → [Planner] → CUDA kernels
//
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define IR_HOST_DEVICE __host__ __device__
#else
#define IR_HOST_DEVICE
#endif

namespace rt_ir {

// ═══════════════════════════════════════════════════════════════
// Core data types — POD, compact, GPU-friendly
// ═══════════════════════════════════════════════════════════════

struct IRFloat3 { float x, y, z; };

struct Ray {
    IRFloat3 origin;
    IRFloat3 dir;
    float  tmin;
    float  tmax;
};

struct Hit {
    float  t;
    IRFloat3 normal;
    int    materialId;
    int    primitiveId;
    int    instanceId;
    int    hit;           // 0 = miss, 1 = hit
};

struct Payload {
    IRFloat3 radiance;
    IRFloat3 throughput;
    int    depth;
    int    flags;         // user-defined payload flags
};

struct Material {
    IRFloat3 albedo;
    float  roughness;
    float  metallic;
    float  ior;           // index of refraction
};

// ═══════════════════════════════════════════════════════════════
// IR Operations — the instruction set
// ═══════════════════════════════════════════════════════════════

enum Op : uint8_t {
    // Ray creation
    OP_MAKE_RAY,           // (origin, dir, tmin, tmax) → Ray
    OP_MAKE_SHADOW_RAY,    // (origin, lightPos) → Ray (tmax = distance)

    // Traversal
    OP_TRACE_CLOSEST,      // (Ray) → Hit  (closest hit, full traversal)
    OP_TRACE_ANY,          // (Ray) → bool (any hit = shadow test, early exit)

    // Shading
    OP_SHADE_DIFFUSE,      // (Hit, Material, lightDir) → radiance
    OP_SHADE_SPECULAR,     // (Hit, Material, viewDir) → radiance
    OP_SHADE_EMISSIVE,     // (Material) → radiance (self-emission)

    // Lighting
    OP_SAMPLE_LIGHT,       // (Hit) → lightDir, lightColor, pdf
    OP_SAMPLE_ENVIRONMENT, // (Ray) → envColor (sky/IBL)

    // Accumulation
    OP_ACCUMULATE,         // throughput *= brdf; radiance += emission
    OP_RUSSIAN_ROULETTE,   // probabilistic path termination

    // Bounce
    OP_REFLECT,            // (viewDir, normal) → reflectDir
    OP_REFRACT,            // (viewDir, normal, ior) → refractDir

    // Control
    OP_TERMINATE,          // end this ray path
    OP_BRANCH,             // conditional: if Hit.hit → jump to node

    // Post-process (tensor pipeline)
    OP_DENOISE,            // (framebuffer) → denoised output
    OP_ACCUMULATE_FRAME,   // temporal accumulation

    // Neural Radiance Cache (tensor core accelerated)
    OP_NRC_QUERY,          // (position, normal) → cached indirect radiance
    OP_NRC_TRAIN_SAMPLE,   // collect (position, radiance) for per-frame training

    OP_COUNT
};

// Human-readable op names (for debugging/logging)
inline const char* opName(Op op) {
    static const char* names[] = {
        "MAKE_RAY", "MAKE_SHADOW_RAY",
        "TRACE_CLOSEST", "TRACE_ANY",
        "SHADE_DIFFUSE", "SHADE_SPECULAR", "SHADE_EMISSIVE",
        "SAMPLE_LIGHT", "SAMPLE_ENVIRONMENT",
        "ACCUMULATE", "RUSSIAN_ROULETTE",
        "REFLECT", "REFRACT",
        "TERMINATE", "BRANCH",
        "DENOISE", "ACCUMULATE_FRAME",
        "NRC_QUERY", "NRC_TRAIN_SAMPLE",
    };
    return (op < OP_COUNT) ? names[op] : "UNKNOWN";
}

// ═══════════════════════════════════════════════════════════════
// IR Node — single instruction in the stream
// ═══════════════════════════════════════════════════════════════

struct Node {
    Op       op;
    uint8_t  flags;     // per-op modifier bits
    uint16_t in0;       // input slot index 0
    uint16_t in1;       // input slot index 1
    uint16_t out0;      // output slot index

    // Extended operand for BRANCH target / constant index
    union {
        uint16_t branchTarget;  // OP_BRANCH: node index to jump to
        uint16_t constIdx;      // index into constant pool
        uint16_t extra;
    };
};
static_assert(sizeof(Node) == 10, "IR Node should be 10 bytes");

// ═══════════════════════════════════════════════════════════════
// Slot types — what's stored in each slot
// ═══════════════════════════════════════════════════════════════

enum SlotType : uint8_t {
    SLOT_EMPTY = 0,
    SLOT_RAY,
    SLOT_HIT,
    SLOT_PAYLOAD,
    SLOT_MATERIAL,
    SLOT_FLOAT3,
    SLOT_FLOAT,
    SLOT_INT,
    SLOT_BOOL,
};

struct SlotInfo {
    SlotType type;
    uint8_t  pad[3];
};

// ═══════════════════════════════════════════════════════════════
// IR Program — the complete instruction stream
// ═══════════════════════════════════════════════════════════════

static constexpr uint32_t IR_MAX_NODES    = 4096;
static constexpr uint32_t IR_MAX_SLOTS    = 1024;
static constexpr uint32_t IR_MAX_CONSTS   = 256;
static constexpr uint32_t IR_MAGIC        = 0x52544952;  // "RTIR"
static constexpr uint32_t IR_VERSION      = 1;

struct Program {
    uint32_t magic;
    uint32_t version;
    uint32_t nodeCount;
    uint32_t slotCount;
    uint32_t constCount;

    // Bounce depth limit for the entire program
    uint32_t maxDepth;

    // Resolution this program targets (0 = any)
    uint32_t width, height;

    Node     nodes[IR_MAX_NODES];
    SlotInfo slots[IR_MAX_SLOTS];

    // Constant pool: float4-aligned values
    float    consts[IR_MAX_CONSTS * 4];

    // Validate program structure
    IR_HOST_DEVICE bool valid() const {
        return magic == IR_MAGIC && version == IR_VERSION && nodeCount > 0;
    }
};

// ═══════════════════════════════════════════════════════════════
// SPIR-V → IR mapping reference
// ═══════════════════════════════════════════════════════════════
//
// | SPIR-V OpCode              | IR Op              |
// |----------------------------|--------------------|
// | OpRayQueryInitializeKHR    | OP_MAKE_RAY        |
// | OpRayQueryProceedKHR       | OP_TRACE_CLOSEST   |
// | OpRayQueryGetIntersection* | (Hit fields)       |
// | (shader math)              | OP_SHADE_*         |
// | OpTerminateRayKHR          | OP_TERMINATE       |
//
// The IR builder analyzes SPIR-V control flow to identify:
// 1. Ray generation patterns (camera rays, shadow rays, bounce rays)
// 2. Hit processing blocks (material lookup, shading)
// 3. Accumulation patterns (throughput × BRDF integration)
// 4. Termination conditions (max depth, Russian roulette)

} // namespace rt_ir
