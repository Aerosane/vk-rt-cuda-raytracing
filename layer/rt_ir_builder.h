// rt_ir_builder.h — Builds IRProgram from SPIR-V ray query patterns
// Analyzes spirv_ray_query_rewriter's parsed state to extract RT intent
// and lower it into the flat IR instruction stream.
//
#pragma once
#include "rt_ir.h"
#include <cstring>
#include <cstdio>

namespace rt_ir {

// ═══════════════════════════════════════════════════════════════
// IR Builder — constructs programs from high-level RT patterns
// ═══════════════════════════════════════════════════════════════

class Builder {
public:
    Builder() { reset(); }

    void reset() {
        memset(&prog_, 0, sizeof(prog_));
        prog_.magic   = IR_MAGIC;
        prog_.version = IR_VERSION;
        prog_.maxDepth = 4;  // default
        nextSlot_ = 0;
    }

    void setResolution(uint32_t w, uint32_t h) {
        prog_.width = w;
        prog_.height = h;
    }

    void setMaxDepth(uint32_t d) { prog_.maxDepth = d; }

    // ─── Slot allocation ─────────────────────────────────────
    uint16_t allocSlot(SlotType type) {
        if (nextSlot_ >= IR_MAX_SLOTS) return 0xFFFF;
        uint16_t s = nextSlot_++;
        prog_.slots[s].type = type;
        prog_.slotCount = nextSlot_;
        return s;
    }

    // ─── Constant pool ───────────────────────────────────────
    uint16_t addConst(float x, float y = 0, float z = 0, float w = 0) {
        if (prog_.constCount >= IR_MAX_CONSTS) return 0xFFFF;
        uint16_t idx = prog_.constCount++;
        prog_.consts[idx * 4 + 0] = x;
        prog_.consts[idx * 4 + 1] = y;
        prog_.consts[idx * 4 + 2] = z;
        prog_.consts[idx * 4 + 3] = w;
        return idx;
    }

    // ─── Emit IR nodes ───────────────────────────────────────

    // Camera/primary ray generation
    uint16_t emitMakeRay(uint16_t originSlot, uint16_t dirSlot) {
        uint16_t out = allocSlot(SLOT_RAY);
        emit(OP_MAKE_RAY, originSlot, dirSlot, out);
        return out;
    }

    // Shadow ray toward a light
    uint16_t emitMakeShadowRay(uint16_t hitSlot, uint16_t lightSlot) {
        uint16_t out = allocSlot(SLOT_RAY);
        emit(OP_MAKE_SHADOW_RAY, hitSlot, lightSlot, out);
        return out;
    }

    // Trace closest hit
    uint16_t emitTraceClosest(uint16_t raySlot) {
        uint16_t out = allocSlot(SLOT_HIT);
        emit(OP_TRACE_CLOSEST, raySlot, 0, out);
        return out;
    }

    // Trace any hit (shadow)
    uint16_t emitTraceAny(uint16_t raySlot) {
        uint16_t out = allocSlot(SLOT_BOOL);
        emit(OP_TRACE_ANY, raySlot, 0, out);
        return out;
    }

    // Shading operations
    uint16_t emitShadeDiffuse(uint16_t hitSlot, uint16_t matSlot) {
        uint16_t out = allocSlot(SLOT_FLOAT3);
        emit(OP_SHADE_DIFFUSE, hitSlot, matSlot, out);
        return out;
    }

    uint16_t emitShadeSpecular(uint16_t hitSlot, uint16_t matSlot) {
        uint16_t out = allocSlot(SLOT_FLOAT3);
        emit(OP_SHADE_SPECULAR, hitSlot, matSlot, out);
        return out;
    }

    // Light sampling
    uint16_t emitSampleLight(uint16_t hitSlot) {
        uint16_t out = allocSlot(SLOT_FLOAT3);
        emit(OP_SAMPLE_LIGHT, hitSlot, 0, out);
        return out;
    }

    uint16_t emitSampleEnvironment(uint16_t raySlot) {
        uint16_t out = allocSlot(SLOT_FLOAT3);
        emit(OP_SAMPLE_ENVIRONMENT, raySlot, 0, out);
        return out;
    }

    // Accumulation
    void emitAccumulate(uint16_t payloadSlot, uint16_t radianceSlot) {
        emit(OP_ACCUMULATE, payloadSlot, radianceSlot, payloadSlot);
    }

    // Russian roulette termination
    void emitRussianRoulette(uint16_t payloadSlot) {
        emit(OP_RUSSIAN_ROULETTE, payloadSlot, 0, payloadSlot);
    }

    // Reflection/refraction bounce
    uint16_t emitReflect(uint16_t dirSlot, uint16_t normalSlot) {
        uint16_t out = allocSlot(SLOT_FLOAT3);
        emit(OP_REFLECT, dirSlot, normalSlot, out);
        return out;
    }

    uint16_t emitRefract(uint16_t dirSlot, uint16_t normalSlot) {
        uint16_t out = allocSlot(SLOT_FLOAT3);
        emit(OP_REFRACT, dirSlot, normalSlot, out);
        return out;
    }

    // Branch if hit
    void emitBranch(uint16_t condSlot, uint16_t targetNode) {
        Node n = {};
        n.op = OP_BRANCH;
        n.in0 = condSlot;
        n.branchTarget = targetNode;
        appendNode(n);
    }

    void emitTerminate() {
        emit(OP_TERMINATE, 0, 0, 0);
    }

    // Denoise pass
    void emitDenoise(uint16_t fbSlot) {
        emit(OP_DENOISE, fbSlot, 0, fbSlot);
    }

    // Get current node count (for branch targets)
    uint32_t currentNode() const { return prog_.nodeCount; }

    // ─── Program templates ───────────────────────────────────

    // Build a standard path tracer program:
    //   for each bounce: trace → shade → accumulate → bounce
    void buildPathTracer(uint32_t maxBounces) {
        reset();
        prog_.maxDepth = maxBounces;

        // Slots: camera origin, camera dir, payload
        uint16_t originSlot = allocSlot(SLOT_FLOAT3);
        uint16_t dirSlot    = allocSlot(SLOT_FLOAT3);
        uint16_t payloadSlot = allocSlot(SLOT_PAYLOAD);

        for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
            // Make ray (first bounce = camera ray, later = bounce ray)
            uint16_t raySlot = emitMakeRay(originSlot, dirSlot);

            // Trace closest hit
            uint16_t hitSlot = emitTraceClosest(raySlot);

            // Branch: miss → sample environment → terminate
            uint32_t missTarget = currentNode() + 5; // skip shading block

            // Sample environment on miss (emitted but guarded by branch)
            uint16_t envSlot = emitSampleEnvironment(raySlot);
            emitAccumulate(payloadSlot, envSlot);

            // Shade on hit
            uint16_t matSlot = allocSlot(SLOT_MATERIAL);
            uint16_t lightSlot = emitSampleLight(hitSlot);
            uint16_t diffSlot = emitShadeDiffuse(hitSlot, matSlot);
            emitAccumulate(payloadSlot, diffSlot);

            // Shadow test
            uint16_t shadowRay = emitMakeShadowRay(hitSlot, lightSlot);
            emitTraceAny(shadowRay);

            // Bounce direction
            uint16_t normalSlot = allocSlot(SLOT_FLOAT3);
            uint16_t reflDir = emitReflect(dirSlot, normalSlot);

            // Update direction for next bounce
            dirSlot = reflDir;

            // Russian roulette after bounce 2
            if (bounce >= 2) emitRussianRoulette(payloadSlot);
        }

        emitTerminate();
    }

    // Build a simple shadow-only program (ambient occlusion):
    //   trace primary → sample N shadow rays → accumulate
    void buildAO(uint32_t numSamples) {
        reset();
        prog_.maxDepth = 1;

        uint16_t originSlot = allocSlot(SLOT_FLOAT3);
        uint16_t dirSlot    = allocSlot(SLOT_FLOAT3);
        uint16_t payloadSlot = allocSlot(SLOT_PAYLOAD);

        // Primary ray
        uint16_t raySlot = emitMakeRay(originSlot, dirSlot);
        uint16_t hitSlot = emitTraceClosest(raySlot);

        // Shadow samples (conceptual — actual count from constant)
        uint16_t aoConst = addConst((float)numSamples);
        uint16_t lightSlot = emitSampleLight(hitSlot);
        uint16_t shadowRay = emitMakeShadowRay(hitSlot, lightSlot);
        uint16_t occluded = emitTraceAny(shadowRay);
        emitAccumulate(payloadSlot, lightSlot);

        emitTerminate();
    }

    // ─── Finalize ────────────────────────────────────────────

    const Program& program() const { return prog_; }
    Program& program() { return prog_; }

    // Print program for debugging
    void dump(FILE* f = stderr) const {
        fprintf(f, "═══ IR Program ═══\n");
        fprintf(f, "  Nodes: %u  Slots: %u  Consts: %u  MaxDepth: %u\n",
                prog_.nodeCount, prog_.slotCount, prog_.constCount, prog_.maxDepth);
        if (prog_.width) fprintf(f, "  Resolution: %ux%u\n", prog_.width, prog_.height);
        for (uint32_t i = 0; i < prog_.nodeCount; i++) {
            const Node& n = prog_.nodes[i];
            fprintf(f, "  [%3u] %-20s in=(%u,%u) out=%u",
                    i, opName(n.op), n.in0, n.in1, n.out0);
            if (n.op == OP_BRANCH)
                fprintf(f, " → node %u", n.branchTarget);
            fprintf(f, "\n");
        }
        fprintf(f, "══════════════════\n");
    }

private:
    Program  prog_;
    uint16_t nextSlot_;

    void emit(Op op, uint16_t in0, uint16_t in1, uint16_t out0) {
        Node n = {};
        n.op = op;
        n.in0 = in0;
        n.in1 = in1;
        n.out0 = out0;
        appendNode(n);
    }

    void appendNode(const Node& n) {
        if (prog_.nodeCount < IR_MAX_NODES) {
            prog_.nodes[prog_.nodeCount++] = n;
        }
    }
};

} // namespace rt_ir
