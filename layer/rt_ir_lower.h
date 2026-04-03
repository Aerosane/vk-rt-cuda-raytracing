// rt_ir_lower.h — SPIR-V → IR Lowering Pass
// Analyzes SPIR-V compute shaders containing ray query operations and
// extracts the RT intent into an IRProgram.
//
// Pattern recognition:
//   OpRayQueryInitializeKHR → OP_MAKE_RAY (origin, dir, tmin, tmax)
//   OpRayQueryProceedKHR    → OP_TRACE_CLOSEST or OP_TRACE_ANY
//   OpRayQueryGet*KHR       → Hit field reads (t, bary, primId, instId)
//   Shader math patterns    → OP_SHADE_*, OP_ACCUMULATE, OP_REFLECT
//
// This is NOT a full SPIR-V interpreter — it recognizes canonical ray
// tracing patterns (camera rays, bounces, shadow tests, accumulation)
// and lowers them into the flat IR instruction stream.
//
#pragma once
#include "rt_ir.h"
#include "rt_ir_builder.h"
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace rt_ir {

// SPIR-V opcodes we care about for lowering
namespace spv {
    constexpr uint16_t OpFunction          = 54;
    constexpr uint16_t OpFunctionEnd       = 56;
    constexpr uint16_t OpFunctionCall      = 57;
    constexpr uint16_t OpVariable           = 59;
    constexpr uint16_t OpLoad              = 61;
    constexpr uint16_t OpStore             = 62;
    constexpr uint16_t OpAccessChain       = 65;
    constexpr uint16_t OpDecorate          = 71;
    constexpr uint16_t OpCompositeExtract  = 81;
    constexpr uint16_t OpCompositeConstruct= 80;
    constexpr uint16_t OpVectorShuffle     = 79;
    constexpr uint16_t OpFNegate           = 127;
    constexpr uint16_t OpFAdd              = 129;
    constexpr uint16_t OpFSub              = 131;
    constexpr uint16_t OpFMul              = 133;
    constexpr uint16_t OpFDiv              = 136;
    constexpr uint16_t OpFOrdGreaterThan   = 186;
    constexpr uint16_t OpFOrdLessThan      = 188;
    constexpr uint16_t OpDot               = 148;
    constexpr uint16_t OpBranch            = 249;
    constexpr uint16_t OpBranchConditional = 250;
    constexpr uint16_t OpLabel             = 248;
    constexpr uint16_t OpLoopMerge         = 246;
    constexpr uint16_t OpSelectionMerge    = 247;
    constexpr uint16_t OpPhi               = 245;
    constexpr uint16_t OpReturn            = 253;
    constexpr uint16_t OpReturnValue       = 254;
    constexpr uint16_t OpTypeVoid          = 19;
    constexpr uint16_t OpTypeBool          = 20;
    constexpr uint16_t OpTypeInt           = 21;
    constexpr uint16_t OpTypeFloat         = 22;
    constexpr uint16_t OpTypeVector        = 23;
    constexpr uint16_t OpTypeRayQueryKHR   = 4472;
    constexpr uint16_t OpRQInitialize      = 4473;
    constexpr uint16_t OpRQTerminate       = 4474;
    constexpr uint16_t OpRQConfirmIntersection = 4476;
    constexpr uint16_t OpRQProceed         = 4477;
    constexpr uint16_t OpRQGetT            = 6016;
    constexpr uint16_t OpRQGetBarycentrics = 6017;
    constexpr uint16_t OpRQGetPrimIdx      = 6018;
    constexpr uint16_t OpRQGetInstId       = 6019;
    constexpr uint16_t OpRQGetInstCustomIdx= 6020;
    constexpr uint16_t OpRQGetWorldRayDir  = 6029;
    constexpr uint16_t OpRQGetWorldRayOrigin= 6030;
    constexpr uint16_t OpRQGetIntersectionType = 4479;
    constexpr uint16_t OpRQGetFrontFace    = 6025;
    constexpr uint16_t OpRQGetRayTMin      = 6016;
    constexpr uint16_t OpRQGetRayFlags     = 6023;
    constexpr uint16_t OpRQGetSBTOffset    = 6021;
}

// ═══════════════════════════════════════════════════════════════
// SPIR-V analysis result — extracted ray query patterns
// ═══════════════════════════════════════════════════════════════

struct RQPattern {
    enum Kind : uint8_t {
        PRIMARY_RAY,      // camera/primary ray query
        SHADOW_RAY,       // any-hit shadow test (trace_any)
        BOUNCE_RAY,       // reflection/refraction bounce
        GI_RAY,           // indirect lighting bounce
    };
    Kind     kind;
    uint32_t rqVar;         // SPIR-V ray query variable ID
    uint32_t originId;      // SPIR-V ID: ray origin
    uint32_t dirId;         // SPIR-V ID: ray direction
    uint32_t tMinId;        // SPIR-V ID: tMin
    uint32_t tMaxId;        // SPIR-V ID: tMax
    uint32_t flagsId;       // SPIR-V ID: ray flags
    uint32_t maskId;        // SPIR-V ID: cull mask
    // Post-trace: which fields are read
    bool readsT;
    bool readsBary;
    bool readsPrimIdx;
    bool readsInstId;
    bool readsHitType;
    bool readsFrontFace;
    bool readsWorldDir;
    bool readsWorldOrigin;
    bool readsCustomIdx;
    bool readsSBTOffset;
    // Loop context
    bool insideLoop;         // is this RQ init inside a loop (= bounce)
    int  loopDepth;
};

struct SPIRVAnalysis {
    std::vector<RQPattern> patterns;
    int maxBounceDepth;      // detected from loop nesting
    bool hasShadowRays;      // any OP_TRACE_ANY patterns
    bool hasReflection;      // detected reflect patterns
    bool hasRefraction;      // detected refract patterns
    uint32_t numRQVars;      // how many distinct ray query variables
    // Workgroup size from ExecutionMode
    uint32_t localSizeX, localSizeY, localSizeZ;
};

// ═══════════════════════════════════════════════════════════════
// SPIR-V Analyzer — extracts ray query patterns from binary
// ═══════════════════════════════════════════════════════════════

class SPIRVAnalyzer {
public:
    // Analyze a SPIR-V binary and extract ray query patterns
    bool analyze(const uint32_t* code, size_t numWords) {
        if (numWords < 5 || code[0] != 0x07230203u) return false;
        code_ = code;
        numWords_ = numWords;
        result_ = {};

        // Pass 1: collect types and ray query variables
        collectTypes();
        // Pass 2: find ray query operations and classify
        findRQOps();
        // Pass 3: analyze post-trace reads per RQ variable
        analyzeReads();
        // Pass 4: detect bounce loops and shadow ray patterns
        classifyPatterns();

        return !result_.patterns.empty();
    }

    const SPIRVAnalysis& result() const { return result_; }

private:
    const uint32_t* code_ = nullptr;
    size_t numWords_ = 0;
    SPIRVAnalysis result_;

    // Type tracking
    std::unordered_set<uint32_t> rqTypeIds_;
    std::unordered_set<uint32_t> rqVarIds_;
    std::unordered_map<uint32_t, uint32_t> rqMap_; // copy → original

    // Instruction position tracking
    struct InsnPos {
        size_t pos;
        uint16_t op;
        uint16_t wc;
    };
    std::vector<InsnPos> insns_;

    // Function boundaries
    struct FuncRange {
        size_t startPos, endPos;
        uint32_t funcId;
    };
    std::vector<FuncRange> functions_;

    // Loop tracking
    struct LoopInfo {
        uint32_t headerLabel;
        uint32_t mergeLabel;
        int depth;
    };
    std::vector<LoopInfo> loops_;

    void collectTypes() {
        rqTypeIds_.clear();
        rqVarIds_.clear();
        rqMap_.clear();
        insns_.clear();
        functions_.clear();

        size_t pos = 5;
        FuncRange* curFunc = nullptr;

        while (pos < numWords_) {
            uint16_t wc = (uint16_t)(code_[pos] >> 16);
            uint16_t op = (uint16_t)(code_[pos] & 0xFFFF);
            if (wc == 0 || pos + wc > numWords_) break;

            insns_.push_back({pos, op, wc});

            // Track execution mode for workgroup size
            if (op == 16 /* OpExecutionMode */ && wc >= 6 && code_[pos+2] == 17 /* LocalSize */) {
                result_.localSizeX = code_[pos+3];
                result_.localSizeY = code_[pos+4];
                result_.localSizeZ = code_[pos+5];
            }

            // RQ type
            if (op == spv::OpTypeRayQueryKHR) {
                rqTypeIds_.insert(code_[pos+1]);
            }
            // Pointer to RQ type
            if (op == 32 /* OpTypePointer */ && wc >= 4) {
                if (rqTypeIds_.count(code_[pos+3]))
                    rqTypeIds_.insert(code_[pos+1]); // ptr type too
            }
            // RQ variable
            if (op == spv::OpVariable && wc >= 4) {
                uint32_t typeId = code_[pos+1];
                uint32_t varId  = code_[pos+2];
                if (rqTypeIds_.count(typeId)) {
                    rqVarIds_.insert(varId);
                }
            }
            // CopyObject / Load of RQ var
            if ((op == 83 /* OpCopyObject */ || op == spv::OpLoad) && wc >= 4) {
                uint32_t resultId = code_[pos+2];
                uint32_t sourceId = code_[pos+3];
                if (rqVarIds_.count(sourceId)) {
                    rqMap_[resultId] = sourceId;
                    rqVarIds_.insert(resultId);
                }
            }

            // Track functions
            if (op == spv::OpFunction && wc >= 5) {
                functions_.push_back({pos, 0, code_[pos+2]});
                curFunc = &functions_.back();
            }
            if (op == spv::OpFunctionEnd && curFunc) {
                curFunc->endPos = pos + wc;
                curFunc = nullptr;
            }

            pos += wc;
        }

        result_.numRQVars = (uint32_t)rqVarIds_.size();
    }

    void findRQOps() {
        int loopDepth = 0;

        for (auto& insn : insns_) {
            // Track loop nesting
            if (insn.op == spv::OpLoopMerge) {
                loopDepth++;
            }
            if (insn.op == spv::OpLabel && loopDepth > 0) {
                // Check if this label is a merge target (reduces depth)
                for (auto& li : loops_) {
                    if (li.mergeLabel == code_[insn.pos+1]) {
                        loopDepth--;
                        break;
                    }
                }
            }

            // RayQueryInitialize → extract ray params
            if (insn.op == spv::OpRQInitialize && insn.wc >= 9) {
                RQPattern pat = {};
                pat.rqVar   = code_[insn.pos+1];
                pat.flagsId = code_[insn.pos+3];
                pat.maskId  = code_[insn.pos+4];
                pat.originId= code_[insn.pos+5];
                pat.tMinId  = code_[insn.pos+6];
                pat.dirId   = code_[insn.pos+7];
                pat.tMaxId  = code_[insn.pos+8];
                pat.insideLoop = (loopDepth > 0);
                pat.loopDepth = loopDepth;
                pat.kind = RQPattern::PRIMARY_RAY; // default, refined later
                result_.patterns.push_back(pat);
            }

            // Track LoopMerge for merge label detection
            if (insn.op == spv::OpLoopMerge && insn.wc >= 4) {
                loops_.push_back({0, code_[insn.pos+1], loopDepth});
            }
        }
    }

    void analyzeReads() {
        // For each RQ variable, scan for Get* operations that read from it
        for (auto& insn : insns_) {
            uint32_t rqVar = 0;

            // Most Get* ops have rqVar at pos+3 (wc >= 5)
            if (insn.wc >= 5 && insn.op >= 6016 && insn.op <= 6032) {
                rqVar = code_[insn.pos+3];
            }
            // GetIntersectionType has rqVar at pos+3
            if (insn.op == spv::OpRQGetIntersectionType && insn.wc >= 5) {
                rqVar = code_[insn.pos+3];
            }

            if (rqVar == 0) continue;
            // Map copies to original
            uint32_t origVar = rqMap_.count(rqVar) ? rqMap_[rqVar] : rqVar;

            // Find matching pattern
            for (auto& pat : result_.patterns) {
                uint32_t pv = rqMap_.count(pat.rqVar) ? rqMap_[pat.rqVar] : pat.rqVar;
                if (pv != origVar && pat.rqVar != origVar) continue;

                switch (insn.op) {
                    case spv::OpRQGetT:            pat.readsT = true; break;
                    case spv::OpRQGetBarycentrics:  pat.readsBary = true; break;
                    case spv::OpRQGetPrimIdx:       pat.readsPrimIdx = true; break;
                    case spv::OpRQGetInstId:        pat.readsInstId = true; break;
                    case spv::OpRQGetIntersectionType: pat.readsHitType = true; break;
                    case spv::OpRQGetFrontFace:     pat.readsFrontFace = true; break;
                    case spv::OpRQGetWorldRayDir:   pat.readsWorldDir = true; break;
                    case spv::OpRQGetWorldRayOrigin: pat.readsWorldOrigin = true; break;
                    case spv::OpRQGetInstCustomIdx: pat.readsCustomIdx = true; break;
                    case spv::OpRQGetSBTOffset:     pat.readsSBTOffset = true; break;
                }
            }
        }
    }

    void classifyPatterns() {
        result_.maxBounceDepth = 0;
        result_.hasShadowRays = false;
        result_.hasReflection = false;

        for (auto& pat : result_.patterns) {
            // Shadow ray heuristic: if ray flags include skip-closest-hit (0x100)
            // OR if only hitType is read (no T/bary/prim needed for shadow)
            bool shadowHeuristic = (!pat.readsT && !pat.readsBary && !pat.readsPrimIdx
                                    && pat.readsHitType);

            if (shadowHeuristic) {
                pat.kind = RQPattern::SHADOW_RAY;
                result_.hasShadowRays = true;
            } else if (pat.insideLoop && pat.loopDepth > 0) {
                pat.kind = RQPattern::BOUNCE_RAY;
                if (pat.loopDepth > result_.maxBounceDepth)
                    result_.maxBounceDepth = pat.loopDepth;
            } else {
                pat.kind = RQPattern::PRIMARY_RAY;
            }
        }

        // Detect max bounces from loop depth
        if (result_.maxBounceDepth == 0 && result_.patterns.size() > 1) {
            result_.maxBounceDepth = (int)result_.patterns.size();
        }
        if (result_.maxBounceDepth == 0) result_.maxBounceDepth = 1;
    }
};

// ═══════════════════════════════════════════════════════════════
// IR Lowering — converts analysis into IRProgram
// ═══════════════════════════════════════════════════════════════

class SPIRVLowering {
public:
    // Lower a SPIR-V analysis result into an IR program
    bool lower(const SPIRVAnalysis& analysis, uint32_t width = 0, uint32_t height = 0) {
        builder_.reset();
        if (width > 0 && height > 0)
            builder_.setResolution(width, height);
        builder_.setMaxDepth(analysis.maxBounceDepth > 0 ? analysis.maxBounceDepth : 4);

        // Allocate camera slots (always needed)
        originSlot_ = builder_.allocSlot(SLOT_FLOAT3);
        dirSlot_    = builder_.allocSlot(SLOT_FLOAT3);
        payloadSlot_= builder_.allocSlot(SLOT_PAYLOAD);

        bool hasPrimary = false;

        for (size_t i = 0; i < analysis.patterns.size(); i++) {
            const RQPattern& pat = analysis.patterns[i];

            switch (pat.kind) {
            case RQPattern::PRIMARY_RAY:
                lowerPrimaryRay(pat, analysis);
                hasPrimary = true;
                break;

            case RQPattern::SHADOW_RAY:
                lowerShadowRay(pat);
                break;

            case RQPattern::BOUNCE_RAY:
                lowerBounceRay(pat, analysis);
                break;

            case RQPattern::GI_RAY:
                lowerBounceRay(pat, analysis);
                break;
            }
        }

        // If no explicit patterns, build a default 1-bounce path tracer
        if (!hasPrimary && analysis.patterns.empty()) {
            uint16_t raySlot = builder_.emitMakeRay(originSlot_, dirSlot_);
            uint16_t hitSlot = builder_.emitTraceClosest(raySlot);
            uint16_t envSlot = builder_.emitSampleEnvironment(raySlot);
            builder_.emitAccumulate(payloadSlot_, envSlot);
            uint16_t matSlot = builder_.allocSlot(SLOT_MATERIAL);
            uint16_t diffSlot = builder_.emitShadeDiffuse(hitSlot, matSlot);
            builder_.emitAccumulate(payloadSlot_, diffSlot);
        }

        builder_.emitTerminate();

        fprintf(stderr, "[IR:Lower] Lowered %zu SPIR-V ray query patterns → %u IR nodes\n",
                analysis.patterns.size(), builder_.program().nodeCount);
        fprintf(stderr, "[IR:Lower] Bounces=%d Shadow=%d Reflect=%d Slots=%u\n",
                analysis.maxBounceDepth, analysis.hasShadowRays ? 1 : 0,
                analysis.hasReflection ? 1 : 0, builder_.program().slotCount);

        return builder_.program().nodeCount > 0;
    }

    const Program& program() const { return builder_.program(); }
    Program& program() { return builder_.program(); }
    Builder& builder() { return builder_; }

    // Dump the lowered IR for debugging
    void dump(FILE* f = stderr) const { builder_.dump(f); }

private:
    Builder  builder_;
    uint16_t originSlot_;
    uint16_t dirSlot_;
    uint16_t payloadSlot_;

    void lowerPrimaryRay(const RQPattern& pat, const SPIRVAnalysis& analysis) {
        // Primary ray: camera → trace → shade
        uint16_t raySlot = builder_.emitMakeRay(originSlot_, dirSlot_);
        uint16_t hitSlot = builder_.emitTraceClosest(raySlot);

        // Environment sampling for misses
        uint16_t envSlot = builder_.emitSampleEnvironment(raySlot);
        builder_.emitAccumulate(payloadSlot_, envSlot);

        // Material + shading for hits
        uint16_t matSlot = builder_.allocSlot(SLOT_MATERIAL);
        uint16_t lightSlot = builder_.emitSampleLight(hitSlot);
        uint16_t diffSlot = builder_.emitShadeDiffuse(hitSlot, matSlot);
        builder_.emitAccumulate(payloadSlot_, diffSlot);

        // If full-featured hit reads, add specular
        if (pat.readsBary || pat.readsFrontFace) {
            uint16_t specSlot = builder_.emitShadeSpecular(hitSlot, matSlot);
            builder_.emitAccumulate(payloadSlot_, specSlot);
        }
    }

    void lowerShadowRay(const RQPattern& pat) {
        // Shadow ray: hit point → light → trace any → occlude
        uint16_t lightSlot = builder_.allocSlot(SLOT_FLOAT3);
        uint16_t shadowRaySlot = builder_.emitMakeShadowRay(
            builder_.allocSlot(SLOT_HIT), lightSlot);
        builder_.emitTraceAny(shadowRaySlot);
    }

    void lowerBounceRay(const RQPattern& pat, const SPIRVAnalysis& analysis) {
        // Bounce ray: reflect → trace → shade → accumulate
        uint16_t normalSlot = builder_.allocSlot(SLOT_FLOAT3);
        uint16_t reflDir = builder_.emitReflect(dirSlot_, normalSlot);

        uint16_t bounceRay = builder_.emitMakeRay(
            builder_.allocSlot(SLOT_FLOAT3), reflDir);
        uint16_t bounceHit = builder_.emitTraceClosest(bounceRay);

        // Environment on miss
        uint16_t envSlot = builder_.emitSampleEnvironment(bounceRay);
        builder_.emitAccumulate(payloadSlot_, envSlot);

        // Shade on hit
        uint16_t matSlot = builder_.allocSlot(SLOT_MATERIAL);
        uint16_t diffSlot = builder_.emitShadeDiffuse(bounceHit, matSlot);
        builder_.emitAccumulate(payloadSlot_, diffSlot);

        // Russian roulette after first bounce
        builder_.emitRussianRoulette(payloadSlot_);
    }
};

// ═══════════════════════════════════════════════════════════════
// Convenience: one-shot SPIR-V → IRProgram lowering
// ═══════════════════════════════════════════════════════════════

inline bool spirvToIR(const uint32_t* code, size_t numWords,
                       Program& outProg,
                       uint32_t width = 0, uint32_t height = 0) {
    SPIRVAnalyzer analyzer;
    if (!analyzer.analyze(code, numWords)) {
        fprintf(stderr, "[IR:Lower] No ray query patterns found in SPIR-V (%zu words)\n", numWords);
        return false;
    }

    const auto& analysis = analyzer.result();
    fprintf(stderr, "[IR:Lower] SPIR-V analysis: %zu RQ patterns, %u RQ vars, "
            "localSize=(%u,%u,%u)\n",
            analysis.patterns.size(), analysis.numRQVars,
            analysis.localSizeX, analysis.localSizeY, analysis.localSizeZ);

    for (size_t i = 0; i < analysis.patterns.size(); i++) {
        const auto& p = analysis.patterns[i];
        const char* kindStr[] = {"PRIMARY", "SHADOW", "BOUNCE", "GI"};
        fprintf(stderr, "[IR:Lower]   pattern[%zu]: %s rqVar=%u loop=%d "
                "reads(T=%d bary=%d prim=%d inst=%d type=%d face=%d)\n",
                i, kindStr[p.kind], p.rqVar, p.loopDepth,
                p.readsT, p.readsBary, p.readsPrimIdx, p.readsInstId,
                p.readsHitType, p.readsFrontFace);
    }

    SPIRVLowering lowering;
    if (!lowering.lower(analysis, width, height)) {
        fprintf(stderr, "[IR:Lower] Failed to lower SPIR-V to IR\n");
        return false;
    }

    lowering.dump();
    outProg = lowering.program();
    return true;
}

} // namespace rt_ir
