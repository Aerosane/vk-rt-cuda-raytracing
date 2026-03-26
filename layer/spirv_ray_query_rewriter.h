#pragma once
// SPIR-V Ray Query -> Software BVH Traversal Rewriter
// Set to 0 for no-hit stub (debug), 1 for full BVH2 traversal
#define SPIRV_RQ_FULL_TRAVERSAL 1
// Set to 1 to skip BLAS traversal — report a hit as soon as TLAS leaf (instance AABB) is hit.
#define SPIRV_RQ_TLAS_ONLY 0  // TEMP: test TLAS traversal only
// Set to 1 to ALWAYS report hit — bypasses traversal entirely (tests RT pipeline feedback path)
#define SPIRV_RQ_ALWAYS_HIT 0
// Set to 1 to force TLAS AABB test to always pass
#define SPIRV_RQ_FORCE_TLAS_HIT 0
// Set to 1 to dump rewritten SPIR-V to /tmp/spirv_rq_dump_*.spv
#define SPIRV_RQ_DUMP 1
// Set to 1 to test SSBO read: load root node word 0, if non-zero → hit
#define SPIRV_RQ_SSBO_TEST 0
// Set to 1 to suppress BVH SSBO injection (debug: test if SSBOs cause issues)
#define SPIRV_RQ_NO_SSBO 0
// Set to 1 to keep original capabilities, extensions, types, variables, decorations
// (only replace RQ operations). Tests if stripping vs replacement causes dark frames.
#define SPIRV_RQ_KEEP_TYPES 0
// MINIMAL mode: when 1, skips full rewrite code and uses simpler RQ op handling.
// Set to 0 for full BVH traversal rewrite in function bodies.
#define SPIRV_RQ_MINIMAL 0
// PASSTHROUGH mode: ONLY stub function calls to RQ functions with 1.0.
// No type/cap/ext stripping, no RQ op replacement, no entry point modification.
// Tests whether call stubbing alone vs everything else causes dark frames.
#define SPIRV_RQ_PASSTHROUGH 0
// Intercepts shaders containing rayQueryEXT and rewrites them to use
// our BVH2 stackless traversal via SSBOs. Makes ray query work on ANY
// GPU without hardware RT cores.
//
// Strategy:
//  1. Scan SPIR-V for OpTypeRayQueryKHR (opcode 4472)
//  2. If found, patch the shader:
//     a. Remove RayQueryKHR capability + AS types
//     b. Replace OpTypeRayQueryKHR with a struct holding traversal state
//     c. Add SSBO descriptors for BVH nodes + triangle data
//     d. Replace OpRayQueryInitializeKHR with full BVH2 traversal (eager)
//     e. Replace OpRayQueryProceedKHR with return-false (traversal done)
//     f. Replace OpRayQueryGet*KHR with struct field reads
//  3. Return modified SPIR-V — app never knows RT was software

#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

// SPIR-V opcodes
enum SpvRQ : uint16_t {
    SpvOpNop                    = 0,
    SpvOpExtInstImport          = 11,
    SpvOpMemoryModel            = 14,
    SpvOpEntryPoint             = 15,
    SpvOpExecutionMode          = 16,
    SpvOpCapability             = 17,
    SpvOpTypeVoid               = 19,
    SpvOpTypeBool               = 20,
    SpvOpTypeInt                = 21,
    SpvOpTypeFloat              = 22,
    SpvOpTypeVector             = 23,
    SpvOpTypeArray              = 28,
    SpvOpTypeRuntimeArray       = 29,
    SpvOpTypeStruct             = 30,
    SpvOpTypePointer            = 32,
    SpvOpTypeFunction           = 33,
    SpvOpConstantTrue           = 41,
    SpvOpConstantFalse          = 42,
    SpvOpConstant               = 43,
    SpvOpFunction               = 54,
    SpvOpFunctionEnd            = 56,
    SpvOpVariable               = 59,
    SpvOpLoad                   = 61,
    SpvOpStore                  = 62,
    SpvOpAccessChain            = 65,
    SpvOpDecorate               = 71,
    SpvOpMemberDecorate         = 72,
    SpvOpCopyObject             = 83,
    SpvOpFNegate                = 127,
    SpvOpIAdd                   = 128,
    SpvOpFAdd                   = 129,
    SpvOpISub                   = 130,
    SpvOpFSub                   = 131,
    SpvOpIMul                   = 132,
    SpvOpFMul                   = 133,
    SpvOpFDiv                   = 136,
    SpvOpSRem                   = 138,
    SpvOpFMod                   = 141,
    SpvOpVectorShuffle          = 79,
    SpvOpCompositeConstruct     = 80,
    SpvOpCompositeExtract       = 81,
    SpvOpIEqual                 = 170,
    SpvOpINotEqual              = 171,
    SpvOpSGreaterThan           = 173,
    SpvOpSGreaterThanEqual      = 175,
    SpvOpSLessThan              = 177,
    SpvOpFOrdNotEqual           = 182,
    SpvOpFOrdLessThan           = 184,
    SpvOpFOrdGreaterThan        = 186,
    SpvOpFOrdLessThanEqual      = 188,
    SpvOpFOrdGreaterThanEqual   = 190,
    SpvOpShiftRightArithmetic   = 195,
    SpvOpBitwiseAnd             = 199,
    SpvOpBitcast                = 124,
    SpvOpConvertSToF            = 111,
    SpvOpConvertFToS            = 110,
    SpvOpSelect                 = 169,
    SpvOpPhi                    = 245,
    SpvOpLoopMerge              = 246,
    SpvOpSelectionMerge          = 247,
    SpvOpLabel                  = 248,
    SpvOpBranch                 = 249,
    SpvOpBranchConditional      = 250,
    SpvOpReturn                 = 253,
    SpvOpExtInst                = 12,
    // KHR ray types
    SpvOpTypeAccelStructKHR     = 5341,
    SpvOpTypeRayQueryKHR        = 4472,
    // Ray query ops
    SpvOpRQInitialize           = 4473,
    SpvOpRQTerminate            = 4474,
    SpvOpRQGenerateIntersection = 4475,
    SpvOpRQConfirmIntersection  = 4476,
    SpvOpRQProceed              = 4477,
    SpvOpRQGetIntersectionType  = 4479,
    // Ray query get ops
    SpvOpRQGetRayTMin           = 6016,
    SpvOpRQGetRayFlags          = 6017,
    SpvOpRQGetT                 = 6018,
    SpvOpRQGetInstCustomIdx     = 6019,
    SpvOpRQGetInstId            = 6020,
    SpvOpRQGetSBTOffset         = 6021,
    SpvOpRQGetGeomIdx           = 6022,
    SpvOpRQGetPrimIdx           = 6023,
    SpvOpRQGetBarycentrics      = 6024,
    SpvOpRQGetFrontFace         = 6025,
    SpvOpRQGetCandidateAABBOpaque = 6026,
    SpvOpRQGetObjRayDir         = 6027,
    SpvOpRQGetObjRayOrigin      = 6028,
    SpvOpRQGetWorldRayDir       = 6029,
    SpvOpRQGetWorldRayOrigin    = 6030,
    SpvOpRQGetObj2World         = 6031,
    SpvOpRQGetWorld2Obj         = 6032,
};

enum {
    CapRayQueryKHR = 4472,
    CapRayTraversalPrimitiveCulling = 4478,
    CapRayTracingKHR = 4479,
    DecDescriptorSet = 34,
    DecBinding = 33,
    DecNonWritable = 24,
    DecOffset = 35,
    DecArrayStride = 6,
    DecBlock = 2,
    SCStorageBuffer = 12,
    SCFunction = 7,
    // GLSL.std.450 extended instructions
    GLSLstd450FAbs = 4,
    GLSLstd450FSign = 6,
    GLSLstd450Cross = 68,
    GLSLstd450FMin = 37,
    GLSLstd450FMax = 40,
    GLSLstd450Fma = 50,
    GLSLstd450Dot = 148, // not an extInst — use OpDot directly
};

// Quick scan: does this SPIR-V contain ray query ops?
static bool spirvHasRayQuery(const uint32_t* code, size_t numWords) {
    if (numWords < 5 || code[0] != 0x07230203u) return false;
    size_t i = 5;
    while (i < numWords) {
        uint16_t wc = (uint16_t)(code[i] >> 16);
        uint16_t op = (uint16_t)(code[i] & 0xFFFF);
        if (wc == 0 || i + wc > numWords) break;
        if (op == SpvOpTypeRayQueryKHR || op == SpvOpRQInitialize || op == SpvOpRQProceed)
            return true;
        i += wc;
    }
    return false;
}

// Find highest descriptor set used
static int spirvMaxDescriptorSet(const uint32_t* code, size_t numWords) {
    int maxSet = -1;
    if (numWords < 5 || code[0] != 0x07230203u) return maxSet;
    size_t i = 5;
    while (i < numWords) {
        uint16_t wc = (uint16_t)(code[i] >> 16);
        uint16_t op = (uint16_t)(code[i] & 0xFFFF);
        if (wc == 0 || i + wc > numWords) break;
        if (op == SpvOpDecorate && wc >= 4 && code[i+2] == DecDescriptorSet) {
            int s = (int)code[i+3];
            if (s > maxSet) maxSet = s;
        }
        i += wc;
    }
    return maxSet;
}

// Helper: emit one SPIR-V instruction into output
static void spvEmit(std::vector<uint32_t>& out, uint16_t opcode,
                    const std::vector<uint32_t>& operands) {
    uint16_t wc = 1 + (uint16_t)operands.size();
    out.push_back(((uint32_t)wc << 16) | (uint32_t)opcode);
    for (auto w : operands) out.push_back(w);
}

// ============================================================================
// Main rewriter: transform ray query SPIR-V to use BVH SSBOs
// ============================================================================
static std::vector<uint32_t> spirvRewriteRayQuery(
    const uint32_t* code, size_t numWords,
    int bvhDescSet, int bvhNodesBinding, int bvhTrisBinding)
{
    if (numWords < 5 || code[0] != 0x07230203u) return {};

    // --- Diagnostic: dump all RQ opcodes in original shader ---
    {
        size_t dp = 5;
        while (dp < numWords) {
            uint16_t dwc = (uint16_t)(code[dp] >> 16);
            uint16_t dop = (uint16_t)(code[dp] & 0xFFFF);
            if (dwc == 0 || dp + dwc > numWords) break;
            if (dop == SpvOpRQInitialize)
                fprintf(stderr, "[SPIRV-DIAG] OpRQInitialize wc=%d rqVar=%u\n", dwc, code[dp+1]);
            else if (dop == SpvOpRQProceed)
                fprintf(stderr, "[SPIRV-DIAG] OpRQProceed wc=%d resultType=%u result=%u rq=%u\n",
                    dwc, code[dp+1], code[dp+2], code[dp+3]);
            else if (dop == SpvOpRQGetIntersectionType)
                fprintf(stderr, "[SPIRV-DIAG] OpRQGetIntersectionType wc=%d resultType=%u result=%u rq=%u intersection=%u\n",
                    dwc, code[dp+1], code[dp+2], dwc>=5 ? code[dp+3] : 0, dwc>=5 ? code[dp+4] : 0);
            else if (dop == SpvOpRQGetT)
                fprintf(stderr, "[SPIRV-DIAG] OpRQGetT wc=%d resultType=%u result=%u rq=%u intersection=%u\n",
                    dwc, code[dp+1], code[dp+2], dwc>=5 ? code[dp+3] : 0, dwc>=5 ? code[dp+4] : 0);
            else if (dop == SpvOpRQGetBarycentrics)
                fprintf(stderr, "[SPIRV-DIAG] OpRQGetBarycentrics wc=%d resultType=%u result=%u rq=%u intersection=%u\n",
                    dwc, code[dp+1], code[dp+2], dwc>=5 ? code[dp+3] : 0, dwc>=5 ? code[dp+4] : 0);
            else if (dop == SpvOpRQGetPrimIdx)
                fprintf(stderr, "[SPIRV-DIAG] OpRQGetPrimIdx wc=%d rq=%u intersection=%u\n",
                    dwc, dwc>=5 ? code[dp+3] : 0, dwc>=5 ? code[dp+4] : 0);
            else if (dop == SpvOpRQGetInstId)
                fprintf(stderr, "[SPIRV-DIAG] OpRQGetInstId wc=%d rq=%u intersection=%u\n",
                    dwc, dwc>=5 ? code[dp+3] : 0, dwc>=5 ? code[dp+4] : 0);
            else if (dop == SpvOpRQConfirmIntersection)
                fprintf(stderr, "[SPIRV-DIAG] OpRQConfirmIntersection wc=%d rq=%u\n", dwc, dwc>=2 ? code[dp+1] : 0);
            else if (dop == SpvOpRQTerminate)
                fprintf(stderr, "[SPIRV-DIAG] OpRQTerminate wc=%d rq=%u\n", dwc, dwc>=2 ? code[dp+1] : 0);
            else if (dop >= 4473 && dop <= 4479)
                fprintf(stderr, "[SPIRV-DIAG] RQ opcode=%u wc=%d\n", dop, dwc);
            else if (dop >= 6016 && dop <= 6032)
                fprintf(stderr, "[SPIRV-DIAG] RQ get opcode=%u wc=%d args:", dop, dwc);
            dp += dwc;
        }
        fprintf(stderr, "[SPIRV-DIAG] --- end scan ---\n");
    }

    uint32_t nextId = code[3]; // current Bound
    auto newId = [&]() -> uint32_t { return nextId++; };

    // --- Pass 1: collect type/variable info ---
    std::unordered_set<uint32_t> rqTypeIds, rqPtrTypeIds, rqVarIds;
    std::unordered_set<uint32_t> asTypeIds, asPtrTypeIds, asVarIds;
    uint32_t tVoid=0, tBool=0, tFloat=0, tUint=0, tInt=0;
    uint32_t tVec2=0, tVec3=0, tVec4=0, tUvec4=0;
    uint32_t cTrue=0, cFalse=0;

    size_t pos = 5;
    while (pos < numWords) {
        uint16_t wc = (uint16_t)(code[pos] >> 16);
        uint16_t op = (uint16_t)(code[pos] & 0xFFFF);
        if (wc == 0 || pos + wc > numWords) break;

        switch (op) {
        case SpvOpTypeVoid:  tVoid = code[pos+1]; break;
        case SpvOpTypeBool:  tBool = code[pos+1]; break;
        case SpvOpTypeFloat: if (wc>=3 && code[pos+2]==32) tFloat = code[pos+1]; break;
        case SpvOpTypeInt:
            if (wc>=4 && code[pos+2]==32) {
                if (code[pos+3]==0) tUint = code[pos+1];
                else                tInt  = code[pos+1];
            }
            break;
        case SpvOpTypeVector:
            if (wc>=4) {
                uint32_t el=code[pos+2], n=code[pos+3];
                if (el==tFloat) {
                    if (n==2) tVec2=code[pos+1]; if (n==3) tVec3=code[pos+1];
                    if (n==4) tVec4=code[pos+1];
                }
                if (el==tUint && n==4) tUvec4=code[pos+1];
            }
            break;
        case SpvOpConstantTrue:  cTrue  = code[pos+2]; break;
        case SpvOpConstantFalse: cFalse = code[pos+2]; break;
        case SpvOpTypeRayQueryKHR:    rqTypeIds.insert(code[pos+1]); break;
        case SpvOpTypeAccelStructKHR: asTypeIds.insert(code[pos+1]); break;
        case SpvOpTypePointer:
            if (wc>=4) {
                if (rqTypeIds.count(code[pos+3])) rqPtrTypeIds.insert(code[pos+1]);
                if (asTypeIds.count(code[pos+3])) asPtrTypeIds.insert(code[pos+1]);
            }
            break;
        case SpvOpVariable:
            if (wc>=4) {
                if (rqPtrTypeIds.count(code[pos+1])) rqVarIds.insert(code[pos+2]);
                if (asPtrTypeIds.count(code[pos+1])) asVarIds.insert(code[pos+2]);
            }
            break;
        }
        pos += wc;
    }

    if (rqTypeIds.empty()) return {};

    fprintf(stderr, "[SPIRV-RQ] ray query types=%zu vars=%zu, AS types=%zu vars=%zu\n",
            rqTypeIds.size(), rqVarIds.size(), asTypeIds.size(), asVarIds.size());

    // Allocate missing base types
    bool needFloat = !tFloat, needUint = !tUint, needInt = !tInt, needBool = !tBool;
    bool needVec2 = !tVec2, needVec3 = !tVec3, needVec4 = !tVec4, needUvec4 = !tUvec4;
    if (needFloat) tFloat = newId();
    if (needUint)  tUint  = newId();
    if (needInt)   tInt   = newId();
    if (needBool)  tBool  = newId();
    if (needVec2)  tVec2  = newId();
    if (needVec3)  tVec3  = newId();
    if (needVec4)  tVec4  = newId();
    if (needUvec4) tUvec4 = newId();
    bool needTrue = !cTrue, needFalse = !cFalse;
    if (needTrue)  cTrue  = newId();
    if (needFalse) cFalse = newId();

    // Struct for ray query state:
    //  0: vec3 origin      1: vec3 direction
    //  2: float tMin       3: float tMax
    //  4: float hitT       5: vec2 hitBary
    //  6: int hitPrimID    7: int hitInstID
    //  8: int hitType      9: int nodeIdx
    // 10: int done
    uint32_t tRQStruct  = newId();
    uint32_t tRQPtr     = newId();
    // Pointer types for AccessChain
    uint32_t pFloat = newId(), pVec3 = newId(), pInt = newId(), pVec2 = newId();
    uint32_t pUint = newId(), pVec4F = newId(); // function-scope uint ptr, vec4 ptr
    // BVH SSBO types
    uint32_t tNodesRA = newId(), tNodesS = newId(), tNodesP = newId(), vNodes = newId();
    uint32_t tTrisRA  = newId(), tTrisS  = newId(), tTrisP  = newId(), vTris  = newId();
    // TLAS SSBO types (binding 2 = TLAS nodes, binding 3 = instances)
    uint32_t tTlasRA = newId(), tTlasS = newId(), tTlasP = newId(), vTlas = newId();
    uint32_t tInstRA = newId(), tInstS = newId(), tInstP = newId(), vInst = newId();
    uint32_t pUvec4SB = newId(), pVec4SB = newId();
    // Constants: integers 0..10 and -1
    uint32_t c0 = newId(), c1 = newId(), cn1 = newId();
    uint32_t c2 = newId(), c3 = newId(), c4 = newId(), c5 = newId();
    uint32_t c6 = newId(), c7 = newId(), c8 = newId(), c9 = newId(), c10 = newId();
    uint32_t cf0 = newId(), cf_huge = newId();
    // Additional constants for traversal
    uint32_t cu0 = newId(), cu1 = newId(), cu2 = newId(), cu3 = newId(), cu7 = newId();
    uint32_t cf1 = newId(), cfn1 = newId(); // float 1.0, float -1.0
    float tiny = 1e-8f; uint32_t cf_tiny = newId();
    // Pre-allocated traversal local variables (per-function, emitted at first OpLabel)
    uint32_t tvNi = newId(), tvBestT = newId(), tvHitU = newId(), tvHitV = newId();
    uint32_t tvHitPrim = newId(), tvHitType = newId(), tvTriIdx = newId();
    uint32_t tvIter = newId(); // iteration counter for max-iteration guard
    uint32_t cMaxIter = newId(); // constant: max iterations (200)
    // TLAS traversal variables
    uint32_t tvTlasNi = newId();     // TLAS node index
    uint32_t tvTlasIter = newId();   // TLAS iteration counter
    uint32_t tvLocalRo = newId();    // local-space ray origin (vec3)
    uint32_t tvLocalRd = newId();    // local-space ray direction (vec3)
    uint32_t tvHitInst = newId();    // hit instance index
    uint32_t tvBlasNodeOff = newId(); // per-instance BLAS node offset
    uint32_t tvBlasTriOff  = newId(); // per-instance BLAS tri offset
    uint32_t cMaxTlasIter = newId(); // constant: max TLAS iterations
    uint32_t glslExtId = 0; // will find existing GLSL.std.450 import

    // Build output
    std::vector<uint32_t> out;
    out.reserve(numWords * 2);
    for (int i = 0; i < 5; i++) out.push_back(code[i]); // header

    // Mapping: original rq var -> new struct var
    // Pre-allocate one struct variable per ray query variable
    std::unordered_map<uint32_t, uint32_t> rqMap;
    for (uint32_t rqv : rqVarIds) {
        rqMap[rqv] = newId();
    }

    // Pre-scan: identify functions that contain RQ operations
    // Used to intercept calls to these functions and replace with constants
    std::unordered_set<uint32_t> rqFuncIds;
    // Also find shader's native types and constants
    uint32_t nativeFloatType = 0, nativeFloat1 = 0, nativeFloat0 = 0;
    uint32_t nativeBoolType = 0, nativeBoolFalse = 0;
    uint32_t nativeUintType = 0, nativeUint0 = 0;
    uint32_t nativeIntType = 0, nativeInt0 = 0, nativeInt1 = 0;
    uint32_t nativeVoidType = 0;
    uint32_t entryPointFunc = 0;
    std::unordered_map<uint32_t, uint32_t> rqFuncRetType; // funcId -> returnTypeId
    {
        uint32_t curFunc = 0;
        size_t sp = 5;
        // Pass 1: find basic types
        while (sp < numWords) {
            uint16_t swc = (uint16_t)(code[sp] >> 16);
            uint16_t sop = (uint16_t)(code[sp] & 0xFFFF);
            if (swc == 0 || sp + swc > numWords) break;
            if (sop == 22 /*OpTypeFloat*/ && swc >= 3 && code[sp+2] == 32)
                nativeFloatType = code[sp+1];
            if (sop == 20 /*OpTypeBool*/ && swc >= 2)
                nativeBoolType = code[sp+1];
            if (sop == 21 /*OpTypeInt*/ && swc >= 4 && code[sp+2] == 32 && code[sp+3] == 0)
                nativeUintType = code[sp+1];
            if (sop == 21 /*OpTypeInt*/ && swc >= 4 && code[sp+2] == 32 && code[sp+3] == 1)
                nativeIntType = code[sp+1];
            if (sop == 19 /*OpTypeVoid*/ && swc >= 2)
                nativeVoidType = code[sp+1];
            sp += swc;
        }
        // Pass 2: find constants and RQ functions
        sp = 5;
        while (sp < numWords) {
            uint16_t swc = (uint16_t)(code[sp] >> 16);
            uint16_t sop = (uint16_t)(code[sp] & 0xFFFF);
            if (swc == 0 || sp + swc > numWords) break;
            if (sop == 43 /*OpConstant*/ && swc >= 4) {
                if (code[sp+1] == nativeFloatType) {
                    float val; memcpy(&val, &code[sp+3], 4);
                    if (val == 1.0f) nativeFloat1 = code[sp+2];
                    if (val == 0.0f) nativeFloat0 = code[sp+2];
                }
                if (code[sp+1] == nativeUintType && code[sp+3] == 0)
                    nativeUint0 = code[sp+2];
                if (code[sp+1] == nativeIntType) {
                    int32_t ival; memcpy(&ival, &code[sp+3], 4);
                    if (ival == 0) nativeInt0 = code[sp+2];
                    if (ival == 1) nativeInt1 = code[sp+2];
                }
            }
            if (sop == 42 /*OpConstantFalse*/ && swc >= 3 && code[sp+1] == nativeBoolType)
                nativeBoolFalse = code[sp+2];
            if (sop == SpvOpFunction && swc >= 5) {
                curFunc = code[sp+2];
                rqFuncRetType[curFunc] = code[sp+1];
            }
            if (sop == SpvOpFunctionEnd) curFunc = 0;
            if (curFunc && (sop == SpvOpRQInitialize || sop == SpvOpRQProceed ||
                sop == SpvOpRQGetIntersectionType || sop == SpvOpRQGetT))
                rqFuncIds.insert(curFunc);
            // Capture entry point function ID
            if (sop == SpvOpEntryPoint && swc >= 3)
                entryPointFunc = code[sp+2];
            sp += swc;
        }
        fprintf(stderr, "[SPIRV-RQ] Functions with RQ ops: %zu, entryPoint=%%%u, nativeFloat=%u f1=%u f0=%u bool=%u bF=%u uint=%u u0=%u\n",
                rqFuncIds.size(), entryPointFunc, nativeFloatType, nativeFloat1, nativeFloat0,
                nativeBoolType, nativeBoolFalse, nativeUintType, nativeUint0);
        for (uint32_t fid : rqFuncIds) {
            fprintf(stderr, "[SPIRV-RQ]   RQ func %%%u retType=%%%u\n", fid, rqFuncRetType[fid]);
        }
    }
    bool entryHasRQ = rqFuncIds.count(entryPointFunc) > 0;
    fprintf(stderr, "[SPIRV-RQ] entryHasRQ=%d\n", (int)entryHasRQ);

#if SPIRV_RQ_PASSTHROUGH
    // PASSTHROUGH: gut the RQ function body to just `return 1.0`
    // and KEEP calls live (not stubbed) to test function body replacement
    {
        // Pre-scan for existing types we can reuse (avoids SPIR-V duplicate type errors)
        uint32_t existVec3 = 0, existPtrFloat = 0, existPtrUint = 0;
        uint32_t phB_intType = 0;
        uint32_t phB_int0 = 0, phB_int1 = 0, phB_int2 = 0, phB_int3 = 0, phB_int4 = 0;
        {
            size_t ss = 5;
            while (ss < numWords) {
                uint16_t sw = (uint16_t)(code[ss] >> 16), so = (uint16_t)(code[ss] & 0xFFFF);
                if (sw == 0 || ss + sw > numWords) break;
                if (so == 23 /*OpTypeVector*/ && sw == 4 && code[ss+2] == nativeFloatType && code[ss+3] == 3)
                    existVec3 = code[ss+1];
                if (so == 32 /*OpTypePointer*/ && sw == 4 && code[ss+2] == 7 /*Function*/) {
                    if (code[ss+3] == nativeFloatType && !existPtrFloat) existPtrFloat = code[ss+1];
                    if (code[ss+3] == nativeUintType && !existPtrUint)   existPtrUint = code[ss+1];
                }
                if (so == 21 /*OpTypeInt*/ && sw >= 4 && code[ss+2] == 32 && code[ss+3] == 1)
                    phB_intType = code[ss+1];
                ss += sw;
            }
            if (!phB_intType) phB_intType = nextId++;
        }
        // Search for int constants 0-4
        {
            size_t ss = 5;
            while (ss < numWords) {
                uint16_t sw = (uint16_t)(code[ss] >> 16), so = (uint16_t)(code[ss] & 0xFFFF);
                if (sw == 0 || ss + sw > numWords) break;
                if (so == 43 /*OpConstant*/ && sw >= 4 && code[ss+1] == phB_intType) {
                    int32_t v; memcpy(&v, &code[ss+3], 4);
                    if (v == 0) phB_int0 = code[ss+2];
                    if (v == 1) phB_int1 = code[ss+2];
                    if (v == 2) phB_int2 = code[ss+2];
                    if (v == 3) phB_int3 = code[ss+2];
                    if (v == 4) phB_int4 = code[ss+2];
                }
                ss += sw;
            }
        }
        
        // Allocate IDs only for types that don't already exist
        uint32_t phB_vec3 = existVec3 ? existVec3 : nextId++;
        uint32_t phB_struct = nextId++, phB_structPtr = nextId++;
        uint32_t phB_pFloat = existPtrFloat ? existPtrFloat : nextId++;
        uint32_t phB_pUint = existPtrUint ? existPtrUint : nextId++;
        
        // First pass: collect IDs defined inside RQ function body for OpName stripping
        std::unordered_set<uint32_t> rqBodyIds;
        {
            size_t sp = 5;
            bool scanInRQ = false, scanPastLabel = false;
            while (sp < numWords) {
                uint16_t swc = (uint16_t)(code[sp] >> 16);
                uint16_t sop = (uint16_t)(code[sp] & 0xFFFF);
                if (swc == 0 || sp + swc > numWords) break;
                if (sop == SpvOpFunction && swc >= 5) {
                    uint32_t fid = code[sp+2];
                    if (rqFuncIds.count(fid) && fid != entryPointFunc) {
                        scanInRQ = true; scanPastLabel = false;
                    }
                }
                if (sop == SpvOpFunctionEnd && scanInRQ) { scanInRQ = false; }
                if (scanInRQ && scanPastLabel && sop != SpvOpFunctionEnd) {
                    // Collect result ID (position 2 for most instructions with result)
                    if (swc >= 3) rqBodyIds.insert(code[sp+2]);
                    if (swc >= 2 && sop == 59 /*OpVariable*/) rqBodyIds.insert(code[sp+2]);
                }
                if (scanInRQ && sop == SpvOpLabel && !scanPastLabel) scanPastLabel = true;
                sp += swc;
            }
            fprintf(stderr, "[SPIRV-RQ] PASSTHROUGH: collected %zu body IDs to strip\n", rqBodyIds.size());
        }

        size_t pp = 5;
        int stubCount = 0;
        bool inRQFunc = false;
        bool pastFirstLabel = false;
        while (pp < numWords) {
            uint16_t pwc = (uint16_t)(code[pp] >> 16);
            uint16_t pop = (uint16_t)(code[pp] & 0xFFFF);
            if (pwc == 0 || pp + pwc > numWords) break;
            bool pskip = false;

            // Strip OpName/OpMemberName for gutted IDs
            if ((pop == 5 /*OpName*/ || pop == 6 /*OpMemberName*/) && pwc >= 2 && rqBodyIds.count(code[pp+1])) {
                pskip = true;
            }
            // Also strip OpDecorate for gutted IDs
            if (pop == 71 /*OpDecorate*/ && pwc >= 2 && rqBodyIds.count(code[pp+1])) {
                pskip = true;
            }

            // Track entering/exiting RQ functions (NOT the entry point)
            if (pop == SpvOpFunction && pwc >= 5) {
                uint32_t funcId = code[pp+2];
                if (rqFuncIds.count(funcId) && funcId != entryPointFunc) {
                    inRQFunc = true;
                    pastFirstLabel = false;
                }
            }
            if (pop == SpvOpFunctionEnd && inRQFunc) {
                inRQFunc = false;
                pastFirstLabel = false;
            }

            // Inside RQ function: keep OpFunction, params, first OpLabel
            // Then emit OpReturnValue 1.0 and skip everything until OpFunctionEnd
            if (inRQFunc) {
                if (pop == SpvOpLabel && !pastFirstLabel) {
                    pastFirstLabel = true;
                    for (uint16_t j = 0; j < pwc; j++)
                        out.push_back(code[pp + j]);
                    
                    // Emit a function body with a simple loop + return 1.0
                    if (nativeFloat1 && nativeFloatType && nativeUintType && nativeUint0) {
                        // Find native pointer types (Function storage)
                        uint32_t ptrUint = 0, ptrFloat = 0;
                        {
                            size_t ss = 5;
                            while (ss < numWords) {
                                uint16_t sw = (uint16_t)(code[ss] >> 16);
                                uint16_t so = (uint16_t)(code[ss] & 0xFFFF);
                                if (sw == 0 || ss + sw > numWords) break;
                                if (so == 32 /*OpTypePointer*/ && sw == 4 && code[ss+2] == 7 /*Function*/) {
                                    if (code[ss+3] == nativeUintType && !ptrUint) ptrUint = code[ss+1];
                                    if (code[ss+3] == nativeFloatType && !ptrFloat) ptrFloat = code[ss+1];
                                }
                                ss += sw;
                            }
                        }
                        
                        // Find or create uint constant 1
                        uint32_t nUint1 = 0;
                        {
                            size_t ss = 5;
                            while (ss < numWords) {
                                uint16_t sw = (uint16_t)(code[ss] >> 16);
                                uint16_t so = (uint16_t)(code[ss] & 0xFFFF);
                                if (sw == 0 || ss + sw > numWords) break;
                                if (so == 43 /*OpConstant*/ && sw >= 4 && code[ss+1] == nativeUintType && code[ss+3] == 1)
                                    nUint1 = code[ss+2];
                                ss += sw;
                            }
                        }
                        
                        if (ptrUint && ptrFloat && nUint1) {
                            // Declare struct variable using Phase B types
                            uint32_t varS = nextId++;  // struct variable
                            uint32_t varI = nextId++;
                            uint32_t varF = nextId++;
                            spvEmit(out, 59 /*OpVariable*/, {phB_structPtr, varS, 7});
                            spvEmit(out, 59 /*OpVariable*/, {ptrUint, varI, 7});
                            spvEmit(out, 59 /*OpVariable*/, {ptrFloat, varF, 7});
                            
                            // Access struct member 2 (float) via OpAccessChain
                            // Need int constant 2 for the index
                            if (phB_int2 && phB_pFloat) {
                                uint32_t ptrMember = nextId++;
                                spvEmit(out, 65 /*OpAccessChain*/, {phB_pFloat, ptrMember, varS, phB_int2});
                                spvEmit(out, 62 /*OpStore*/, {ptrMember, nativeFloat1});
                                uint32_t loadM = nextId++;
                                spvEmit(out, 61 /*OpLoad*/, {nativeFloatType, loadM, ptrMember});
                                spvEmit(out, 254 /*OpReturnValue*/, {loadM});
                                fprintf(stderr, "[SPIRV-RQ] PASSTHROUGH: gutted RQ func with STRUCT access body\n");
                            } else {
                                // Fallback: just use the counter loop
                                spvEmit(out, 62 /*OpStore*/, {varI, nativeUint0});
                                spvEmit(out, 62 /*OpStore*/, {varF, nativeFloat1});
                                
                                uint32_t lblMerge = nextId++, lblLoop = nextId++;
                                uint32_t lblBody = nextId++, lblCont = nextId++;
                                spvEmit(out, 249 /*OpBranch*/, {lblLoop});
                                
                                spvEmit(out, SpvOpLabel, {lblLoop});
                                spvEmit(out, 246 /*OpLoopMerge*/, {lblMerge, lblCont, 0});
                                spvEmit(out, 249 /*OpBranch*/, {lblBody});
                                
                                spvEmit(out, SpvOpLabel, {lblBody});
                                uint32_t loadI = nextId++;
                                spvEmit(out, 61 /*OpLoad*/, {nativeUintType, loadI, varI});
                                uint32_t cmp = nextId++;
                                spvEmit(out, 172 /*OpULessThan*/, {nativeBoolType, cmp, loadI, nUint1});
                                spvEmit(out, 250 /*OpBranchConditional*/, {cmp, lblCont, lblMerge});
                                
                                spvEmit(out, SpvOpLabel, {lblCont});
                                uint32_t addI = nextId++;
                                spvEmit(out, 128 /*OpIAdd*/, {nativeUintType, addI, loadI, nUint1});
                                spvEmit(out, 62 /*OpStore*/, {varI, addI});
                                spvEmit(out, 249 /*OpBranch*/, {lblLoop});
                                
                                spvEmit(out, SpvOpLabel, {lblMerge});
                                uint32_t loadF = nextId++;
                                spvEmit(out, 61 /*OpLoad*/, {nativeFloatType, loadF, varF});
                                spvEmit(out, 254 /*OpReturnValue*/, {loadF});
                                
                                fprintf(stderr, "[SPIRV-RQ] PASSTHROUGH: gutted RQ func with LOOP (no int2=%u pF=%u)\n", phB_int2, phB_pFloat);
                            }
                        } else {
                            spvEmit(out, 254 /*OpReturnValue*/, {nativeFloat1});
                            fprintf(stderr, "[SPIRV-RQ] PASSTHROUGH: gutted RQ func return 1.0 (ptrU=%u ptrF=%u u1=%u)\n",
                                    ptrUint, ptrFloat, nUint1);
                        }
                    } else {
                        spvEmit(out, 254 /*OpReturnValue*/, {nativeFloat1});
                        fprintf(stderr, "[SPIRV-RQ] PASSTHROUGH: gutted RQ func to return 1.0 (basic)\n");
                    }
                    pskip = true;
                } else if (pastFirstLabel && pop != SpvOpFunctionEnd) {
                    pskip = true;
                }
            }

            // DO NOT stub calls — keep them live
            // (the gutted function body returns 1.0 when called)

            if (!pskip) {
                for (uint16_t j = 0; j < pwc; j++)
                    out.push_back(code[pp + j]);
            }
            pp += pwc;
        }
        
        // Inject Phase B types before first OpFunction using pre-allocated IDs
        {
            std::vector<uint32_t> typeInject;
            auto E = [&](uint16_t o, std::vector<uint32_t> a){ spvEmit(typeInject, o, a); };
            
            if (nativeFloatType && nativeUintType) {
                // Only emit types that don't already exist (pre-scanned above)
                if (!existVec3)     E(SpvOpTypeVector,  {phB_vec3, nativeFloatType, 3});
                if (!existPtrFloat) E(SpvOpTypePointer, {phB_pFloat, 7, nativeFloatType});
                if (!existPtrUint)  E(SpvOpTypePointer, {phB_pUint, 7, nativeUintType});
                E(SpvOpTypeStruct,  {phB_struct, phB_vec3, phB_vec3, nativeFloatType, nativeFloatType, nativeFloatType, nativeUintType, nativeUintType});
                E(SpvOpTypePointer, {phB_structPtr, 7 /*Function*/, phB_struct});
                
                // Emit signed int type if shader doesn't have one
                {
                    bool intFound = false;
                    size_t ss = 5;
                    while (ss < numWords) {
                        uint16_t sw = (uint16_t)(code[ss] >> 16), so = (uint16_t)(code[ss] & 0xFFFF);
                        if (sw == 0 || ss + sw > numWords) break;
                        if (so == 21 && sw >= 4 && code[ss+2] == 32 && code[ss+3] == 1)
                            intFound = true;
                        ss += sw;
                    }
                    if (!intFound) E(SpvOpTypeInt, {phB_intType, 32, 1});
                }
                
                if (!phB_int0) { phB_int0 = nextId++; E(SpvOpConstant, {phB_intType, phB_int0, 0}); }
                if (!phB_int1) { phB_int1 = nextId++; E(SpvOpConstant, {phB_intType, phB_int1, 1}); }
                if (!phB_int2) { phB_int2 = nextId++; E(SpvOpConstant, {phB_intType, phB_int2, 2}); }
                if (!phB_int3) { phB_int3 = nextId++; E(SpvOpConstant, {phB_intType, phB_int3, 3}); }
                if (!phB_int4) { phB_int4 = nextId++; E(SpvOpConstant, {phB_intType, phB_int4, 4}); }
            }
            
            size_t funcPos = 0;
            for (size_t i = 5; i < out.size(); ) {
                uint16_t w = (uint16_t)(out[i] >> 16);
                uint16_t o2 = (uint16_t)(out[i] & 0xFFFF);
                if (w == 0 || i + w > out.size()) break;
                if (o2 == SpvOpFunction) { funcPos = i; break; }
                i += w;
            }
            if (funcPos > 0 && !typeInject.empty()) {
                out.insert(out.begin() + funcPos, typeInject.begin(), typeInject.end());
                fprintf(stderr, "[SPIRV-RQ] PASSTHROUGH: injected %zu type words before OpFunction\n", typeInject.size());
            }
        }

        out[3] = nextId; // set bound AFTER all ID allocations

        fprintf(stderr, "[SPIRV-RQ] PASSTHROUGH: %zu -> %zu words, calls LIVE (not stubbed)\n",
                numWords, out.size());
#if SPIRV_RQ_DUMP
        {
            static int dumpIdx = 0;
            char path[128];
            snprintf(path, sizeof(path), "/tmp/spirv_rq_dump_%d.spv", dumpIdx++);
            FILE* f = fopen(path, "wb");
            if (f) { fwrite(out.data(), 4, out.size(), f); fclose(f); }
            fprintf(stderr, "[SPIRV-RQ] Dumped %zu words to %s\n", out.size(), path);
        }
#endif
        return out;
    }
#endif // SPIRV_RQ_PASSTHROUGH

    // Track when we're inside a function's first block to emit variables
    bool inFunction = false;
    bool emittedFuncVars = false;
    uint32_t currentFuncId = 0;  // tracks which function we're currently inside
    bool inEntryRQ = false;      // true when inside entry point that has RQ ops

    // Two-phase injection:
    // Phase A: Decorations (must go in annotation section, before types)
    // Phase B: Types + variables (must go before functions)
    bool injectedDecorations = false;
    bool injectedTypes = false;
    pos = 5;
    while (pos < numWords) {
        uint16_t wc = (uint16_t)(code[pos] >> 16);
        uint16_t op = (uint16_t)(code[pos] & 0xFFFF);
        if (wc == 0 || pos + wc > numWords) break;

        bool skip = false;

#if !SPIRV_RQ_KEEP_TYPES
        // Remove ray query / accel struct capabilities
        if (op == SpvOpCapability && wc >= 2) {
            uint32_t cap = code[pos+1];
            if (cap == CapRayQueryKHR || cap == CapRayTraversalPrimitiveCulling ||
                cap == CapRayTracingKHR)
                skip = true;
        }

        // Remove OpExtension "SPV_KHR_ray_query" (opcode 10)
        if (op == 10 /*OpExtension*/ && wc >= 2) {
            const char* extName = (const char*)&code[pos+1];
            if (strstr(extName, "ray_query") || strstr(extName, "ray_tracing"))
                skip = true;
        }

        // Remove OpSourceExtension with ray query
        if (op == 4 /*OpSourceExtension*/ && wc >= 2) {
            const char* extName = (const char*)&code[pos+1];
            if (strstr(extName, "ray_query") || strstr(extName, "ray_tracing"))
                skip = true;
        }
#endif // !SPIRV_RQ_KEEP_TYPES

        // Capture GLSL.std.450 extended instruction set ID
        if (op == SpvOpExtInstImport && wc >= 3) {
            const char* name = (const char*)&code[pos+2];
            if (strstr(name, "GLSL.std.450"))
                glslExtId = code[pos+1];
        }

#if !SPIRV_RQ_KEEP_TYPES
        // Remove OpName for deleted variables/types
        if (op == 5 /*OpName*/ && wc >= 3) {
            uint32_t target = code[pos+1];
            if (rqVarIds.count(target) || asVarIds.count(target) ||
                rqTypeIds.count(target) || asTypeIds.count(target) ||
                rqPtrTypeIds.count(target) || asPtrTypeIds.count(target))
                skip = true;
        }
#endif // !SPIRV_RQ_KEEP_TYPES

        // Rewrite OpEntryPoint: remove deleted interface variable IDs
        if (op == SpvOpEntryPoint && wc >= 4) {
            // Copy: execution model, entry point ID, name string
            // Then filter interface variables
            std::vector<uint32_t> newEP;
            // First word (wc|op) will be rebuilt
            newEP.push_back(code[pos+1]); // execution model
            newEP.push_back(code[pos+2]); // entry point function ID
            // Name is a literal string (multiple words, null terminated)
            size_t nameStart = 3;
            size_t nameEnd = nameStart;
            while (nameEnd < wc) {
                uint32_t w = code[pos + nameEnd];
                nameEnd++;
                if ((w & 0xFF000000) == 0 || (w & 0x00FF0000) == 0 ||
                    (w & 0x0000FF00) == 0 || (w & 0x000000FF) == 0)
                    break; // found null terminator
            }
            for (size_t j = nameStart; j < nameEnd; j++)
                newEP.push_back(code[pos + j]);
            // Interface variables: filter out deleted ones, add BVH vars
            for (size_t j = nameEnd; j < wc; j++) {
                uint32_t id = code[pos + j];
#if !SPIRV_RQ_KEEP_TYPES
                if (!rqVarIds.count(id) && !asVarIds.count(id))
#endif
                    newEP.push_back(id);
            }
            // Add our BVH SSBO variables to the interface
#if !SPIRV_RQ_NO_SSBO
            newEP.push_back(vNodes);
            newEP.push_back(vTris);
            newEP.push_back(vTlas);
            newEP.push_back(vInst);
#endif
            spvEmit(out, SpvOpEntryPoint, newEP);
            skip = true;
        }

#if !SPIRV_RQ_KEEP_TYPES
        // Remove ray query and accel struct types (always — we rewrite all RQ ops)
        {
        if (op == SpvOpTypeRayQueryKHR || op == SpvOpTypeAccelStructKHR) skip = true;
        // Remove pointer-to-rq and pointer-to-AS types
        if (op == SpvOpTypePointer && wc >= 4) {
            if (rqTypeIds.count(code[pos+3]) || asTypeIds.count(code[pos+3]))
                skip = true;
        }
        // Remove rq and AS variables
        if (op == SpvOpVariable && wc >= 4) {
            if (rqVarIds.count(code[pos+2]) || asVarIds.count(code[pos+2]))
                skip = true;
        }
        // Remove decorations on AS vars/types and rq types
        if (op == SpvOpDecorate && wc >= 3) {
            if (asVarIds.count(code[pos+1]) || asTypeIds.count(code[pos+1]) ||
                rqTypeIds.count(code[pos+1]) || rqVarIds.count(code[pos+1]))
                skip = true;
        }

        // Remove loads of rq/AS variables (we'll inline access)
        if (op == SpvOpLoad && wc >= 4) {
            if (rqVarIds.count(code[pos+3]) || asVarIds.count(code[pos+3]))
                skip = true;
        }
        } // type/var removal block
#endif // !SPIRV_RQ_KEEP_TYPES

        // --- Phase A: Inject decorations before type definitions start ---
        // Types start with OpTypeVoid/Bool/Int/Float/Vector/etc.
        // Decorations must come before them.
#if !SPIRV_RQ_MINIMAL
        if (!injectedDecorations &&
            (op == SpvOpTypeVoid || op == SpvOpTypeBool || op == SpvOpTypeInt ||
             op == SpvOpTypeFloat || op == SpvOpTypeVector || op == SpvOpTypeStruct ||
             op == SpvOpTypePointer || op == SpvOpTypeArray || op == SpvOpTypeRuntimeArray ||
             op == SpvOpTypeFunction || op == SpvOpTypeRayQueryKHR || op == SpvOpTypeAccelStructKHR)) {
            injectedDecorations = true;
            auto E = [&](uint16_t o, std::vector<uint32_t> a){ spvEmit(out, o, a); };

            // BVH SSBO decorations
#if !SPIRV_RQ_NO_SSBO
            E(SpvOpDecorate, {tNodesRA, (uint32_t)DecArrayStride, 16});
            E(SpvOpDecorate, {tTrisRA,  (uint32_t)DecArrayStride, 16});
            E(SpvOpDecorate, {tNodesS,  (uint32_t)DecBlock});
            E(SpvOpDecorate, {tTrisS,   (uint32_t)DecBlock});
            E(SpvOpMemberDecorate, {tNodesS, 0, (uint32_t)DecOffset, 0});
            E(SpvOpMemberDecorate, {tTrisS,  0, (uint32_t)DecOffset, 0});
            E(SpvOpMemberDecorate, {tNodesS, 0, (uint32_t)DecNonWritable});
            E(SpvOpMemberDecorate, {tTrisS,  0, (uint32_t)DecNonWritable});
            E(SpvOpDecorate, {vNodes, (uint32_t)DecDescriptorSet, (uint32_t)bvhDescSet});
            E(SpvOpDecorate, {vNodes, (uint32_t)DecBinding, (uint32_t)bvhNodesBinding});
            E(SpvOpDecorate, {vTris,  (uint32_t)DecDescriptorSet, (uint32_t)bvhDescSet});
            E(SpvOpDecorate, {vTris,  (uint32_t)DecBinding, (uint32_t)bvhTrisBinding});
            // TLAS SSBO decorations (binding 2 = TLAS nodes, binding 3 = instances)
            E(SpvOpDecorate, {tTlasRA, (uint32_t)DecArrayStride, 16});
            E(SpvOpDecorate, {tInstRA, (uint32_t)DecArrayStride, 16});
            E(SpvOpDecorate, {tTlasS,  (uint32_t)DecBlock});
            E(SpvOpDecorate, {tInstS,  (uint32_t)DecBlock});
            E(SpvOpMemberDecorate, {tTlasS, 0, (uint32_t)DecOffset, 0});
            E(SpvOpMemberDecorate, {tInstS, 0, (uint32_t)DecOffset, 0});
            E(SpvOpMemberDecorate, {tTlasS, 0, (uint32_t)DecNonWritable});
            E(SpvOpMemberDecorate, {tInstS, 0, (uint32_t)DecNonWritable});
            E(SpvOpDecorate, {vTlas, (uint32_t)DecDescriptorSet, (uint32_t)bvhDescSet});
            E(SpvOpDecorate, {vTlas, (uint32_t)DecBinding, 2u});
            E(SpvOpDecorate, {vInst, (uint32_t)DecDescriptorSet, (uint32_t)bvhDescSet});
            E(SpvOpDecorate, {vInst, (uint32_t)DecBinding, 3u});
#endif // !SPIRV_RQ_NO_SSBO
        }
#endif // !SPIRV_RQ_MINIMAL (Phase A)

#if !SPIRV_RQ_MINIMAL
        // --- Phase B: Inject types + variables before first OpFunction ---
        if (op == SpvOpFunction && !injectedTypes) {
            injectedTypes = true;
            auto E = [&](uint16_t o, std::vector<uint32_t> a){ spvEmit(out, o, a); };

            if (needFloat) E(SpvOpTypeFloat,   {tFloat, 32});
            if (needUint)  E(SpvOpTypeInt,     {tUint, 32, 0});
            if (needInt)   E(SpvOpTypeInt,     {tInt, 32, 1});
            if (needBool)  E(SpvOpTypeBool,    {tBool});
            if (needVec2)  E(SpvOpTypeVector,  {tVec2, tFloat, 2});
            if (needVec3)  E(SpvOpTypeVector,  {tVec3, tFloat, 3});
            if (needVec4)  E(SpvOpTypeVector,  {tVec4, tFloat, 4});
            if (needUvec4) E(SpvOpTypeVector,  {tUvec4, tUint, 4});

            // Constants
            uint32_t zb = 0; float hugef = 1e30f; uint32_t hb; memcpy(&hb, &hugef, 4);
            float f1 = 1.0f; uint32_t f1b; memcpy(&f1b, &f1, 4);
            float fn1 = -1.0f; uint32_t fn1b; memcpy(&fn1b, &fn1, 4);
            uint32_t tinyb; memcpy(&tinyb, &tiny, 4);
            E(SpvOpConstant, {tInt, c0, 0});
            E(SpvOpConstant, {tInt, c1, 1});
            E(SpvOpConstant, {tInt, cn1, (uint32_t)-1});
            E(SpvOpConstant, {tInt, c2, 2});
            E(SpvOpConstant, {tInt, c3, 3});
            E(SpvOpConstant, {tInt, c4, 4});
            E(SpvOpConstant, {tInt, c5, 5});
            E(SpvOpConstant, {tInt, c6, 6});
            E(SpvOpConstant, {tInt, c7, 7});
            E(SpvOpConstant, {tInt, c8, 8});
            E(SpvOpConstant, {tInt, c9, 9});
            E(SpvOpConstant, {tInt, c10, 10});
            E(SpvOpConstant, {tInt, cMaxIter, 30}); // max BLAS traversal iterations (30 = safe for TDR)
            E(SpvOpConstant, {tInt, cMaxTlasIter, 512}); // max TLAS traversal iterations (137K+ nodes)
            E(SpvOpConstant, {tFloat, cf0, zb});
            E(SpvOpConstant, {tFloat, cf_huge, hb});
            E(SpvOpConstant, {tFloat, cf1, f1b});
            E(SpvOpConstant, {tFloat, cfn1, fn1b});
            E(SpvOpConstant, {tFloat, cf_tiny, tinyb});
            E(SpvOpConstant, {tUint, cu0, 0});
            E(SpvOpConstant, {tUint, cu1, 1});
            E(SpvOpConstant, {tUint, cu2, 2});
            E(SpvOpConstant, {tUint, cu3, 3});
            E(SpvOpConstant, {tUint, cu7, 7});
            if (needTrue)  E(SpvOpConstantTrue,  {tBool, cTrue});
            if (needFalse) E(SpvOpConstantFalse, {tBool, cFalse});

            // RQ state struct (member 8 = hitType is uint to match OpRayQueryGetIntersectionTypeKHR)
            E(SpvOpTypeStruct, {tRQStruct,
                tVec3, tVec3, tFloat, tFloat,
                tFloat, tVec2, tInt, tInt, tUint, tInt, tInt});
            E(SpvOpTypePointer, {tRQPtr, (uint32_t)SCFunction, tRQStruct});

            // Function-scope pointer types for AccessChain
            E(SpvOpTypePointer, {pFloat, (uint32_t)SCFunction, tFloat});
            E(SpvOpTypePointer, {pVec3,  (uint32_t)SCFunction, tVec3});
            E(SpvOpTypePointer, {pInt,   (uint32_t)SCFunction, tInt});
            E(SpvOpTypePointer, {pVec2,  (uint32_t)SCFunction, tVec2});
            E(SpvOpTypePointer, {pUint,  (uint32_t)SCFunction, tUint});
            E(SpvOpTypePointer, {pVec4F, (uint32_t)SCFunction, tVec4});

            // BVH nodes SSBO: buffer { uvec4 data[]; }
#if !SPIRV_RQ_NO_SSBO
            E(SpvOpTypeRuntimeArray, {tNodesRA, tUvec4});
            E(SpvOpTypeStruct,       {tNodesS, tNodesRA});
            E(SpvOpTypePointer,      {tNodesP, (uint32_t)SCStorageBuffer, tNodesS});
            E(SpvOpVariable,         {tNodesP, vNodes, (uint32_t)SCStorageBuffer});
            // Triangles SSBO: buffer { vec4 data[]; }
            E(SpvOpTypeRuntimeArray, {tTrisRA, tVec4});
            E(SpvOpTypeStruct,       {tTrisS, tTrisRA});
            E(SpvOpTypePointer,      {tTrisP, (uint32_t)SCStorageBuffer, tTrisS});
            E(SpvOpVariable,         {tTrisP, vTris, (uint32_t)SCStorageBuffer});
            // SSBO element pointer types
            E(SpvOpTypePointer, {pUvec4SB, (uint32_t)SCStorageBuffer, tUvec4});
            E(SpvOpTypePointer, {pVec4SB,  (uint32_t)SCStorageBuffer, tVec4});
            // TLAS nodes SSBO: buffer { uvec4 data[]; } (same format as BLAS nodes)
            E(SpvOpTypeRuntimeArray, {tTlasRA, tUvec4});
            E(SpvOpTypeStruct,       {tTlasS, tTlasRA});
            E(SpvOpTypePointer,      {tTlasP, (uint32_t)SCStorageBuffer, tTlasS});
            E(SpvOpVariable,         {tTlasP, vTlas, (uint32_t)SCStorageBuffer});
            // Instances SSBO: buffer { vec4 data[]; }
            E(SpvOpTypeRuntimeArray, {tInstRA, tVec4});
            E(SpvOpTypeStruct,       {tInstS, tInstRA});
            E(SpvOpTypePointer,      {tInstP, (uint32_t)SCStorageBuffer, tInstS});
            E(SpvOpVariable,         {tInstP, vInst, (uint32_t)SCStorageBuffer});

            fprintf(stderr, "[SPIRV-RQ] Injected BVH SSBOs set=%d bind=0,1,2,3\n", bvhDescSet);
#else
            fprintf(stderr, "[SPIRV-RQ] SSBO injection SUPPRESSED (SPIRV_RQ_NO_SSBO=1)\n");
#endif
        }
#endif // !SPIRV_RQ_MINIMAL (Phase B)

#if !SPIRV_RQ_MINIMAL
        // Track function entry: emit rq struct variables after first OpLabel
        if (op == SpvOpFunction && wc >= 3) {
            inFunction = true;
            emittedFuncVars = false;
            currentFuncId = code[pos+2];
            inEntryRQ = (currentFuncId == entryPointFunc && entryHasRQ);
        }
        if (op == SpvOpFunctionEnd) {
            inFunction = false;
            inEntryRQ = false;
        }
        // After each OpLabel in a function's first block, emit our variables
        // Only emit variables that haven't been emitted yet (they should only
        // exist in the function where the original ray query var was defined)
        if (op == SpvOpLabel && inFunction && !emittedFuncVars) {
            emittedFuncVars = true;
            // Check if this function uses any rq variables by scanning ahead
            bool hasRQOps = false;
            uint32_t scanPos = pos + wc;
            while (scanPos < numWords) {
                uint16_t scanWc = (uint16_t)(code[scanPos] >> 16);
                uint16_t scanOp = (uint16_t)(code[scanPos] & 0xFFFF);
                if (scanWc == 0 || scanPos + scanWc > numWords) break;
                if (scanOp == SpvOpFunctionEnd) break;
                if (scanOp == SpvOpRQInitialize || scanOp == SpvOpRQProceed ||
                    scanOp == SpvOpRQGetT || scanOp == SpvOpRQGetBarycentrics ||
                    scanOp == SpvOpRQGetPrimIdx || scanOp == SpvOpRQGetInstId ||
                    scanOp == SpvOpRQGetIntersectionType || scanOp == SpvOpRQConfirmIntersection ||
                    scanOp == SpvOpRQTerminate) {
                    hasRQOps = true;
                    break;
                }
                scanPos += scanWc;
            }
            // Copy the OpLabel first
            for (uint16_t j = 0; j < wc; j++)
                out.push_back(code[pos + j]);
            // Emit rq struct variables in ALL functions that use ray queries
            if (hasRQOps) {
                for (auto& kv : rqMap) {
                    spvEmit(out, SpvOpVariable, {tRQPtr, kv.second, (uint32_t)SCFunction});
                }
                // Traversal local variables (Function storage class, must be in first block)
                spvEmit(out, SpvOpVariable, {pInt,   tvNi,      (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pFloat, tvBestT,   (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pFloat, tvHitU,    (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pFloat, tvHitV,    (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pInt,   tvHitPrim, (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pUint,  tvHitType, (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pInt,   tvTriIdx,  (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pInt,   tvIter,    (uint32_t)SCFunction});
                // TLAS traversal variables
                spvEmit(out, SpvOpVariable, {pInt,   tvTlasNi,    (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pInt,   tvTlasIter,  (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pVec3,  tvLocalRo,   (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pVec3,  tvLocalRd,   (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pInt,   tvHitInst,   (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pInt,   tvBlasNodeOff, (uint32_t)SCFunction});
                spvEmit(out, SpvOpVariable, {pInt,   tvBlasTriOff,  (uint32_t)SCFunction});
            }
            skip = true; // already copied OpLabel
        }
#endif // !SPIRV_RQ_MINIMAL (func vars & RQ ops section start)

#if SPIRV_RQ_MINIMAL
        // MINIMAL mode: keep RQ ops unchanged. CRITICAL: Do NOT replace RQ ops with
        // OpUndef or constants in dead functions — NVIDIA's compiler corrupts the entire
        // shader even when the function containing modified ops is never called.
        // Function calls are stubbed by STUB_CALLS instead.
        { /* intentionally empty — all RQ ops pass through unchanged */ }
#else // !SPIRV_RQ_MINIMAL — full rewrite mode

        // Rewrite ALL RQ operations, including those in the entry point.
        // V100 has NO native RT — all RQ ops must be replaced with BVH2 traversal.
        {

        // --- Replace OpRayQueryInitializeKHR ---
        // Run FULL BVH2 stackless traversal eagerly, store results in struct
        if (op == SpvOpRQInitialize && wc >= 9) {
            uint32_t rqVar   = code[pos+1];
            // pos+2 = accel struct (ignored)
            // pos+3 = flags, pos+4 = mask
            uint32_t origin  = code[pos+5];
            uint32_t tMinId  = code[pos+6];
            uint32_t dirId   = code[pos+7];
            uint32_t tMaxId  = code[pos+8];

            uint32_t sv = rqMap.count(rqVar) ? rqMap[rqVar] : rqVar;
            auto E = [&](uint16_t o, std::vector<uint32_t> a){ spvEmit(out, o, a); };

            // Store ray params into struct
            // member 0: origin
            uint32_t a0 = newId();
            E(SpvOpAccessChain, {pVec3, a0, sv, c0});
            E(SpvOpStore, {a0, origin});
            // member 1: direction
            uint32_t a1 = newId();
            E(SpvOpAccessChain, {pVec3, a1, sv, c1});
            E(SpvOpStore, {a1, dirId});

            // member 2: tMin
            uint32_t a2 = newId();
            E(SpvOpAccessChain, {pFloat, a2, sv, c2});
            E(SpvOpStore, {a2, tMinId});
            // member 3: tMax
            uint32_t a3 = newId();
            E(SpvOpAccessChain, {pFloat, a3, sv, c3});
            E(SpvOpStore, {a3, tMaxId});
            // member 4: hitT = 1e30
            uint32_t a4 = newId();
            E(SpvOpAccessChain, {pFloat, a4, sv, c4});
            E(SpvOpStore, {a4, cf_huge});
            // member 6: hitPrimID = -1
            uint32_t a6 = newId();
            E(SpvOpAccessChain, {pInt, a6, sv, c6});
            E(SpvOpStore, {a6, cn1});
            // member 7: hitInstID = -1
            uint32_t a7 = newId();
            E(SpvOpAccessChain, {pInt, a7, sv, c7});
            E(SpvOpStore, {a7, cn1});
            // member 8: hitType = 0u (none) — uint to match OpRayQueryGetIntersectionTypeKHR
            uint32_t a8 = newId();
            E(SpvOpAccessChain, {pUint, a8, sv, c8});
            E(SpvOpStore, {a8, cu0});
            // member 9: nodeIdx = 0
            uint32_t a9 = newId();
            E(SpvOpAccessChain, {pInt, a9, sv, c9});
            E(SpvOpStore, {a9, c0});
            // member 10: done = 0
            uint32_t a10 = newId();
            E(SpvOpAccessChain, {pInt, a10, sv, c10});
            E(SpvOpStore, {a10, c0});

            // === V2: INLINE BVH2 STACKLESS TRAVERSAL ===
            // Emits structured SPIR-V loop that traverses BVH2, tests triangles,
            // stores tHit/bary/primIndex/hitType into the ray query struct.
            // After this, Proceed returns false and Get* reads from struct.

#if SPIRV_RQ_FULL_TRAVERSAL
            // Initialize traversal variables
            E(SpvOpStore, {tvNi, c0});
            E(SpvOpStore, {tvBestT, tMaxId});
            E(SpvOpStore, {tvHitU, cf0});
            E(SpvOpStore, {tvHitV, cf0});
            E(SpvOpStore, {tvHitPrim, cn1});
            E(SpvOpStore, {tvHitType, cu0});
            E(SpvOpStore, {tvIter, c0});
            // TLAS traversal init
            E(SpvOpStore, {tvTlasNi, c0});
            E(SpvOpStore, {tvTlasIter, c0});
            E(SpvOpStore, {tvHitInst, cn1});

#if SPIRV_RQ_ALWAYS_HIT
            // ALWAYS-HIT: set results and prevent TLAS loop from executing
            E(SpvOpStore, {tvHitType, cu1});     // committed triangle hit
            E(SpvOpStore, {tvBestT, cf1});       // hitT = 1.0 (dummy distance)
            E(SpvOpStore, {tvHitInst, c0});      // instance 0
            E(SpvOpStore, {tvHitPrim, c0});      // primitive 0
            E(SpvOpStore, {tvTlasNi, cn1});      // prevent loop from executing (ni=-1 < 0)
            fprintf(stderr, "[SPIRV-RQ] ALWAYS-HIT mode: bypassing traversal\n");
#endif

#if SPIRV_RQ_SSBO_TEST
            // SSBO READ TEST: Load uint word 0 from TLAS SSBO, check if non-zero (integer)
            {
                uint32_t ssboPtr = newId(), ssboVal = newId();
                E(SpvOpAccessChain, {pUvec4SB, ssboPtr, vTlas, c0, c0});  // tlas[0]
                E(SpvOpLoad, {tUvec4, ssboVal, ssboPtr});
                uint32_t ssboW0 = newId();
                E(SpvOpCompositeExtract, {tUint, ssboW0, ssboVal, 0});
                // Integer comparison: if raw uint bits != 0 → data present
                uint32_t cu0 = newId(), ssboNz = newId();
                E(SpvOpBitcast, {tUint, cu0, c0}); // int 0 → uint 0
                E(SpvOpINotEqual, {tBool, ssboNz, ssboW0, cu0});
                uint32_t ssboHitType = newId();
                E(SpvOpSelect, {tUint, ssboHitType, ssboNz, cu1, cu0});
                E(SpvOpStore, {tvHitType, ssboHitType});
                E(SpvOpStore, {tvBestT, cf1}); // dummy hitT
                E(SpvOpStore, {tvTlasNi, cn1}); // skip traversal loop
                fprintf(stderr, "[SPIRV-RQ] SSBO_TEST: uint read from TLAS[0].x\n");
            }
#endif

            // World-space ray origin and direction
            uint32_t ro = newId(), rd = newId();
            E(SpvOpCopyObject, {tVec3, ro, origin});
            E(SpvOpCopyObject, {tVec3, rd, dirId});

            // Compute world-space invD = 1.0 / rd (with epsilon)
            uint32_t rdx = newId(), rdy = newId(), rdz = newId();
            E(SpvOpCompositeExtract, {tFloat, rdx, rd, 0});
            E(SpvOpCompositeExtract, {tFloat, rdy, rd, 1});
            E(SpvOpCompositeExtract, {tFloat, rdz, rd, 2});
            auto safeInv = [&](uint32_t comp) -> uint32_t {
                uint32_t absv = newId(), cmp = newId(), sign = newId();
                uint32_t safe = newId(), inv = newId();
                E(SpvOpExtInst, {tFloat, absv, glslExtId, GLSLstd450FAbs, comp});
                E(SpvOpFOrdLessThan, {tBool, cmp, absv, cf_tiny});
                E(SpvOpExtInst, {tFloat, sign, glslExtId, GLSLstd450FSign, comp});
                uint32_t epsSigned = newId();
                E(SpvOpFMul, {tFloat, epsSigned, sign, cf_tiny});
                E(SpvOpSelect, {tFloat, safe, cmp, epsSigned, comp});
                E(SpvOpFDiv, {tFloat, inv, cf1, safe});
                return inv;
            };
            uint32_t wInvDx = safeInv(rdx), wInvDy = safeInv(rdy), wInvDz = safeInv(rdz);
            uint32_t wInvD = newId();
            E(SpvOpCompositeConstruct, {tVec3, wInvD, wInvDx, wInvDy, wInvDz});

            // World-space ood = -ro * invD
            uint32_t negRo = newId(), wOod = newId();
            E(SpvOpFNegate, {tVec3, negRo, ro});
            E(SpvOpFMul, {tVec3, wOod, negRo, wInvD});
            uint32_t wOodx = newId(), wOody = newId(), wOodz = newId();
            E(SpvOpCompositeExtract, {tFloat, wOodx, wOod, 0});
            E(SpvOpCompositeExtract, {tFloat, wOody, wOod, 1});
            E(SpvOpCompositeExtract, {tFloat, wOodz, wOod, 2});

            // World-space ro components for triangle test (used in BLAS too)
            uint32_t rox = newId(), roy = newId(), roz = newId();
            E(SpvOpCompositeExtract, {tFloat, rox, ro, 0});
            E(SpvOpCompositeExtract, {tFloat, roy, ro, 1});
            E(SpvOpCompositeExtract, {tFloat, roz, ro, 2});

            // === TLAS TRAVERSAL LOOP (outer) ===
            uint32_t tlH = newId(), tlBody = newId(), tlMerge = newId(), tlCont = newId();
            uint32_t tlInner = newId();

            E(SpvOpBranch, {tlH});
            spvEmit(out, SpvOpLabel, {tlH});
            spvEmit(out, SpvOpLoopMerge, {tlMerge, tlCont, 0});
            E(SpvOpBranch, {tlBody});

            spvEmit(out, SpvOpLabel, {tlBody});
            uint32_t tlNiVal = newId(), tlNiCond = newId();
            E(SpvOpLoad, {tInt, tlNiVal, tvTlasNi});
            E(SpvOpSGreaterThanEqual, {tBool, tlNiCond, tlNiVal, c0});
            uint32_t tlIterVal = newId(), tlIterCond = newId(), tlLoopCond = newId();
            E(SpvOpLoad, {tInt, tlIterVal, tvTlasIter});
            E(SpvOpSLessThan, {tBool, tlIterCond, tlIterVal, cMaxTlasIter});
            spvEmit(out, (uint16_t)167/*LogicalAnd*/, {tBool, tlLoopCond, tlNiCond, tlIterCond});
            E(SpvOpBranchConditional, {tlLoopCond, tlInner, tlMerge});

            spvEmit(out, SpvOpLabel, {tlInner});

#if SPIRV_RQ_FORCE_TLAS_HIT
            // DEBUG: Force hit on EVERY iteration
            E(SpvOpStore, {tvHitType, cu1});    // committed hit
            E(SpvOpStore, {tvTlasNi, cn1});    // exit loop after this iteration
#endif

            // Load TLAS node: tw0 = tlasNodes[tlasNi*2], tw1 = tlasNodes[tlasNi*2+1]
            uint32_t tlNi2 = newId(), tlNi2p1 = newId();
            E(SpvOpIMul, {tInt, tlNi2, tlNiVal, c2});
            E(SpvOpIAdd, {tInt, tlNi2p1, tlNi2, c1});
            uint32_t ptw0 = newId(), ptw1 = newId(), tw0 = newId(), tw1 = newId();
            E(SpvOpAccessChain, {pUvec4SB, ptw0, vTlas, c0, tlNi2});
            E(SpvOpLoad, {tUvec4, tw0, ptw0});
            E(SpvOpAccessChain, {pUvec4SB, ptw1, vTlas, c0, tlNi2p1});
            E(SpvOpLoad, {tUvec4, tw1, ptw1});

            // Extract leaf_enc = int(tw1.z), skip = int(tw1.w)
            uint32_t tw1z = newId(), tw1w = newId(), tlLeafEnc = newId(), tlSkipVal = newId();
            E(SpvOpCompositeExtract, {tUint, tw1z, tw1, 2});
            E(SpvOpCompositeExtract, {tUint, tw1w, tw1, 3});
            E(SpvOpBitcast, {tInt, tlLeafEnc, tw1z});
            E(SpvOpBitcast, {tInt, tlSkipVal, tw1w});

            // if (leaf_enc != 0) → TLAS leaf (instance), else → TLAS internal (AABB)
            uint32_t tlIsLeaf = newId();
            E(SpvOpINotEqual, {tBool, tlIsLeaf, tlLeafEnc, c0});
            uint32_t tlLeafLbl = newId(), tlInternalLbl = newId(), tlEndIfLbl = newId();
            spvEmit(out, SpvOpSelectionMerge, {tlEndIfLbl, 0});
            E(SpvOpBranchConditional, {tlIsLeaf, tlLeafLbl, tlInternalLbl});

            // --- TLAS LEAF: Instance traversal ---
            spvEmit(out, SpvOpLabel, {tlLeafLbl});
            {
                // instIdx = -(leaf_enc + 2) >> 3
                uint32_t tlEncP2 = newId(), tlEnc = newId(), instIdx = newId();
                E(SpvOpIAdd, {tInt, tlEncP2, tlLeafEnc, c2});
                E(SpvOpISub, {tInt, tlEnc, c0, tlEncP2});
                E(SpvOpShiftRightArithmetic, {tInt, instIdx, tlEnc, c3});

                // Load inverse transform rows from instances[instIdx*8 + 3..5]
                // base = instIdx * 8
#if SPIRV_RQ_TLAS_ONLY
                // TLAS-ONLY SHADOW MODE: Skip BLAS traversal entirely.
                // Report a committed hit as soon as any TLAS leaf (instance AABB) is reached.
                // Correct for shadow/occlusion queries; approximate for exact intersection queries.
                {
                    E(SpvOpStore, {tvHitType, cu1});     // committed triangle hit
                    E(SpvOpStore, {tvHitInst, instIdx}); // which instance
                    E(SpvOpStore, {tvTlasNi, cn1});     // break TLAS loop (any hit is enough)
                }
#else
                uint32_t instBase = newId();
                E(SpvOpIMul, {tInt, instBase, instIdx, c8});
                // ir0 = instances[instBase + 3]
                uint32_t ir0Idx = newId(), pIr0 = newId(), ir0 = newId();
                E(SpvOpIAdd, {tInt, ir0Idx, instBase, c3});
                E(SpvOpAccessChain, {pVec4SB, pIr0, vInst, c0, ir0Idx});
                E(SpvOpLoad, {tVec4, ir0, pIr0});
                // ir1 = instances[instBase + 4]
                uint32_t ir1Idx = newId(), pIr1 = newId(), ir1 = newId();
                E(SpvOpIAdd, {tInt, ir1Idx, instBase, c4});
                E(SpvOpAccessChain, {pVec4SB, pIr1, vInst, c0, ir1Idx});
                E(SpvOpLoad, {tVec4, ir1, pIr1});
                // ir2 = instances[instBase + 5]
                uint32_t ir2Idx = newId(), pIr2 = newId(), ir2 = newId();
                E(SpvOpIAdd, {tInt, ir2Idx, instBase, c5});
                E(SpvOpAccessChain, {pVec4SB, pIr2, vInst, c0, ir2Idx});
                E(SpvOpLoad, {tVec4, ir2, pIr2});

                // Load per-instance BLAS offsets from instances[instBase+6].w and [instBase+7].w
                uint32_t ib6Idx = newId(), pIb6 = newId(), ib6 = newId();
                E(SpvOpIAdd, {tInt, ib6Idx, instBase, c6});
                E(SpvOpAccessChain, {pVec4SB, pIb6, vInst, c0, ib6Idx});
                E(SpvOpLoad, {tVec4, ib6, pIb6});
                uint32_t ib7Idx = newId(), pIb7 = newId(), ib7 = newId();
                E(SpvOpIAdd, {tInt, ib7Idx, instBase, c7});
                E(SpvOpAccessChain, {pVec4SB, pIb7, vInst, c0, ib7Idx});
                E(SpvOpLoad, {tVec4, ib7, pIb7});
                // blasNodeOff = int(ib6.w), blasTriOff = int(ib7.w)
                // Values stored as actual floats (not bit-punned) to avoid denormal flush
                uint32_t ib6wF = newId(), ib7wF = newId();
                uint32_t blasNOff = newId(), blasTOff = newId();
                E(SpvOpCompositeExtract, {tFloat, ib6wF, ib6, 3});
                E(SpvOpCompositeExtract, {tFloat, ib7wF, ib7, 3});
                E(SpvOpConvertFToS, {tInt, blasNOff, ib6wF});
                E(SpvOpConvertFToS, {tInt, blasTOff, ib7wF});
                E(SpvOpStore, {tvBlasNodeOff, blasNOff});
                E(SpvOpStore, {tvBlasTriOff, blasTOff});

                // Extract inverse transform components
                uint32_t ir0x = newId(), ir0y = newId(), ir0z = newId(), ir0w = newId();
                E(SpvOpCompositeExtract, {tFloat, ir0x, ir0, 0});
                E(SpvOpCompositeExtract, {tFloat, ir0y, ir0, 1});
                E(SpvOpCompositeExtract, {tFloat, ir0z, ir0, 2});
                E(SpvOpCompositeExtract, {tFloat, ir0w, ir0, 3});
                uint32_t ir1x = newId(), ir1y = newId(), ir1z = newId(), ir1w = newId();
                E(SpvOpCompositeExtract, {tFloat, ir1x, ir1, 0});
                E(SpvOpCompositeExtract, {tFloat, ir1y, ir1, 1});
                E(SpvOpCompositeExtract, {tFloat, ir1z, ir1, 2});
                E(SpvOpCompositeExtract, {tFloat, ir1w, ir1, 3});
                uint32_t ir2x = newId(), ir2y = newId(), ir2z = newId(), ir2w = newId();
                E(SpvOpCompositeExtract, {tFloat, ir2x, ir2, 0});
                E(SpvOpCompositeExtract, {tFloat, ir2y, ir2, 1});
                E(SpvOpCompositeExtract, {tFloat, ir2z, ir2, 2});
                E(SpvOpCompositeExtract, {tFloat, ir2w, ir2, 3});

                // Transform ray origin to local space:
                // localRo.x = dot(ir0.xyz, worldRo) + ir0.w
                // localRo.y = dot(ir1.xyz, worldRo) + ir1.w
                // localRo.z = dot(ir2.xyz, worldRo) + ir2.w
                uint32_t ir0xyz = newId(), ir1xyz = newId(), ir2xyz = newId();
                E(SpvOpCompositeConstruct, {tVec3, ir0xyz, ir0x, ir0y, ir0z});
                E(SpvOpCompositeConstruct, {tVec3, ir1xyz, ir1x, ir1y, ir1z});
                E(SpvOpCompositeConstruct, {tVec3, ir2xyz, ir2x, ir2y, ir2z});

                uint32_t d0ro = newId(), d1ro = newId(), d2ro = newId();
                spvEmit(out, (uint16_t)148/*OpDot*/, {tFloat, d0ro, ir0xyz, ro});
                spvEmit(out, (uint16_t)148, {tFloat, d1ro, ir1xyz, ro});
                spvEmit(out, (uint16_t)148, {tFloat, d2ro, ir2xyz, ro});
                uint32_t lox = newId(), loy = newId(), loz = newId();
                E(SpvOpFAdd, {tFloat, lox, d0ro, ir0w});
                E(SpvOpFAdd, {tFloat, loy, d1ro, ir1w});
                E(SpvOpFAdd, {tFloat, loz, d2ro, ir2w});
                uint32_t localRo = newId();
                E(SpvOpCompositeConstruct, {tVec3, localRo, lox, loy, loz});
                E(SpvOpStore, {tvLocalRo, localRo});

                // Transform ray direction to local space (no translation):
                // localRd.x = dot(ir0.xyz, worldRd)
                // localRd.y = dot(ir1.xyz, worldRd)
                // localRd.z = dot(ir2.xyz, worldRd)
                uint32_t d0rd = newId(), d1rd = newId(), d2rd = newId();
                spvEmit(out, (uint16_t)148, {tFloat, d0rd, ir0xyz, rd});
                spvEmit(out, (uint16_t)148, {tFloat, d1rd, ir1xyz, rd});
                spvEmit(out, (uint16_t)148, {tFloat, d2rd, ir2xyz, rd});
                uint32_t localRd = newId();
                E(SpvOpCompositeConstruct, {tVec3, localRd, d0rd, d1rd, d2rd});
                E(SpvOpStore, {tvLocalRd, localRd});

                // Compute local-space invD and ood
                uint32_t lInvDx = safeInv(d0rd), lInvDy = safeInv(d1rd), lInvDz = safeInv(d2rd);
                uint32_t negLRo = newId(), lOod = newId();
                E(SpvOpFNegate, {tVec3, negLRo, localRo});
                uint32_t lInvD = newId();
                E(SpvOpCompositeConstruct, {tVec3, lInvD, lInvDx, lInvDy, lInvDz});
                E(SpvOpFMul, {tVec3, lOod, negLRo, lInvD});
                uint32_t lOodx = newId(), lOody = newId(), lOodz = newId();
                E(SpvOpCompositeExtract, {tFloat, lOodx, lOod, 0});
                E(SpvOpCompositeExtract, {tFloat, lOody, lOod, 1});
                E(SpvOpCompositeExtract, {tFloat, lOodz, lOod, 2});

                // === BLAS TRAVERSAL LOOP (inner) ===
                E(SpvOpStore, {tvNi, c0});
                E(SpvOpStore, {tvIter, c0});

                uint32_t lHeader = newId(), lBody = newId(), lMerge = newId(), lContinue = newId();
                uint32_t lLeaf = newId(), lInternal = newId(), lEndIf = newId();

                E(SpvOpBranch, {lHeader});
                spvEmit(out, SpvOpLabel, {lHeader});
                spvEmit(out, SpvOpLoopMerge, {lMerge, lContinue, 0});
                E(SpvOpBranch, {lBody});

                spvEmit(out, SpvOpLabel, {lBody});
                uint32_t niVal = newId(), niCond = newId();
                E(SpvOpLoad, {tInt, niVal, tvNi});
                E(SpvOpSGreaterThanEqual, {tBool, niCond, niVal, c0});
                uint32_t iterVal = newId(), iterCond = newId(), loopCond = newId();
                E(SpvOpLoad, {tInt, iterVal, tvIter});
                E(SpvOpSLessThan, {tBool, iterCond, iterVal, cMaxIter});
                spvEmit(out, (uint16_t)167, {tBool, loopCond, niCond, iterCond});
                uint32_t lInner2 = newId();
                E(SpvOpBranchConditional, {loopCond, lInner2, lMerge});

                spvEmit(out, SpvOpLabel, {lInner2});

                // Load BVH node: w0 = bvhNodes[(ni+blasNodeOff)*2], w1 = bvhNodes[(ni+blasNodeOff)*2+1]
                uint32_t bnoVal = newId(), niOff = newId();
                E(SpvOpLoad, {tInt, bnoVal, tvBlasNodeOff});
                E(SpvOpIAdd, {tInt, niOff, niVal, bnoVal});
                uint32_t ni2 = newId(), ni2p1 = newId();
                E(SpvOpIMul, {tInt, ni2, niOff, c2});
                E(SpvOpIAdd, {tInt, ni2p1, ni2, c1});
                uint32_t pw0 = newId(), pw1 = newId(), w0 = newId(), w1 = newId();
                E(SpvOpAccessChain, {pUvec4SB, pw0, vNodes, c0, ni2});
                E(SpvOpLoad, {tUvec4, w0, pw0});
                E(SpvOpAccessChain, {pUvec4SB, pw1, vNodes, c0, ni2p1});
                E(SpvOpLoad, {tUvec4, w1, pw1});

                // Extract leaf_enc = int(w1.z), skip = int(w1.w)
                uint32_t w1z = newId(), w1w = newId(), leafEnc = newId(), skipVal = newId();
                E(SpvOpCompositeExtract, {tUint, w1z, w1, 2});
                E(SpvOpCompositeExtract, {tUint, w1w, w1, 3});
                E(SpvOpBitcast, {tInt, leafEnc, w1z});
                E(SpvOpBitcast, {tInt, skipVal, w1w});

                // if (leaf_enc != 0) → leaf, else → internal
                uint32_t isLeafCond = newId();
                E(SpvOpINotEqual, {tBool, isLeafCond, leafEnc, c0});
                spvEmit(out, SpvOpSelectionMerge, {lEndIf, 0});
                E(SpvOpBranchConditional, {isLeafCond, lLeaf, lInternal});

                // --- BLAS LEAF: Triangle intersection ---
                spvEmit(out, SpvOpLabel, {lLeaf});
                {
                    uint32_t encP2 = newId(), enc = newId();
                    E(SpvOpIAdd, {tInt, encP2, leafEnc, c2});
                    E(SpvOpISub, {tInt, enc, c0, encP2});
                    uint32_t ts = newId(), tcm = newId(), tc = newId();
                    E(SpvOpShiftRightArithmetic, {tInt, ts, enc, c3});
                    E(SpvOpBitwiseAnd, {tInt, tcm, enc, c7});
                    E(SpvOpIAdd, {tInt, tc, tcm, c1});

                    E(SpvOpStore, {tvTriIdx, c0});
                    uint32_t triH = newId(), triB = newId(), triM = newId(), triC = newId();
                    E(SpvOpBranch, {triH});
                    spvEmit(out, SpvOpLabel, {triH});
                    spvEmit(out, SpvOpLoopMerge, {triM, triC, 0});
                    E(SpvOpBranch, {triB});

                    spvEmit(out, SpvOpLabel, {triB});
                    uint32_t tVal = newId(), tCond = newId();
                    E(SpvOpLoad, {tInt, tVal, tvTriIdx});
                    E(SpvOpSLessThan, {tBool, tCond, tVal, tc});
                    uint32_t triBody = newId();
                    E(SpvOpBranchConditional, {tCond, triBody, triM});

                    spvEmit(out, SpvOpLabel, {triBody});
                    uint32_t tst0 = newId(), btoVal = newId(), tst = newId(), b = newId();
                    E(SpvOpIAdd, {tInt, tst0, ts, tVal});
                    E(SpvOpLoad, {tInt, btoVal, tvBlasTriOff});
                    E(SpvOpIAdd, {tInt, tst, tst0, btoVal});  // absolute tri index
                    E(SpvOpIMul, {tInt, b, tst, c3});
                    uint32_t bp1 = newId(), bp2 = newId();
                    E(SpvOpIAdd, {tInt, bp1, b, c1});
                    E(SpvOpIAdd, {tInt, bp2, b, c2});
                    uint32_t pp0 = newId(), pp1 = newId(), pp2 = newId();
                    E(SpvOpAccessChain, {pVec4SB, pp0, vTris, c0, b});
                    E(SpvOpAccessChain, {pVec4SB, pp1, vTris, c0, bp1});
                    E(SpvOpAccessChain, {pVec4SB, pp2, vTris, c0, bp2});
                    uint32_t p0 = newId(), p1 = newId(), p2 = newId();
                    E(SpvOpLoad, {tVec4, p0, pp0});
                    E(SpvOpLoad, {tVec4, p1, pp1});
                    E(SpvOpLoad, {tVec4, p2, pp2});

                    // v0 = p0.xyz
                    uint32_t v0x = newId(), v0y = newId(), v0z = newId(), v0 = newId();
                    E(SpvOpCompositeExtract, {tFloat, v0x, p0, 0});
                    E(SpvOpCompositeExtract, {tFloat, v0y, p0, 1});
                    E(SpvOpCompositeExtract, {tFloat, v0z, p0, 2});
                    E(SpvOpCompositeConstruct, {tVec3, v0, v0x, v0y, v0z});
                    // e1 = vec3(p0.w, p1.x, p1.y) - v0
                    uint32_t p0w = newId(), p1x = newId(), p1y = newId();
                    E(SpvOpCompositeExtract, {tFloat, p0w, p0, 3});
                    E(SpvOpCompositeExtract, {tFloat, p1x, p1, 0});
                    E(SpvOpCompositeExtract, {tFloat, p1y, p1, 1});
                    uint32_t e1pre = newId(), e1 = newId();
                    E(SpvOpCompositeConstruct, {tVec3, e1pre, p0w, p1x, p1y});
                    E(SpvOpFSub, {tVec3, e1, e1pre, v0});
                    // e2 = vec3(p1.z, p1.w, p2.x) - v0
                    uint32_t p1z = newId(), p1w = newId(), p2x = newId();
                    E(SpvOpCompositeExtract, {tFloat, p1z, p1, 2});
                    E(SpvOpCompositeExtract, {tFloat, p1w, p1, 3});
                    E(SpvOpCompositeExtract, {tFloat, p2x, p2, 0});
                    uint32_t e2pre = newId(), e2 = newId();
                    E(SpvOpCompositeConstruct, {tVec3, e2pre, p1z, p1w, p2x});
                    E(SpvOpFSub, {tVec3, e2, e2pre, v0});

                    // Moller-Trumbore using LOCAL-space ray
                    // Load local ro and rd from variables
                    uint32_t lRo = newId(), lRd = newId();
                    E(SpvOpLoad, {tVec3, lRo, tvLocalRo});
                    E(SpvOpLoad, {tVec3, lRd, tvLocalRd});
                    uint32_t ppv = newId();
                    E(SpvOpExtInst, {tVec3, ppv, glslExtId, (uint32_t)GLSLstd450Cross, lRd, e2});
                    uint32_t det = newId();
                    spvEmit(out, (uint16_t)148, {tFloat, det, e1, ppv});
                    uint32_t invDet = newId();
                    E(SpvOpFDiv, {tFloat, invDet, cf1, det});
                    uint32_t tv = newId();
                    E(SpvOpFSub, {tVec3, tv, lRo, v0});
                    uint32_t dtvpp = newId(), uu = newId();
                    spvEmit(out, (uint16_t)148, {tFloat, dtvpp, tv, ppv});
                    E(SpvOpFMul, {tFloat, uu, invDet, dtvpp});
                    uint32_t qq = newId();
                    E(SpvOpExtInst, {tVec3, qq, glslExtId, (uint32_t)GLSLstd450Cross, tv, e1});
                    uint32_t drdqq = newId(), vv = newId();
                    spvEmit(out, (uint16_t)148, {tFloat, drdqq, lRd, qq});
                    E(SpvOpFMul, {tFloat, vv, invDet, drdqq});
                    uint32_t de2qq = newId(), tt = newId();
                    spvEmit(out, (uint16_t)148, {tFloat, de2qq, e2, qq});
                    E(SpvOpFMul, {tFloat, tt, invDet, de2qq});
                    uint32_t uv = newId();
                    E(SpvOpFAdd, {tFloat, uv, uu, vv});

                    // Hit check
                    uint32_t curBest = newId();
                    E(SpvOpLoad, {tFloat, curBest, tvBestT});
                    uint32_t c_uu0 = newId(), c_uu1 = newId(), c_vv0 = newId(), c_uv1 = newId();
                    uint32_t c_ttMin = newId(), c_ttMax = newId();
                    E(SpvOpFOrdGreaterThanEqual, {tBool, c_uu0, uu, cf0});
                    E(SpvOpFOrdLessThanEqual,    {tBool, c_uu1, uu, cf1});
                    E(SpvOpFOrdGreaterThanEqual, {tBool, c_vv0, vv, cf0});
                    E(SpvOpFOrdLessThanEqual,    {tBool, c_uv1, uv, cf1});
                    E(SpvOpFOrdGreaterThan,      {tBool, c_ttMin, tt, tMinId});
                    E(SpvOpFOrdLessThan,         {tBool, c_ttMax, tt, curBest});
                    uint32_t ca = newId(), cb = newId(), cc = newId(), cd = newId(), hitCond = newId();
                    uint32_t SpvOpLogicalAnd = 167;
                    spvEmit(out, (uint16_t)SpvOpLogicalAnd, {tBool, ca, c_uu0, c_uu1});
                    spvEmit(out, (uint16_t)SpvOpLogicalAnd, {tBool, cb, ca, c_vv0});
                    spvEmit(out, (uint16_t)SpvOpLogicalAnd, {tBool, cc, cb, c_uv1});
                    spvEmit(out, (uint16_t)SpvOpLogicalAnd, {tBool, cd, cc, c_ttMin});
                    spvEmit(out, (uint16_t)SpvOpLogicalAnd, {tBool, hitCond, cd, c_ttMax});

                    // if (hit) update bestT, bary, hitPrim, hitType, hitInst
                    uint32_t lHitYes = newId(), lHitEnd = newId();
                    spvEmit(out, SpvOpSelectionMerge, {lHitEnd, 0});
                    E(SpvOpBranchConditional, {hitCond, lHitYes, lHitEnd});
                    spvEmit(out, SpvOpLabel, {lHitYes});
                    E(SpvOpStore, {tvBestT, tt});
                    E(SpvOpStore, {tvHitU, uu});
                    E(SpvOpStore, {tvHitV, vv});
                    E(SpvOpStore, {tvHitPrim, tst0}); // BLAS-local primitive index
                    E(SpvOpStore, {tvHitType, cu1});
                    E(SpvOpStore, {tvHitInst, instIdx}); // record which instance was hit
                    E(SpvOpBranch, {lHitEnd});
                    spvEmit(out, SpvOpLabel, {lHitEnd});

                    // tri loop continue: t++
                    E(SpvOpBranch, {triC});
                    spvEmit(out, SpvOpLabel, {triC});
                    uint32_t tNext = newId();
                    E(SpvOpIAdd, {tInt, tNext, tVal, c1});
                    E(SpvOpStore, {tvTriIdx, tNext});
                    E(SpvOpBranch, {triH});
                    spvEmit(out, SpvOpLabel, {triM});
                }
                // ni = skip (BLAS)
                E(SpvOpStore, {tvNi, skipVal});
                E(SpvOpBranch, {lEndIf});

                // --- BLAS INTERNAL: Ray-AABB test (local-space) ---
                spvEmit(out, SpvOpLabel, {lInternal});
                {
                    uint32_t w0x = newId(), w0y = newId(), w0z = newId(), w0w = newId();
                    uint32_t w1x = newId(), w1y = newId();
                    E(SpvOpCompositeExtract, {tUint, w0x, w0, 0});
                    E(SpvOpCompositeExtract, {tUint, w0y, w0, 1});
                    E(SpvOpCompositeExtract, {tUint, w0z, w0, 2});
                    E(SpvOpCompositeExtract, {tUint, w0w, w0, 3});
                    E(SpvOpCompositeExtract, {tUint, w1x, w1, 0});
                    E(SpvOpCompositeExtract, {tUint, w1y, w1, 1});
                    uint32_t fx0 = newId(), fy0 = newId(), fz0 = newId();
                    uint32_t fx1 = newId(), fy1 = newId(), fz1 = newId();
                    E(SpvOpBitcast, {tFloat, fx0, w0x});
                    E(SpvOpBitcast, {tFloat, fy0, w0y});
                    E(SpvOpBitcast, {tFloat, fz0, w0z});
                    E(SpvOpBitcast, {tFloat, fx1, w0w});
                    E(SpvOpBitcast, {tFloat, fy1, w1x});
                    E(SpvOpBitcast, {tFloat, fz1, w1y});

                    // AABB test using LOCAL-space invD, ood
                    uint32_t t1x = newId(), t2x = newId(), t1y = newId(), t2y = newId();
                    uint32_t t1z = newId(), t2z = newId();
                    E(SpvOpExtInst, {tFloat, t1x, glslExtId, (uint32_t)GLSLstd450Fma, fx0, lInvDx, lOodx});
                    E(SpvOpExtInst, {tFloat, t2x, glslExtId, (uint32_t)GLSLstd450Fma, fx1, lInvDx, lOodx});
                    E(SpvOpExtInst, {tFloat, t1y, glslExtId, (uint32_t)GLSLstd450Fma, fy0, lInvDy, lOody});
                    E(SpvOpExtInst, {tFloat, t2y, glslExtId, (uint32_t)GLSLstd450Fma, fy1, lInvDy, lOody});
                    E(SpvOpExtInst, {tFloat, t1z, glslExtId, (uint32_t)GLSLstd450Fma, fz0, lInvDz, lOodz});
                    E(SpvOpExtInst, {tFloat, t2z, glslExtId, (uint32_t)GLSLstd450Fma, fz1, lInvDz, lOodz});

                    uint32_t mn_x = newId(), mn_y = newId(), mn_z = newId();
                    E(SpvOpExtInst, {tFloat, mn_x, glslExtId, (uint32_t)GLSLstd450FMin, t1x, t2x});
                    E(SpvOpExtInst, {tFloat, mn_y, glslExtId, (uint32_t)GLSLstd450FMin, t1y, t2y});
                    E(SpvOpExtInst, {tFloat, mn_z, glslExtId, (uint32_t)GLSLstd450FMin, t1z, t2z});
                    uint32_t tN_xy = newId(), tN = newId();
                    E(SpvOpExtInst, {tFloat, tN_xy, glslExtId, (uint32_t)GLSLstd450FMax, mn_x, mn_y});
                    E(SpvOpExtInst, {tFloat, tN,    glslExtId, (uint32_t)GLSLstd450FMax, tN_xy, mn_z});

                    uint32_t mx_x = newId(), mx_y = newId(), mx_z = newId();
                    E(SpvOpExtInst, {tFloat, mx_x, glslExtId, (uint32_t)GLSLstd450FMax, t1x, t2x});
                    E(SpvOpExtInst, {tFloat, mx_y, glslExtId, (uint32_t)GLSLstd450FMax, t1y, t2y});
                    E(SpvOpExtInst, {tFloat, mx_z, glslExtId, (uint32_t)GLSLstd450FMax, t1z, t2z});
                    uint32_t tF_xy = newId(), tF = newId();
                    E(SpvOpExtInst, {tFloat, tF_xy, glslExtId, (uint32_t)GLSLstd450FMin, mx_x, mx_y});
                    E(SpvOpExtInst, {tFloat, tF,    glslExtId, (uint32_t)GLSLstd450FMin, tF_xy, mx_z});

                    uint32_t curBestI = newId();
                    E(SpvOpLoad, {tFloat, curBestI, tvBestT});
                    uint32_t c1_ = newId(), c2_ = newId(), c3_ = newId(), aabbHit = newId(), ah2 = newId();
                    E(SpvOpFOrdLessThanEqual, {tBool, c1_, tN, tF});
                    E(SpvOpFOrdGreaterThan,   {tBool, c2_, tF, cf0});
                    E(SpvOpFOrdLessThan,      {tBool, c3_, tN, curBestI});
                    spvEmit(out, (uint16_t)167, {tBool, aabbHit, c1_, c2_});
                    spvEmit(out, (uint16_t)167, {tBool, ah2, aabbHit, c3_});

                    uint32_t niPlus1 = newId(), newNi = newId();
                    E(SpvOpIAdd, {tInt, niPlus1, niVal, c1});
                    E(SpvOpSelect, {tInt, newNi, ah2, niPlus1, skipVal});
                    E(SpvOpStore, {tvNi, newNi});
                }
                E(SpvOpBranch, {lEndIf});

                // End of BLAS leaf/internal selection
                spvEmit(out, SpvOpLabel, {lEndIf});
                E(SpvOpBranch, {lContinue});

                spvEmit(out, SpvOpLabel, {lContinue});
                {
                    uint32_t curIter = newId(), nextIter = newId();
                    E(SpvOpLoad, {tInt, curIter, tvIter});
                    E(SpvOpIAdd, {tInt, nextIter, curIter, c1});
                    E(SpvOpStore, {tvIter, nextIter});
                }
                E(SpvOpBranch, {lHeader});

                spvEmit(out, SpvOpLabel, {lMerge});
                // End of BLAS inner loop

                // TLAS leaf done → ni = skip (continue TLAS)
                E(SpvOpStore, {tvTlasNi, tlSkipVal});
#endif // !SPIRV_RQ_TLAS_ONLY
            }
            E(SpvOpBranch, {tlEndIfLbl});

            // --- TLAS INTERNAL: Ray-AABB test (world-space) ---
            spvEmit(out, SpvOpLabel, {tlInternalLbl});
            {
                uint32_t tw0x = newId(), tw0y = newId(), tw0z = newId(), tw0w = newId();
                uint32_t tw1x = newId(), tw1y = newId();
                E(SpvOpCompositeExtract, {tUint, tw0x, tw0, 0});
                E(SpvOpCompositeExtract, {tUint, tw0y, tw0, 1});
                E(SpvOpCompositeExtract, {tUint, tw0z, tw0, 2});
                E(SpvOpCompositeExtract, {tUint, tw0w, tw0, 3});
                E(SpvOpCompositeExtract, {tUint, tw1x, tw1, 0});
                E(SpvOpCompositeExtract, {tUint, tw1y, tw1, 1});
                uint32_t tfx0 = newId(), tfy0 = newId(), tfz0 = newId();
                uint32_t tfx1 = newId(), tfy1 = newId(), tfz1 = newId();
                E(SpvOpBitcast, {tFloat, tfx0, tw0x});
                E(SpvOpBitcast, {tFloat, tfy0, tw0y});
                E(SpvOpBitcast, {tFloat, tfz0, tw0z});
                E(SpvOpBitcast, {tFloat, tfx1, tw0w});
                E(SpvOpBitcast, {tFloat, tfy1, tw1x});
                E(SpvOpBitcast, {tFloat, tfz1, tw1y});

                // AABB test using WORLD-space invD, ood
                uint32_t tt1x = newId(), tt2x = newId(), tt1y = newId(), tt2y = newId();
                uint32_t tt1z = newId(), tt2z = newId();
                E(SpvOpExtInst, {tFloat, tt1x, glslExtId, (uint32_t)GLSLstd450Fma, tfx0, wInvDx, wOodx});
                E(SpvOpExtInst, {tFloat, tt2x, glslExtId, (uint32_t)GLSLstd450Fma, tfx1, wInvDx, wOodx});
                E(SpvOpExtInst, {tFloat, tt1y, glslExtId, (uint32_t)GLSLstd450Fma, tfy0, wInvDy, wOody});
                E(SpvOpExtInst, {tFloat, tt2y, glslExtId, (uint32_t)GLSLstd450Fma, tfy1, wInvDy, wOody});
                E(SpvOpExtInst, {tFloat, tt1z, glslExtId, (uint32_t)GLSLstd450Fma, tfz0, wInvDz, wOodz});
                E(SpvOpExtInst, {tFloat, tt2z, glslExtId, (uint32_t)GLSLstd450Fma, tfz1, wInvDz, wOodz});

                uint32_t tmn_x = newId(), tmn_y = newId(), tmn_z = newId();
                E(SpvOpExtInst, {tFloat, tmn_x, glslExtId, (uint32_t)GLSLstd450FMin, tt1x, tt2x});
                E(SpvOpExtInst, {tFloat, tmn_y, glslExtId, (uint32_t)GLSLstd450FMin, tt1y, tt2y});
                E(SpvOpExtInst, {tFloat, tmn_z, glslExtId, (uint32_t)GLSLstd450FMin, tt1z, tt2z});
                uint32_t ttN_xy = newId(), ttN = newId();
                E(SpvOpExtInst, {tFloat, ttN_xy, glslExtId, (uint32_t)GLSLstd450FMax, tmn_x, tmn_y});
                E(SpvOpExtInst, {tFloat, ttN,    glslExtId, (uint32_t)GLSLstd450FMax, ttN_xy, tmn_z});

                uint32_t tmx_x = newId(), tmx_y = newId(), tmx_z = newId();
                E(SpvOpExtInst, {tFloat, tmx_x, glslExtId, (uint32_t)GLSLstd450FMax, tt1x, tt2x});
                E(SpvOpExtInst, {tFloat, tmx_y, glslExtId, (uint32_t)GLSLstd450FMax, tt1y, tt2y});
                E(SpvOpExtInst, {tFloat, tmx_z, glslExtId, (uint32_t)GLSLstd450FMax, tt1z, tt2z});
                uint32_t ttF_xy = newId(), ttF = newId();
                E(SpvOpExtInst, {tFloat, ttF_xy, glslExtId, (uint32_t)GLSLstd450FMin, tmx_x, tmx_y});
                E(SpvOpExtInst, {tFloat, ttF,    glslExtId, (uint32_t)GLSLstd450FMin, ttF_xy, tmx_z});

                uint32_t tlCurBest = newId();
                E(SpvOpLoad, {tFloat, tlCurBest, tvBestT});
                uint32_t tc1 = newId(), tc2 = newId(), tc3 = newId(), tlAabbHit = newId(), tlAh2 = newId();
                E(SpvOpFOrdLessThanEqual, {tBool, tc1, ttN, ttF});
                E(SpvOpFOrdGreaterThan,   {tBool, tc2, ttF, cf0});
                E(SpvOpFOrdLessThan,      {tBool, tc3, ttN, tlCurBest});
                spvEmit(out, (uint16_t)167, {tBool, tlAabbHit, tc1, tc2});
                spvEmit(out, (uint16_t)167, {tBool, tlAh2, tlAabbHit, tc3});

                uint32_t tlNiPlus1 = newId(), tlNewNi = newId();
                E(SpvOpIAdd, {tInt, tlNiPlus1, tlNiVal, c1});
#if SPIRV_RQ_FORCE_TLAS_HIT
                // Force AABB test to always pass — always go to left child
                E(SpvOpCopyObject, {tInt, tlNewNi, tlNiPlus1});
#else
                E(SpvOpSelect, {tInt, tlNewNi, tlAh2, tlNiPlus1, tlSkipVal});
#endif
                E(SpvOpStore, {tvTlasNi, tlNewNi});
            }
            E(SpvOpBranch, {tlEndIfLbl});

            // End of TLAS leaf/internal selection
            spvEmit(out, SpvOpLabel, {tlEndIfLbl});
            E(SpvOpBranch, {tlCont});

            spvEmit(out, SpvOpLabel, {tlCont});
            {
                uint32_t curTlasIter = newId(), nextTlasIter = newId();
                E(SpvOpLoad, {tInt, curTlasIter, tvTlasIter});
                E(SpvOpIAdd, {tInt, nextTlasIter, curTlasIter, c1});
                E(SpvOpStore, {tvTlasIter, nextTlasIter});
            }
            E(SpvOpBranch, {tlH});

            // --- TLAS Loop merge: store results to struct ---
            spvEmit(out, SpvOpLabel, {tlMerge});
            {
                // Only overwrite struct members if we actually found a hit.
                // On miss, keep the initial values (hitT=1e30, hitType=0, etc.)
                uint32_t finalType = newId();
                E(SpvOpLoad, {tUint, finalType, tvHitType});
                uint32_t hasHit = newId();
                E(SpvOpINotEqual, {tBool, hasHit, finalType, cu0});
                uint32_t storeLbl = newId(), skipStoreLbl = newId(), mergeStoreLbl = newId();
                spvEmit(out, SpvOpSelectionMerge, {mergeStoreLbl, 0});
                E(SpvOpBranchConditional, {hasHit, storeLbl, skipStoreLbl});
                spvEmit(out, SpvOpLabel, {storeLbl});
                {
                    // Store bestT → member 4 (hitT)
                    uint32_t finalT = newId(), pa4 = newId();
                    E(SpvOpLoad, {tFloat, finalT, tvBestT});
                    E(SpvOpAccessChain, {pFloat, pa4, sv, c4});
                    E(SpvOpStore, {pa4, finalT});
                    // Store hitU, hitV → member 5 (bary as vec2)
                    uint32_t finalU = newId(), finalV = newId(), bary = newId(), pa5 = newId();
                    E(SpvOpLoad, {tFloat, finalU, tvHitU});
                    E(SpvOpLoad, {tFloat, finalV, tvHitV});
                    E(SpvOpCompositeConstruct, {tVec2, bary, finalU, finalV});
                    E(SpvOpAccessChain, {pVec2, pa5, sv, c5});
                    E(SpvOpStore, {pa5, bary});
                    // Store hitPrim → member 6
                    uint32_t finalPrim = newId(), pa6 = newId();
                    E(SpvOpLoad, {tInt, finalPrim, tvHitPrim});
                    E(SpvOpAccessChain, {pInt, pa6, sv, c6});
                    E(SpvOpStore, {pa6, finalPrim});
                    // Store hitInst → member 7
                    uint32_t finalInst = newId(), pa7 = newId();
                    E(SpvOpLoad, {tInt, finalInst, tvHitInst});
                    E(SpvOpAccessChain, {pInt, pa7, sv, c7});
                    E(SpvOpStore, {pa7, finalInst});
                    // Store hitType → member 8 (uint)
                    uint32_t pa8 = newId();
                    E(SpvOpAccessChain, {pUint, pa8, sv, c8});
                    E(SpvOpStore, {pa8, finalType});
                }
                E(SpvOpBranch, {mergeStoreLbl});
                spvEmit(out, SpvOpLabel, {skipStoreLbl});
                E(SpvOpBranch, {mergeStoreLbl});
                spvEmit(out, SpvOpLabel, {mergeStoreLbl});
            }

            fprintf(stderr, "[SPIRV-RQ] V2 2-level BVH traversal emitted for rq=%u\n", rqVar);
#else
            // NO-HIT STUB: skip traversal, just leave hitType=0 (miss)
            fprintf(stderr, "[SPIRV-RQ] V2 NO-HIT STUB for rq=%u (traversal disabled)\n", rqVar);
#endif

            skip = true;
        }

        // --- Replace OpRayQueryProceedKHR ---
        if (op == SpvOpRQProceed && wc >= 4) {
            uint32_t resultId = code[pos+2];
            // Return false: traversal complete in one call (eager mode)
            spvEmit(out, SpvOpCopyObject, {tBool, resultId, cFalse});
            skip = true;
        }

        // --- Replace Get* operations ---
        // Helper: read struct member as given type (uses pre-allocated constants)
        uint32_t cArr[11] = {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10};
        auto getField = [&](uint32_t resultId, uint32_t rqVar, int memberIdx, uint32_t fieldType, uint32_t ptrType) {
            uint32_t sv = rqMap.count(rqVar) ? rqMap[rqVar] : rqVar;
            uint32_t ac = newId();
            spvEmit(out, SpvOpAccessChain, {ptrType, ac, sv, cArr[memberIdx]});
            spvEmit(out, SpvOpLoad, {fieldType, resultId, ac});
        };

        if (op == SpvOpRQGetT && wc >= 5) {
            getField(code[pos+2], code[pos+3], 4, tFloat, pFloat); skip = true;
        }
        if (op == SpvOpRQGetBarycentrics && wc >= 5) {
            getField(code[pos+2], code[pos+3], 5, tVec2, pVec2); skip = true;
        }
        if (op == SpvOpRQGetPrimIdx && wc >= 5) {
            getField(code[pos+2], code[pos+3], 6, tInt, pInt); skip = true;
        }
        if (op == SpvOpRQGetInstId && wc >= 5) {
            getField(code[pos+2], code[pos+3], 7, tInt, pInt); skip = true;
        }
        if (op == SpvOpRQGetIntersectionType && wc >= 5) {
            getField(code[pos+2], code[pos+3], 8, tUint, pUint); skip = true;
        }
        if (op == SpvOpRQGetWorldRayDir && wc >= 4) {
            uint32_t sv = rqMap.count(code[pos+3]) ? rqMap[code[pos+3]] : code[pos+3];
            uint32_t ac = newId();
            spvEmit(out, SpvOpAccessChain, {pVec3, ac, sv, c1});
            spvEmit(out, SpvOpLoad, {tVec3, code[pos+2], ac});
            skip = true;
        }
        if (op == SpvOpRQGetWorldRayOrigin && wc >= 4) {
            uint32_t sv = rqMap.count(code[pos+3]) ? rqMap[code[pos+3]] : code[pos+3];
            uint32_t ac = newId();
            spvEmit(out, SpvOpAccessChain, {pVec3, ac, sv, c0});
            spvEmit(out, SpvOpLoad, {tVec3, code[pos+2], ac});
            skip = true;
        }
        if (op == SpvOpRQGetRayTMin && wc >= 4) {
            uint32_t sv = rqMap.count(code[pos+3]) ? rqMap[code[pos+3]] : code[pos+3];
            uint32_t ac = newId();
            spvEmit(out, SpvOpAccessChain, {pFloat, ac, sv, c2});
            spvEmit(out, SpvOpLoad, {tFloat, code[pos+2], ac});
            skip = true;
        }
        if (op == SpvOpRQGetFrontFace && wc >= 5) {
            spvEmit(out, SpvOpCopyObject, {tBool, code[pos+2], cTrue});
            skip = true;
        }
        // Return 0 for inst custom idx, geom idx, SBT offset, ray flags
        if ((op == SpvOpRQGetInstCustomIdx || op == SpvOpRQGetGeomIdx ||
             op == SpvOpRQGetSBTOffset || op == SpvOpRQGetRayFlags) && wc >= 4) {
            spvEmit(out, SpvOpCopyObject, {tInt, code[pos+2], c0});
            skip = true;
        }
        // Object-space ray = world-space ray (no transforms for V1)
        if (op == SpvOpRQGetObjRayDir && wc >= 5) {
            uint32_t sv = rqMap.count(code[pos+3]) ? rqMap[code[pos+3]] : code[pos+3];
            uint32_t ac = newId();
            spvEmit(out, SpvOpAccessChain, {pVec3, ac, sv, c1});
            spvEmit(out, SpvOpLoad, {tVec3, code[pos+2], ac});
            skip = true;
        }
        if (op == SpvOpRQGetObjRayOrigin && wc >= 5) {
            uint32_t sv = rqMap.count(code[pos+3]) ? rqMap[code[pos+3]] : code[pos+3];
            uint32_t ac = newId();
            spvEmit(out, SpvOpAccessChain, {pVec3, ac, sv, c0});
            spvEmit(out, SpvOpLoad, {tVec3, code[pos+2], ac});
            skip = true;
        }

        // Terminate / Confirm — no-ops
        if (op == SpvOpRQTerminate || op == SpvOpRQConfirmIntersection) skip = true;
        // AABB opaque check — return true
        if (op == SpvOpRQGetCandidateAABBOpaque && wc >= 4) {
            spvEmit(out, SpvOpCopyObject, {tBool, code[pos+2], cTrue});
            skip = true;
        }
        } // end of RQ operations rewrite block
#endif // !SPIRV_RQ_MINIMAL (end of full-rewrite RQ ops)

        // DIAGNOSTIC: Force all image writes to white vec4(1,1,1,1)
#define SPIRV_RQ_FORCE_WHITE 0
#if SPIRV_RQ_FORCE_WHITE
        if (op == 99 /*OpImageWrite*/ && wc >= 4) {
            uint32_t whiteVec = newId();
            spvEmit(out, SpvOpCompositeConstruct, {tVec4, whiteVec, cf1, cf1, cf1, cf1});
            std::vector<uint32_t> args;
            args.push_back(code[pos+1]);
            args.push_back(code[pos+2]);
            args.push_back(whiteVec);
            for (uint16_t j = 4; j < wc; j++)
                args.push_back(code[pos+j]);
            spvEmit(out, (uint16_t)99, args);
            skip = true;
        }
#endif
        // DIAGNOSTIC: Skip image writes in shaders where entry point has RQ ops
        // Tests if these shaders overwrite other shaders' correct output
#define SPIRV_RQ_SKIP_ENTRY_RQ_WRITES 0
#if SPIRV_RQ_SKIP_ENTRY_RQ_WRITES
        if (op == 99 /*OpImageWrite*/ && entryHasRQ) {
            skip = true; // drop the image write entirely
        }
#endif

        // DIAGNOSTIC: Replace calls to RQ-containing functions with constant
#define SPIRV_RQ_STUB_CALLS 0
#if SPIRV_RQ_STUB_CALLS
        if (op == 57 /*OpFunctionCall*/ && wc >= 4) {
            uint32_t resultType = code[pos+1];
            uint32_t resultId = code[pos+2];
            uint32_t calledFunc = code[pos+3];
            bool isRqFunc = rqFuncIds.count(calledFunc) > 0;
            if (isRqFunc) {
                if (nativeFloat1 && resultType == nativeFloatType) {
                    spvEmit(out, SpvOpCopyObject, {nativeFloatType, resultId, nativeFloat1});
                    fprintf(stderr, "[SPIRV-RQ] STUB_CALL: %%%u = call %%%u -> 1.0f\n", resultId, calledFunc);
                } else {
                    spvEmit(out, 1 /*OpUndef*/, {resultType, resultId});
                    fprintf(stderr, "[SPIRV-RQ] STUB_CALL: %%%u = call %%%u -> OpUndef (type %%%u)\n",
                            resultId, calledFunc, resultType);
                }
                skip = true;
            }
        }
#endif

        // Copy unchanged instructions
        copy_unchanged:
        if (!skip) {
            for (uint16_t j = 0; j < wc; j++)
                out.push_back(code[pos + j]);
        }
        pos += wc;
    }

    out[3] = nextId; // patch Bound
    fprintf(stderr, "[SPIRV-RQ] Rewrite: %zu -> %zu words, bound %u\n",
            numWords, out.size(), nextId);

#if SPIRV_RQ_DUMP
    {
        static int dumpIdx = 0;
        char path[128];
        snprintf(path, sizeof(path), "/tmp/spirv_rq_dump_%d.spv", dumpIdx++);
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(out.data(), 4, out.size(), f);
            fclose(f);
            fprintf(stderr, "[SPIRV-RQ] Dumped %zu words to %s\n", out.size(), path);
        }
    }
#endif

    return out;
}

// ============================================================================
// High-level API
// ============================================================================
struct SpirvRewriteResult {
    bool rewritten;
    std::vector<uint32_t> code;
    int bvhDescSet;
    int bvhNodesBinding;
    int bvhTrisBinding;
    int bvhTlasBinding;
    int bvhInstBinding;
};

static SpirvRewriteResult spirvTryRewriteRayQuery(
    const uint32_t* code, size_t numWords)
{
    SpirvRewriteResult r = {};
    r.rewritten = false;
    if (!spirvHasRayQuery(code, numWords)) return r;

    int maxSet = spirvMaxDescriptorSet(code, numWords);
    r.bvhDescSet = maxSet + 1;
    r.bvhNodesBinding = 0;
    r.bvhTrisBinding = 1;
    r.bvhTlasBinding = 2;
    r.bvhInstBinding = 3;

    fprintf(stderr, "[SPIRV-RQ] Shader has ray queries! Rewriting (BVH set=%d)\n", r.bvhDescSet);
    r.code = spirvRewriteRayQuery(code, numWords, r.bvhDescSet, r.bvhNodesBinding, r.bvhTrisBinding);
    r.rewritten = !r.code.empty();
    return r;
}
