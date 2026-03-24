# 🎯 RT Quality / DLSS-Style Pipeline Checklist

## 🥇 TIER 1 — SIGNAL QUALITY (biggest visual gains)

### 1. Importance Sampling (MUST)

* [x] Cosine-weighted diffuse sampling
* [x] GGX / microfacet importance sampling for specular (VNDF, Heitz 2018)
* [x] Next Event Estimation (direct light sampling)
* [x] Multiple Importance Sampling (MIS) for light + BRDF (power heuristic)
* [ ] Environment map importance sampling (if HDRI used)

**Goal:** fewer noisy rays → more useful contribution per ray

---

### 2. Variance Control

* [x] Firefly clamping (limit extreme samples)
* [ ] Clamp indirect lighting separately
* [ ] Normalize BRDF energy (no over-bright spikes)
* [x] Russian roulette termination (depth-based)

**Goal:** remove rare but destructive high-energy noise

---

### 3. Ray Budget Allocation

* [x] More rays for primary / first bounce
* [ ] Fewer rays for deeper bounces
* [x] Limit max bounces (start with 2–4)
* [ ] Separate paths: diffuse vs specular

**Goal:** spend rays where they matter most

---

## 🥈 TIER 2 — TEMPORAL REUSE (HUGE multiplier)

### 4. Temporal Accumulation

* [x] Accumulate previous frames (history buffer)
* [x] Exponential moving average or weighted blend
* [ ] Reset/decay on large changes (camera/object motion)

---

### 5. Motion Vectors (CRITICAL)

* [ ] Per-pixel motion vectors (prev → current)
* [ ] Reproject history correctly
* [ ] Handle disocclusion (newly visible pixels)

---

### 6. History Validation

* [ ] Depth test (reject mismatched geometry)
* [ ] Normal test (reject wrong surfaces)
* [ ] Clamp history (avoid ghosting)

**Goal:** reuse past samples safely → 10–50× effective samples

---

## 🥉 TIER 3 — DENOISING (Tensor cores sweet spot)

### 7. Spatial Denoiser (baseline)

* [x] Edge-aware filter (bilateral / cross-bilateral)
* [x] Use depth + normal buffers as guides

---

### 8. Temporal Denoiser (better)

* [ ] Combine current + history
* [ ] Stabilize flicker across frames

---

### 9. Neural Denoiser (advanced)

* [ ] Small CNN (U-Net style)
* [ ] Inputs: noisy color + albedo + normals + depth
* [ ] Run on Tensor cores if available

**Goal:** turn 100–300 MRays into “clean” output

---

## 🧠 TIER 4 — UPSCALING (DLSS-style)

### 10. Render at Lower Resolution

* [ ] 50–70% internal resolution
* [ ] Maintain high-quality G-buffer (depth, normals)

---

### 11. Temporal Upscaling

* [ ] Use motion vectors + history
* [ ] Reconstruct high-frequency detail

---

### 12. Sharpening / Final Pass

* [ ] Mild sharpening after denoise/upscale
* [ ] Avoid amplifying noise

---

## ⚡ TIER 5 — HYBRID STRATEGY

### 13. Selective Ray Tracing

* [ ] RT only for reflections / GI / shadows
* [ ] Raster for base shading

---

### 14. Screen-Space First

* [ ] SSR / SSAO fallback
* [ ] RT only where screen-space fails

---

### 15. Resolution Scaling for RT

* [ ] RT at half/quarter res
* [ ] Upscale + denoise

---

## 🧪 METRICS TO TRACK (QUALITY, not just speed)

* [ ] Noise level per pixel (visual / variance)
* [ ] Stability across frames (flicker ↓)
* [ ] Ghosting artifacts
* [ ] Rays per pixel (effective)
* [ ] Denoiser error vs ground truth (if available)

---

## 🎯 TARGET STATE

* [ ] Stable image at low samples (~1–4 spp/frame)
* [ ] Minimal flicker with camera motion
* [ ] Clean reflections / GI without fireflies
* [ ] Real-time capable with hybrid + temporal reuse

---

## 🔥 FINAL GOAL

* [ ] “Low rays + smart reconstruction” > brute force rendering
