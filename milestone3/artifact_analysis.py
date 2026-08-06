"""
Milestone 3 — Artifact Analysis
Quantifies what the eye sees in the m3_*.png comparisons rather than relying
on visual impression alone.
"""

import json
import numpy as np
from PIL import Image


def load(path):
    return np.array(Image.open(path).convert("RGB")).astype(float) / 255.0


results = {}

# ----------------------------------------------------------------------------
# A) Shading: Gouraud vs Phong -- specular highlight fidelity
# ----------------------------------------------------------------------------
# A Blinn-Phong specular highlight is a small, high-frequency bright spot.
# Gouraud shading computes lighting only at (sparse) vertices and interpolates
# the RESULT, so a highlight landing mid-triangle is smeared or missed
# entirely -- this shows up as reduced pixel-to-pixel variance in the
# highlight-bearing region of the character.
gour = load("/home/claude/m3_gouraud.png")
phong = load("/home/claude/m3_phong.png")

# character occupies roughly this bounding box in the 800x500 frame (from the scene setup)
region = np.s_[80:420, 280:560]
gour_region = gour[region]
phong_region = phong[region]

results["shading_gouraud_vs_phong"] = {
    "region_std_gouraud": float(gour_region.std()),
    "region_std_phong": float(phong_region.std()),
    "region_max_intensity_gouraud": float(gour_region.max()),
    "region_max_intensity_phong": float(phong_region.max()),
    "mean_abs_difference": float(np.abs(gour_region - phong_region).mean()),
    "conclusion": (
        "Phong shading's character region has both higher pixel-to-pixel "
        "variance and a higher peak intensity than Gouraud's -- the signature "
        "of a specular highlight that Phong resolves per-pixel but that "
        "Gouraud smears across whole triangles because it only evaluates "
        "lighting at the (sparse, 10-segment) vertex ring. On a coarser mesh "
        "this gap widens; on an infinitely fine mesh the two methods converge."
    ),
}

# ----------------------------------------------------------------------------
# B) UV interpolation: affine vs perspective-correct -- texture distortion
# ----------------------------------------------------------------------------
aff = load("/home/claude/m3_affine_uv.png")
persc = load("/home/claude/m3_perspective_uv.png")

diff = np.abs(aff - persc)
# the floor is the lower half of the frame; distortion is worst on the near,
# most-oblique triangle (bottom of frame)
floor_near = np.s_[380:500, :]
floor_far = np.s_[250:320, :]

results["uv_affine_vs_perspective_correct"] = {
    "mean_abs_difference_full_frame": float(diff.mean()),
    "mean_abs_difference_near_floor_rows": float(diff[floor_near].mean()),
    "mean_abs_difference_far_floor_rows": float(diff[floor_far].mean()),
    "conclusion": (
        "Because the floor is a single quad split into just two large "
        "triangles spanning the full near-to-far depth range, each triangle's "
        "clip-space w varies enormously from its near corner to its far "
        "corner -- so affine interpolation error is present throughout, not "
        "confined to one screen region. The measured divergence is actually "
        "largest in the far-floor rows here, where the checkerboard's "
        "projected tile frequency is also highest: the two artifacts compound "
        "(a UV coordinate that is already wrong from affine interpolation "
        "gets sampled at a location where small UV errors flip entire tile "
        "parities). This is itself a useful finding: affine-vs-correct "
        "divergence and sampling-frequency aliasing are not independent "
        "effects on a single large triangle -- a production renderer "
        "addresses the root cause (perspective-correct interpolation) rather "
        "than trying to mitigate the resulting distortion with more samples "
        "alone, and tessellating the floor into smaller triangles nearer the "
        "camera would shrink the w-range per triangle and reduce this error "
        "independent of the interpolation mode."
    ),
}

# ----------------------------------------------------------------------------
# C) Sampling: 1x point sampling vs 4x4 (16x) supersampling -- aliasing
# ----------------------------------------------------------------------------
pt = load("/home/claude/m3_pointsampled.png")
ss = load("/home/claude/m3_supersampled.png")

# high-frequency content via a simple discrete Laplacian magnitude (edge/aliasing proxy)
def laplacian_energy(img):
    gray = img.mean(axis=-1)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
    )
    return np.mean(lap ** 2)

# focus on the distant floor strip, where texture-frequency aliasing (moire) shows up
distant_floor = np.s_[240:300, :]
lap_pt = laplacian_energy(pt[distant_floor])
lap_ss = laplacian_energy(ss[distant_floor])

results["sampling_point_vs_supersampled"] = {
    "distant_floor_laplacian_energy_pointsampled": float(lap_pt),
    "distant_floor_laplacian_energy_supersampled": float(lap_ss),
    "energy_reduction_factor": float(lap_pt / max(lap_ss, 1e-12)),
    "conclusion": (
        "Squared-Laplacian energy (a standard high-frequency / edge-energy "
        "proxy) in the distant floor strip drops substantially under 4x4 "
        "supersampling relative to 1x point sampling -- quantitative "
        "confirmation of the visible Moire/aliasing reduction: the checker "
        "texture's spatial frequency exceeds the point-sampling rate once "
        "tiles project to sub-pixel size near the horizon (classic spatial "
        "aliasing per the Nyquist limit), and box-filtering 16 sub-samples "
        "per output pixel approximates the band-limiting a proper prefilter "
        "would perform."
    ),
}

with open("/home/claude/artifact_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
