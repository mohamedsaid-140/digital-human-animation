"""
Milestone 4b driver.

Builds the Milestone 1 character's sphere-proxy occluder, casts Monte Carlo
shadow rays from a floor grid to a rectangular area light at several sample
counts K, and quantifies convergence against a high-K reference.

Outputs:
  m4_shadow_k{K}.png          - shadow maps at K = 4, 16, 64, 256
  m4_shadow_reference.png     - K = 2048 reference ("ground truth")
  m4_shadow_convergence.png   - RMSE vs K (log-log), with fitted slope
  m4_shadow_time_vs_k.png     - render time vs K (performance/cost side of the trade-off)
  m4_shadow_results.json      - all quantitative results
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from digital_human_core import Skeleton, SkinnedMesh
from mc_shadows import build_sphere_proxy, compute_soft_shadow

rng_master = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# Character pose (same stance as Milestone 3) -> sphere proxy occluder
# ----------------------------------------------------------------------------
skel = Skeleton()
skel.add_joint("hips", None, [0, 0, 0])
skel.add_joint("spine", "hips", [0, 0.25, 0])
skel.add_joint("chest", "spine", [0, 0.25, 0])
skel.add_joint("neck", "chest", [0, 0.15, 0])
skel.add_joint("head", "neck", [0, 0.15, 0])
skel.add_joint("shoulder_l", "chest", [0.18, 0.05, 0])
skel.add_joint("elbow_l", "shoulder_l", [0.25, 0, 0])
skel.add_joint("wrist_l", "elbow_l", [0.25, 0, 0])
skel.add_joint("shoulder_r", "chest", [-0.18, 0.05, 0])
skel.add_joint("elbow_r", "shoulder_r", [-0.25, 0, 0])
skel.add_joint("wrist_r", "elbow_r", [-0.25, 0, 0])
skel.add_joint("hip_l", "hips", [0.1, 0, 0])
skel.add_joint("knee_l", "hip_l", [0, -0.4, 0])
BONES = [
    ("hips", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
    ("chest", "shoulder_l"), ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("chest", "shoulder_r"), ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
    ("hips", "hip_l"), ("hip_l", "knee_l"),
]
bind_world = skel.bind_pose_world()
mesh = SkinnedMesh(skel)
for parent, child in BONES:
    mesh.add_bone_tube(parent, child, bind_world, radius=0.06, segments=8)
mesh.bind(bind_world)

posed_world = skel.forward_kinematics()
FEET_LIFT = np.array([0, 0.62, 0])
sphere_centers, sphere_radii = build_sphere_proxy(posed_world, skel, BONES, radius=0.09,
                                                   offset=FEET_LIFT, samples_per_bone=3)

# ----------------------------------------------------------------------------
# Floor grid (shadow receiver) directly beneath/around the character
# ----------------------------------------------------------------------------
GX, GZ = 100, 85
xs = np.linspace(-1.6, 1.6, GX)
zs = np.linspace(-1.4, 1.6, GZ)
XX, ZZ = np.meshgrid(xs, zs)
grid_points = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()], axis=-1)

LIGHT_CENTER = np.array([1.3, 3.2, 0.8])
LIGHT_HALF_EXTENTS = (0.9, 0.9)

K_VALUES = [4, 16, 64, 256]
K_REFERENCE = 1536

results = {"K": [], "time_s": [], "rmse_vs_reference": []}
shadow_maps = {}

for K in K_VALUES + [K_REFERENCE]:
    rng = np.random.default_rng(1000 + K)  # distinct stream per K, reproducible
    t0 = time.perf_counter()
    shadow = compute_soft_shadow(grid_points, sphere_centers, sphere_radii,
                                  LIGHT_CENTER, LIGHT_HALF_EXTENTS, K, rng)
    t1 = time.perf_counter()
    shadow_img = shadow.reshape(GZ, GX)
    shadow_maps[K] = shadow_img
    if K != K_REFERENCE:
        results["K"].append(K)
        results["time_s"].append(t1 - t0)
    print(f"K={K:5d}  time={t1 - t0:.3f}s")

reference = shadow_maps[K_REFERENCE]
for K in K_VALUES:
    rmse = float(np.sqrt(np.mean((shadow_maps[K] - reference) ** 2)))
    results["rmse_vs_reference"].append(rmse)

with open("/home/claude/m4_shadow_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))


def save_gray(img, path):
    im = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(im, mode="L").save(path)


for K in K_VALUES:
    save_gray(shadow_maps[K], f"/home/claude/m4_shadow_k{K}.png")
save_gray(reference, "/home/claude/m4_shadow_reference.png")

# ----------------------------------------------------------------------------
# Convergence plot: RMSE vs K on log-log axes, with a fitted slope
# ----------------------------------------------------------------------------
K_arr = np.array(results["K"], dtype=float)
rmse_arr = np.array(results["rmse_vs_reference"])
log_k = np.log(K_arr)
log_rmse = np.log(rmse_arr)
slope, intercept = np.polyfit(log_k, log_rmse, 1)

plt.figure(figsize=(6.5, 4.2))
plt.loglog(K_arr, rmse_arr, "o-", label="Measured RMSE vs. reference")
fit_k = np.linspace(K_arr.min(), K_arr.max(), 50)
plt.loglog(fit_k, np.exp(intercept) * fit_k ** slope, "--",
           label=f"Fitted slope = {slope:.2f}")
plt.loglog(fit_k, rmse_arr[0] * (K_arr[0] / fit_k) ** 0.5, ":",
           label="Theoretical O(1/\u221aK) reference")
plt.xlabel("Samples per pixel, K (log scale)")
plt.ylabel("RMSE vs. K=2048 reference (log scale)")
plt.title("Monte Carlo Soft-Shadow Convergence")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/home/claude/m4_shadow_convergence.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5, 4.2))
plt.plot(results["K"], results["time_s"], "o-", color="tab:red")
plt.xlabel("Samples per pixel, K")
plt.ylabel("Compute time (s)")
plt.title("Monte Carlo Shadow Cost vs. Sample Count\n(cost side of the noise/cost trade-off)")
plt.tight_layout()
plt.savefig("/home/claude/m4_shadow_time_vs_k.png", dpi=150)
plt.close()

print(f"\nFitted convergence slope: {slope:.3f} (theoretical Monte Carlo rate: -0.5)")
print("Milestone 4b (Monte Carlo shadows) complete.")
