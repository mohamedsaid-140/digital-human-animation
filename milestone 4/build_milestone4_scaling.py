"""
Milestone 4a — Scalability: Crowd Rendering With / Without Frustum Culling

Scene: many copies of the Milestone 1 character scattered across a wide
world-space area. The camera has a normal (~50 deg) field of view, so only a
fraction of instances are ever actually visible -- exactly the situation
where frustum culling earns its keep.

Two pipelines are timed, at several crowd sizes N:
  BRUTE FORCE:   project + rasterize every instance, regardless of visibility.
  ACCELERATED:   test each instance's world-space AABB against the camera's
                 frustum planes (accel.py) first; project + rasterize only
                 the survivors.

Both pipelines render into the SAME single framebuffer per frame (one
render_scene call), matching how a real frame is actually drawn.
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt

from digital_human_core import Skeleton, SkinnedMesh
from camera import Camera
from renderer import render_scene
from accel import extract_frustum_planes, aabb_intersects_frustum, bind_aabb, translate_aabb

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# Build one character (shared geometry across all instances -- only the
# per-instance world offset differs, exactly like GPU instancing)
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
base_verts = mesh.deform(posed_world) + np.array([0, 0.62, 0])
base_normals = mesh.deform_normals(posed_world)
base_albedo = np.tile(np.array([0.7, 0.5, 0.4]), (len(base_verts), 1))
AABB_MIN, AABB_MAX = bind_aabb(base_verts)

W, H = 800, 500
cam = Camera(eye=[0, 1.3, -3.5], target=[0, 0.7, 6.0], fovy_deg=50,
             aspect=W / H, near=0.1, far=60.0, projection="perspective")
LIGHT_DIR = np.array([0.5, 0.8, -0.3])


def make_instance_offsets(n, area_x=140.0, area_z=(1.0, 60.0)):
    """Scatter n instances over a wide world area (most will fall outside a ~50deg FOV)."""
    offsets = []
    while len(offsets) < n:
        x = rng.uniform(-area_x, area_x)
        z = rng.uniform(*area_z)
        offsets.append([x, 0.0, z])
    return np.array(offsets)


def project_instance(offset):
    verts = base_verts + offset
    ndc, w = cam.project_points(verts)
    screen = np.stack([
        (ndc[:, 0] * 0.5 + 0.5) * W,
        (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H,
    ], axis=-1)
    return verts, ndc, w, screen


def triangles_for_instance(verts, ndc, w, screen):
    tris = []
    for a, b, c in mesh.faces:
        if w[a] <= 1e-4 or w[b] <= 1e-4 or w[c] <= 1e-4:
            continue
        tris.append(dict(
            screen_xy=screen[[a, b, c]], ndc_z=ndc[[a, b, c], 2], clip_w=w[[a, b, c]],
            world_pos=verts[[a, b, c]], normals=base_normals[[a, b, c]],
            albedo_vertex=base_albedo[[a, b, c]],
        ))
    return tris


def render_brute_force(offsets):
    all_tris = []
    for offset in offsets:
        verts, ndc, w, screen = project_instance(offset)
        all_tris.extend(triangles_for_instance(verts, ndc, w, screen))
    img = render_scene(W, H, all_tris, LIGHT_DIR, cam.eye, shading="phong",
                        interpolation="perspective", supersample=1,
                        bg_color=(0.55, 0.65, 0.85))
    return img, len(all_tris)


def render_accelerated(offsets):
    planes = extract_frustum_planes(cam.view_proj_matrix())
    all_tris = []
    n_visible = 0
    for offset in offsets:
        amin, amax = translate_aabb(AABB_MIN, AABB_MAX, offset)
        if not aabb_intersects_frustum(amin, amax, planes):
            continue  # <-- the entire per-vertex + per-triangle cost below is skipped
        n_visible += 1
        verts, ndc, w, screen = project_instance(offset)
        all_tris.extend(triangles_for_instance(verts, ndc, w, screen))
    img = render_scene(W, H, all_tris, LIGHT_DIR, cam.eye, shading="phong",
                        interpolation="perspective", supersample=1,
                        bg_color=(0.55, 0.65, 0.85))
    return img, len(all_tris), n_visible


# ----------------------------------------------------------------------------
# Timing sweep across crowd sizes
# ----------------------------------------------------------------------------
N_VALUES = [10, 25, 50, 100, 150, 250]
results = {"N": [], "brute_time_s": [], "accel_time_s": [], "brute_tris": [],
           "accel_tris": [], "n_visible": [], "speedup": []}

for N in N_VALUES:
    offsets = make_instance_offsets(N)

    t0 = time.perf_counter()
    img_brute, n_tris_brute = render_brute_force(offsets)
    t1 = time.perf_counter()
    brute_time = t1 - t0

    t0 = time.perf_counter()
    img_accel, n_tris_accel, n_visible = render_accelerated(offsets)
    t1 = time.perf_counter()
    accel_time = t1 - t0

    results["N"].append(N)
    results["brute_time_s"].append(brute_time)
    results["accel_time_s"].append(accel_time)
    results["brute_tris"].append(n_tris_brute)
    results["accel_tris"].append(n_tris_accel)
    results["n_visible"].append(n_visible)
    results["speedup"].append(brute_time / accel_time if accel_time > 0 else float("inf"))

    print(f"N={N:4d}  visible={n_visible:4d}  brute={brute_time:.3f}s  "
          f"accel={accel_time:.3f}s  speedup={results['speedup'][-1]:.2f}x")

    if N == 150:  # save a representative pair of images at a mid-large crowd size
        from PIL import Image
        Image.fromarray((np.clip(img_brute, 0, 1) * 255).astype(np.uint8)).save(
            "/home/claude/m4_crowd_render.png")

with open("/home/claude/m4_scaling_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
ax1.plot(results["N"], results["brute_time_s"], "o-", label="Brute force (all instances)")
ax1.plot(results["N"], results["accel_time_s"], "o-", label="Frustum-culled (accelerated)")
ax1.set_xlabel("Total instances in scene (N)")
ax1.set_ylabel("Render time (s)")
ax1.set_title("Render Time vs. Crowd Size\n(single 800\u00d7500 frame)")
ax1.legend()
plt.tight_layout()
plt.savefig("/home/claude/m4_scaling_time.png", dpi=150)
plt.close()

fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
ax1.plot(results["N"], results["speedup"], "o-", color="tab:green")
ax1.set_xlabel("Total instances in scene (N)")
ax1.set_ylabel("Speedup (brute time / accelerated time)")
ax1.set_title("Frustum-Culling Speedup vs. Crowd Size")
plt.tight_layout()
plt.savefig("/home/claude/m4_scaling_speedup.png", dpi=150)
plt.close()

print("\nMilestone 4a (scaling) complete.")
