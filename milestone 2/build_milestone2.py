"""
Milestone 2 driver.

Reuses the Milestone 1 skeleton + LBS mesh, then pushes every vertex/joint
through a from-scratch Model -> View -> Projection -> NDC -> Screen pipeline
(camera.py) and rasterizes with plain 2D line drawing (matplotlib 2D axes,
NOT mplot3d) — proving spatial correctness comes from our own matrices, not
a plotting library's built-in 3D handling.

Outputs:
  m2_perspective_view.png   - character seen through a perspective camera
  m2_orthographic_view.png  - same scene, orthographic projection (comparison)
  m2_three_views.png        - front / 3-quarter / top, same MVP code path
  m2_turntable.gif          - camera orbits the character (view changes,
                              object stays fixed - the key spatial-correctness
                              test that M1's object-space animation didn't cover)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from digital_human_core import Skeleton, SkinnedMesh, quat_from_axis_angle
from camera import Camera, viewport_transform

# ----------------------------------------------------------------------------
# Rebuild the Milestone 1 humanoid (same definition, so M2 is a strict extension)
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
    mesh.add_bone_tube(parent, child, bind_world, radius=0.035, segments=8)
mesh.bind(bind_world)

# a mild pose so the figure isn't a flat T-pose for these spatial demos
skel.set_local_rotation("shoulder_l", quat_from_axis_angle([0, 0, 1], np.radians(60)))
skel.set_local_rotation("elbow_r", quat_from_axis_angle([1, 0, 0], np.radians(-50)))
skel.set_local_rotation("knee_l", quat_from_axis_angle([1, 0, 0], np.radians(25)))
posed_world = skel.forward_kinematics()
verts = mesh.deform(posed_world)


def bone_world_endpoints(world):
    lines = []
    for parent, child in BONES:
        pi = skel.name_to_index[parent]
        ci = skel.name_to_index[child]
        lines.append((world[pi][:3, 3], world[ci][:3, 3]))
    return lines


BONE_LINES = bone_world_endpoints(posed_world)


def render_view(ax, camera, title):
    ax.clear()
    W, H = 640, 720

    # skinned mesh: project every vertex, draw thinned wireframe
    ndc, w_clip = camera.project_points(verts)
    behind = w_clip <= 1e-4
    screen = viewport_transform(ndc[:, :2], W, H)
    for (a, b, c) in mesh.faces[::4]:
        if behind[a] or behind[b] or behind[c]:
            continue
        tri = screen[[a, b, c, a]]
        ax.plot(tri[:, 0], tri[:, 1], color="steelblue", alpha=0.35, linewidth=0.6)

    # skeleton bones, same MVP pipeline, drawn bold
    for p0, p1 in BONE_LINES:
        pts = np.array([p0, p1])
        ndc_b, w_b = camera.project_points(pts)
        if np.any(w_b <= 1e-4):
            continue
        screen_b = viewport_transform(ndc_b[:, :2], W, H)
        ax.plot(screen_b[:, 0], screen_b[:, 1], color="firebrick",
                linewidth=2.5, marker="o", markersize=3)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # image coordinates: y grows downward
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


# ----------------------------------------------------------------------------
# 1) Perspective vs orthographic, single view each
# ----------------------------------------------------------------------------
persp_cam = Camera(eye=[1.2, 0.9, 1.6], target=[0, 0.25, 0], fovy_deg=40,
                    aspect=640/720, near=0.1, far=10.0, projection="perspective")
fig, ax = plt.subplots(figsize=(5, 5.6))
render_view(ax, persp_cam, "Perspective Camera\n(from-scratch view + projection matrices)")
plt.tight_layout()
plt.savefig("/home/claude/m2_perspective_view.png", dpi=150)
plt.close(fig)

ortho_cam = Camera(eye=[1.2, 0.9, 1.6], target=[0, 0.25, 0], fovy_deg=40,
                    aspect=640/720, near=0.1, far=10.0, projection="orthographic")
fig, ax = plt.subplots(figsize=(5, 5.6))
render_view(ax, ortho_cam, "Orthographic Camera\n(same scene, parallel projection)")
plt.tight_layout()
plt.savefig("/home/claude/m2_orthographic_view.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 2) Three canonical views through the SAME pipeline code path
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
cams = {
    "Front": Camera(eye=[0, 0.35, 2.2], target=[0, 0.25, 0], fovy_deg=35, aspect=1, near=0.1, far=10),
    "Three-Quarter": Camera(eye=[1.4, 0.9, 1.6], target=[0, 0.25, 0], fovy_deg=40, aspect=1, near=0.1, far=10),
    "Top": Camera(eye=[0, 2.4, 0.01], target=[0, 0.25, 0], up=(0, 0, -1), fovy_deg=45, aspect=1, near=0.1, far=10),
}
for ax, (label, cam) in zip(axes, cams.items()):
    render_view(ax, cam, label)
plt.tight_layout()
plt.savefig("/home/claude/m2_three_views.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Turntable: orbit the CAMERA around the fixed character
#    (M1's animation moved joints in object space; this moves the observer -
#    the actual test of "spatial correctness" for a camera system.)
# ----------------------------------------------------------------------------
N_FRAMES = 36
fig, ax = plt.subplots(figsize=(5, 5.6))


def animate(i):
    theta = 2 * np.pi * i / N_FRAMES
    radius = 1.9
    eye = [radius * np.sin(theta), 0.9, radius * np.cos(theta)]
    cam = Camera(eye=eye, target=[0, 0.25, 0], fovy_deg=40, aspect=640/720,
                 near=0.1, far=10.0, projection="perspective")
    render_view(ax, cam, f"Turntable  (camera orbit \u03b8={np.degrees(theta):.0f}\u00b0)")
    return ax,


anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=80)
anim.save("/home/claude/m2_turntable.gif", writer="pillow", fps=14)
plt.close(fig)

print("Milestone 2 renders complete.")
