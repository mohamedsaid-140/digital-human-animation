"""
Milestone 1 prototype driver.

Builds a minimal (13-joint) humanoid skeleton, generates a tube mesh bound
to it via Linear Blend Skinning, then produces:
  1. bind_pose.png       - the T-pose skeleton + skinned mesh
  2. posed.png           - skeleton + mesh after applying joint rotations
  3. animation.gif       - smooth quaternion-slerp interpolation bind -> pose
as the "basic functional output mechanism" required by Milestone 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation

from digital_human_core import (
    Skeleton, SkinnedMesh, quat_from_axis_angle, quat_identity, quat_slerp
)

# ----------------------------------------------------------------------------
# Build a minimal humanoid skeleton (13 joints)
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

# ----------------------------------------------------------------------------
# Bind pose (T-pose) world transforms + skinned mesh
# ----------------------------------------------------------------------------
bind_world = skel.bind_pose_world()

mesh = SkinnedMesh(skel)
for parent, child in BONES:
    mesh.add_bone_tube(parent, child, bind_world, radius=0.035, segments=8)
mesh.bind(bind_world)


def skeleton_lines(world):
    lines = []
    for parent, child in BONES:
        pi = skel.name_to_index[parent]
        ci = skel.name_to_index[child]
        p0 = world[pi][:3, 3]
        p1 = world[ci][:3, 3]
        lines.append((p0, p1))
    return lines


def render_frame(ax, world, verts, title):
    ax.clear()
    # skinned mesh as a translucent wireframe
    for (a, b, c) in mesh.faces[::3]:  # thin the wireframe for clarity
        tri = verts[[a, b, c]]
        ax.plot(tri[:, 0], tri[:, 2], tri[:, 1], color="steelblue", alpha=0.25, linewidth=0.5)

    # skeleton as bold bones + joints
    for p0, p1 in skeleton_lines(world):
        ax.plot([p0[0], p1[0]], [p0[2], p1[2]], [p0[1], p1[1]],
                color="firebrick", linewidth=2.5, marker="o", markersize=3)

    ax.set_title(title)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.9, 0.9)
    ax.set_box_aspect([1, 1, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")


# ----------------------------------------------------------------------------
# 1) Bind pose render
# ----------------------------------------------------------------------------
bind_verts = mesh.deform(bind_world)
fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111, projection="3d")
render_frame(ax, bind_world, bind_verts, "Bind Pose (T-Pose)\nSkeleton + LBS-Skinned Mesh")
plt.tight_layout()
plt.savefig("/home/claude/bind_pose.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 2) Posed frame: raise left arm, bend right elbow, bend left knee
# ----------------------------------------------------------------------------
target_rotations = {
    "shoulder_l": quat_from_axis_angle([0, 0, 1], np.radians(100)),
    "elbow_l": quat_from_axis_angle([0, 1, 0], np.radians(30)),
    "shoulder_r": quat_from_axis_angle([0, 0, 1], np.radians(-20)),
    "elbow_r": quat_from_axis_angle([1, 0, 0], np.radians(-70)),
    "knee_l": quat_from_axis_angle([1, 0, 0], np.radians(45)),
    "spine": quat_from_axis_angle([0, 1, 0], np.radians(15)),
}
for name, q in target_rotations.items():
    skel.set_local_rotation(name, q)

posed_world = skel.forward_kinematics()
posed_verts = mesh.deform(posed_world)

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111, projection="3d")
render_frame(ax, posed_world, posed_verts, "Posed Frame\n(Forward Kinematics + LBS)")
plt.tight_layout()
plt.savefig("/home/claude/posed.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Animation: slerp every joint's rotation from bind -> target over N frames
# ----------------------------------------------------------------------------
N_FRAMES = 24
fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111, projection="3d")

joint_names = list(target_rotations.keys())
identity = quat_identity()


def animate(frame_idx):
    t = frame_idx / (N_FRAMES - 1)
    for name in joint_names:
        q = quat_slerp(identity, target_rotations[name], t)
        skel.set_local_rotation(name, q)
    world = skel.forward_kinematics()
    verts = mesh.deform(world)
    render_frame(ax, world, verts, f"FK + LBS Animation  (t={t:.2f})")
    return ax,


anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=80)
anim.save("/home/claude/animation.gif", writer="pillow", fps=12)
plt.close(fig)

print("Done. Outputs: bind_pose.png, posed.png, animation.gif")
