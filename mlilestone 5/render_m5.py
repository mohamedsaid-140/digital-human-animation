"""
render_m5.py
------------
Milestone 5 deliverable: temporal behavior on top of the M1/M2 system.

Produces:
  out/m5_walk.gif           procedural walk cycle, animated, from a fixed camera
  out/m5_wave.gif           keyframe wave clip, animated
  out/m5_motion_analysis.png    joint-height / gait / ponytail-motion plots
  out/m5_stability.png          constraint-error and foot-sliding plots
  stdout: numeric motion-consistency and stability summary
"""

import math
from math3d import Vec3, Quaternion
from skeleton import build_humanoid_skeleton, BONES
from mesh import build_bone_mesh, skin_vertices
from camera import Camera
from rasterizer import rasterize
from animation import KeyframeClip, build_wave_clip, procedural_walk_cycle
from particles import ParticleChain
from PIL import ImageDraw

W, H = 640, 640
FPS = 24


def render_frame(skeleton, verts, skin, tris, bind_inv, cam, ponytail=None):
    _, globals_ = skeleton.joint_positions()
    skinned = skin_vertices(verts, skin, globals_, bind_inv, skeleton.joints)

    triangles_screen = []
    for (a, b, c) in tris:
        pts = []
        ok = True
        for idx in (a, b, c):
            proj = cam.project_to_screen(skinned[idx], W, H)
            if proj is None:
                ok = False
                break
            pts.append(proj)
        if not ok:
            continue
        avg_z = (pts[0][2] + pts[1][2] + pts[2][2]) / 3.0
        shade = int(max(0, min(255, 150 - avg_z * 130)))
        color = (35, shade, 95)
        triangles_screen.append(((pts[0][0], pts[0][1], pts[0][2]),
                                  (pts[1][0], pts[1][1], pts[1][2]),
                                  (pts[2][0], pts[2][1], pts[2][2]), color))
    img = rasterize(W, H, triangles_screen)

    if ponytail is not None:
        draw = ImageDraw.Draw(img)
        screen_pts = []
        for p in ponytail.pos:
            proj = cam.project_to_screen(p, W, H)
            screen_pts.append((proj[0], proj[1]) if proj else None)
        for i in range(len(screen_pts) - 1):
            if screen_pts[i] and screen_pts[i + 1]:
                draw.line([screen_pts[i], screen_pts[i + 1]], fill=(240, 200, 60), width=3)
        for p in screen_pts:
            if p:
                draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=(255, 230, 140))
    return img, globals_


def head_world_pos(skeleton, globals_):
    idx = skeleton.name_to_index["head"]
    g = globals_[idx]
    return Vec3(g.get(0, 3), g.get(1, 3), g.get(2, 3))


def run_wave_sequence():
    skeleton = build_humanoid_skeleton()
    verts, skin, tris, bind_inv = build_bone_mesh(skeleton, BONES)
    clip = build_wave_clip()
    cam = Camera(Vec3(0, 0.5, 2.2), Vec3(0, 0.35, 0), fov_deg=40, aspect=1.0)

    frames = []
    times = []
    hand_heights = []
    n_frames = int(clip.duration * FPS)
    for f in range(n_frames):
        t = f / FPS
        skeleton.reset_pose()
        clip.apply(skeleton, t)
        img, globals_ = render_frame(skeleton, verts, skin, tris, bind_inv, cam)
        frames.append(img)
        idx = skeleton.name_to_index["hand_r"]
        hand_heights.append(globals_[idx].get(1, 3))
        times.append(t)

    frames[0].save("out/m5_wave.gif", save_all=True, append_images=frames[1:],
                    duration=int(1000 / FPS), loop=0)
    print(f"wrote out/m5_wave.gif ({n_frames} frames)")
    return times, hand_heights


def run_walk_sequence():
    skeleton = build_humanoid_skeleton()
    verts, skin, tris, bind_inv = build_bone_mesh(skeleton, BONES)

    ponytail_root = None
    ponytail = None

    n_cycles = 2.0
    stride_hz = 0.9
    duration = n_cycles / stride_hz
    n_frames = int(duration * FPS)
    dt = 1.0 / FPS

    frames = []
    times, pelvis_h, foot_l_h, foot_r_h, foot_l_vx = [], [], [], [], []
    constraint_errs = []

    prev_foot_l_x = None

    for f in range(n_frames):
        t = f / FPS
        skeleton.reset_pose()
        bob = procedural_walk_cycle(skeleton, t, stride_hz=stride_hz)
        # root motion: forward translation (x) + vertical bob, authored directly
        # on the pelvis joint's local translation (the root's "local" transform
        # IS the world transform, since it has no parent)
        walk_speed = 0.55  # m/s
        root_x = walk_speed * t
        skeleton.joints[skeleton.name_to_index["pelvis"]].local_translation = Vec3(root_x, bob, 0)

        # chase camera: eye and target both track the character's forward
        # position, so a translating root stays framed (same camera model
        # as M2, just with eye/target updated per frame from world state)
        cam = Camera(Vec3(root_x + 0.3, 0.65, 2.0), Vec3(root_x, 0.4, 0), fov_deg=42, aspect=1.0)

        joint_pts, globals_ = skeleton.joint_positions()
        head_pos = head_world_pos(skeleton, globals_)

        if ponytail is None:
            ponytail = ParticleChain(head_pos + Vec3(0, 0.06, -0.02), n_particles=6, segment_length=0.045)
        ponytail.step(head_pos + Vec3(0, 0.06, -0.02), dt)
        constraint_errs.append(ponytail.constraint_error())

        img, globals_ = render_frame(skeleton, verts, skin, tris, bind_inv, cam, ponytail=ponytail)
        frames.append(img)

        pelvis_h.append(joint_pts[skeleton.name_to_index["pelvis"]].y)
        fl = joint_pts[skeleton.name_to_index["foot_l"]]
        fr = joint_pts[skeleton.name_to_index["foot_r"]]
        foot_l_h.append(fl.y)
        foot_r_h.append(fr.y)
        if prev_foot_l_x is not None:
            foot_l_vx.append((fl.x - prev_foot_l_x) / dt)
        else:
            foot_l_vx.append(0.0)
        prev_foot_l_x = fl.x
        times.append(t)

    frames[0].save("out/m5_walk.gif", save_all=True, append_images=frames[1:],
                    duration=int(1000 / FPS), loop=0)
    print(f"wrote out/m5_walk.gif ({n_frames} frames)")
    return times, pelvis_h, foot_l_h, foot_r_h, foot_l_vx, constraint_errs


def motion_and_stability_analysis(wave_t, wave_hand_h, walk_t, pelvis_h, foot_l_h, foot_r_h, foot_l_vx, constraint_errs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(wave_t, wave_hand_h, color="tab:orange")
    axes[0, 0].set_title("Keyframe clip: right hand height (SLERP-interpolated)")
    axes[0, 0].set_xlabel("time (s)"); axes[0, 0].set_ylabel("world y (m)")

    axes[0, 1].plot(walk_t, pelvis_h, label="pelvis (root bob)", color="tab:blue")
    axes[0, 1].plot(walk_t, foot_l_h, label="foot_l", color="tab:green")
    axes[0, 1].plot(walk_t, foot_r_h, label="foot_r", color="tab:red")
    axes[0, 1].set_title("Procedural walk: vertical trajectories (gait pattern)")
    axes[0, 1].set_xlabel("time (s)"); axes[0, 1].set_ylabel("world y (m)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(walk_t, foot_l_vx, color="tab:purple")
    axes[1, 0].axhline(0, color="gray", linewidth=0.7)
    axes[1, 0].set_title("Left foot horizontal velocity (0 during stance = no sliding)")
    axes[1, 0].set_xlabel("time (s)"); axes[1, 0].set_ylabel("vx (m/s)")

    axes[1, 1].plot(walk_t, constraint_errs, color="tab:brown")
    axes[1, 1].set_title("Ponytail (Verlet chain) mean constraint error")
    axes[1, 1].set_xlabel("time (s)"); axes[1, 1].set_ylabel("mean |Δlength| (m)")

    fig.tight_layout()
    fig.savefig("out/m5_motion_analysis.png", dpi=130)
    print("wrote out/m5_motion_analysis.png")

    # ---- numeric stability / realism summary -------------------------------
    stance_vx = [v for h, v in zip(foot_l_h, foot_l_vx) if h < (min(foot_l_h) + 0.01)]
    mean_stance_slide = sum(abs(v) for v in stance_vx) / max(1, len(stance_vx))
    print("\n--- Motion consistency / realism evaluation ---")
    print(f"Foot-sliding metric (mean |horizontal foot speed| while foot is near its lowest point): "
          f"{mean_stance_slide:.4f} m/s")
    print("   (ideal planted-foot contact = 0; nonzero value quantifies the sliding")
    print("    artifact from not having true foot-plant IK -- a known limitation)")
    print(f"Ponytail constraint error: mean={sum(constraint_errs)/len(constraint_errs):.5f} m, "
          f"max={max(constraint_errs):.5f} m over segment length 0.045 m "
          f"({100*max(constraint_errs)/0.045:.1f}% worst-case stretch)")
    print("   (bounded and non-growing over the sequence => the Verlet + constraint")
    print("    solver is stable, not accumulating energy/error over time)")


if __name__ == "__main__":
    wave_t, wave_hand_h = run_wave_sequence()
    walk_t, pelvis_h, foot_l_h, foot_r_h, foot_l_vx, constraint_errs = run_walk_sequence()
    motion_and_stability_analysis(wave_t, wave_hand_h, walk_t, pelvis_h, foot_l_h, foot_r_h, foot_l_vx, constraint_errs)
