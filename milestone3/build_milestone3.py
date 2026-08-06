"""
Milestone 3 driver.

Scene: the Milestone 1/2 humanoid standing on a large checkerboard floor,
viewed with the Milestone 2 camera pipeline, rendered with the from-scratch
software rasterizer in renderer.py.

Produces three head-to-head comparisons (each isolating ONE variable):

  A) Shading model:       Gouraud (per-vertex lit, interpolated color)
                           vs Phong (per-pixel normal, lit per pixel)
  B) UV interpolation:    affine / screen-space linear (classic distortion bug)
                           vs perspective-correct
  C) Sampling/antialiasing: 1x point sampling vs 4x4 = 16x supersampling

Outputs (all in /home/claude/):
  m3_gouraud.png, m3_phong.png
  m3_affine_uv.png, m3_perspective_uv.png
  m3_pointsampled.png, m3_supersampled.png
  m3_pointsampled_crop.png, m3_supersampled_crop.png   (zoomed silhouette edge)
  m3_affine_crop.png, m3_perspective_crop.png          (zoomed floor tiles)
"""

import numpy as np
from PIL import Image

from digital_human_core import Skeleton, SkinnedMesh, quat_from_axis_angle
from camera import Camera
from renderer import render_scene

# ----------------------------------------------------------------------------
# Build the M1 humanoid, lifted so its feet rest on the y=0 floor plane
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
    mesh.add_bone_tube(parent, child, bind_world, radius=0.06, segments=10)
mesh.bind(bind_world)

skel.set_local_rotation("shoulder_l", quat_from_axis_angle([0, 0, 1], np.radians(55)))
skel.set_local_rotation("elbow_r", quat_from_axis_angle([1, 0, 0], np.radians(-45)))
posed_world = skel.forward_kinematics()

FEET_LIFT = 0.62  # shift so knee/foot region sits near y=0
char_verts = mesh.deform(posed_world) + np.array([0, FEET_LIFT, 0])
char_normals = mesh.deform_normals(posed_world)
char_albedo = np.tile(np.array([[0.75, 0.55, 0.45]]), (len(char_verts), 1))  # skin-tone solid color

# ----------------------------------------------------------------------------
# Checkerboard floor: a large quad receding into the distance
# ----------------------------------------------------------------------------
floor_verts = np.array([
    [-4.0, 0.0, -1.0],
    [4.0, 0.0, -1.0],
    [4.0, 0.0, 20.0],
    [-4.0, 0.0, 20.0],
])
floor_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
floor_normal = np.array([0, 1.0, 0])

# ----------------------------------------------------------------------------
# Camera: low-ish angle so the floor recedes toward a horizon (classic Moire setup)
# ----------------------------------------------------------------------------
cam = Camera(eye=[0.0, 1.1, -3.2], target=[0.0, 0.4, 4.0], fovy_deg=55,
             aspect=800 / 500, near=0.05, far=30.0, projection="perspective")
W, H = 800, 500

LIGHT_DIR = np.array([0.5, 0.8, -0.3])
VIEW_POS = cam.eye


def build_triangles(shading_uv_scale=26.0):
    """Project everything once; per-render calls choose shading/interpolation later."""
    tris = []

    # --- character triangles ---
    ndc_c, w_c = cam.project_points(char_verts)
    screen_c = np.stack([
        (ndc_c[:, 0] * 0.5 + 0.5) * W,
        (1.0 - (ndc_c[:, 1] * 0.5 + 0.5)) * H,
    ], axis=-1)
    for (a, b, c) in mesh.faces:
        if w_c[a] <= 1e-4 or w_c[b] <= 1e-4 or w_c[c] <= 1e-4:
            continue
        tris.append(dict(
            screen_xy=screen_c[[a, b, c]],
            ndc_z=ndc_c[[a, b, c], 2],
            clip_w=w_c[[a, b, c]],
            world_pos=char_verts[[a, b, c]],
            normals=char_normals[[a, b, c]],
            albedo_vertex=char_albedo[[a, b, c]],
        ))

    # --- floor triangles (two, textured) ---
    ndc_f, w_f = cam.project_points(floor_verts)
    screen_f = np.stack([
        (ndc_f[:, 0] * 0.5 + 0.5) * W,
        (1.0 - (ndc_f[:, 1] * 0.5 + 0.5)) * H,
    ], axis=-1)
    floor_tris_idx = [(0, 1, 2), (0, 2, 3)]
    for (a, b, c) in floor_tris_idx:
        if w_f[a] <= 1e-4 or w_f[b] <= 1e-4 or w_f[c] <= 1e-4:
            continue
        tris.append(dict(
            screen_xy=screen_f[[a, b, c]],
            ndc_z=ndc_f[[a, b, c], 2],
            clip_w=w_f[[a, b, c]],
            world_pos=floor_verts[[a, b, c]],
            normals=np.tile(floor_normal, (3, 1)),
            uv=floor_uv[[a, b, c]],
            albedo_vertex=np.tile(np.array([0.5, 0.5, 0.5]), (3, 1)),
            texture_scale=shading_uv_scale,
        ))
    return tris


def save(img, path):
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def crop(img, x0, y0, x1, y1, scale=3):
    c = img[y0:y1, x0:x1]
    im = Image.fromarray((c * 255).astype(np.uint8))
    im = im.resize((c.shape[1] * scale, c.shape[0] * scale), Image.NEAREST)
    return np.array(im) / 255.0


# ----------------------------------------------------------------------------
# A) Shading model comparison: Gouraud vs Phong
#    (use perspective-correct interpolation, 1x sampling, so ONLY shading varies)
# ----------------------------------------------------------------------------
tris = build_triangles()
img_gouraud = render_scene(W, H, tris, LIGHT_DIR, VIEW_POS, shading="gouraud",
                            interpolation="perspective", supersample=1)
save(img_gouraud, "/home/claude/m3_gouraud.png")

img_phong = render_scene(W, H, tris, LIGHT_DIR, VIEW_POS, shading="phong",
                          interpolation="perspective", supersample=1)
save(img_phong, "/home/claude/m3_phong.png")

# ----------------------------------------------------------------------------
# B) UV interpolation comparison: affine vs perspective-correct
#    (use Phong shading, 1x sampling, so ONLY the uv interpolation mode varies)
# ----------------------------------------------------------------------------
img_affine = render_scene(W, H, tris, LIGHT_DIR, VIEW_POS, shading="phong",
                           interpolation="affine", supersample=1)
save(img_affine, "/home/claude/m3_affine_uv.png")

img_persp_uv = render_scene(W, H, tris, LIGHT_DIR, VIEW_POS, shading="phong",
                             interpolation="perspective", supersample=1)
save(img_persp_uv, "/home/claude/m3_perspective_uv.png")

# zoomed crops on the near-field floor tiles where affine distortion is worst
save(crop(img_affine, 250, 260, 550, 420, scale=2), "/home/claude/m3_affine_crop.png")
save(crop(img_persp_uv, 250, 260, 550, 420, scale=2), "/home/claude/m3_perspective_crop.png")

# ----------------------------------------------------------------------------
# C) Sampling / antialiasing comparison: 1x vs 4x4 (16x) supersampling
#    (use Phong shading + perspective-correct interpolation -- the "best"
#     pipeline -- so ONLY the sampling rate varies)
# ----------------------------------------------------------------------------
img_1x = render_scene(W, H, tris, LIGHT_DIR, VIEW_POS, shading="phong",
                       interpolation="perspective", supersample=1)
save(img_1x, "/home/claude/m3_pointsampled.png")

img_4x = render_scene(W, H, tris, LIGHT_DIR, VIEW_POS, shading="phong",
                       interpolation="perspective", supersample=4)
save(img_4x, "/home/claude/m3_supersampled.png")

# zoomed crops: character shoulder silhouette (jaggies) and distant floor (moire)
save(crop(img_1x, 330, 150, 470, 260, scale=3), "/home/claude/m3_pointsampled_crop.png")
save(crop(img_4x, 330, 150, 470, 260, scale=3), "/home/claude/m3_supersampled_crop.png")

print("Milestone 3 renders complete.")
print("shapes:", img_gouraud.shape, img_phong.shape, img_1x.shape, img_4x.shape)
