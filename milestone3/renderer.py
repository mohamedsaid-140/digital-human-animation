"""
Milestone 3 — Rendering & Signal Processing
A small software rasterizer built from scratch on top of the Milestone 1/2
pipeline (Skeleton/LBS -> Camera/MVP). No OpenGL, no mplot3d -- every pixel
in every output image is written by the code in this file.

Pipeline stage this milestone adds:

    Screen-space triangles (from Milestone 2's camera.project_points)
        --(rasterize: edge functions + bounding box)-->  covered pixels
        --(depth test against a z-buffer)--------------->  visible pixels
        --(attribute interpolation: affine OR perspective-correct)
        --(shading: Gouraud OR Phong/Blinn-Phong)-------->  per-pixel color
        --(optional supersampling + box downsample)------>  antialiased image
"""

import numpy as np


# ----------------------------------------------------------------------------
# 1. LIGHTING: BLINN-PHONG
# ----------------------------------------------------------------------------

def blinn_phong(N, view_dir, light_dir, albedo, light_color=np.array([1.0, 1.0, 1.0]),
                 ambient=0.15, shininess=32.0, spec_strength=0.6):
    """
    Per-point (or per-vertex) shading:
        color = ambient*albedo + diffuse*(N.L)*albedo*light + specular*(N.H)^shininess*light
    N, view_dir, light_dir are expected unit vectors (or arrays of them, Nx3).
    """
    N = N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-8)
    L = light_dir / np.linalg.norm(light_dir)
    V = view_dir / np.maximum(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-8)
    H = (L + V)
    H = H / np.maximum(np.linalg.norm(H, axis=-1, keepdims=True), 1e-8)

    ndotl = np.clip(np.sum(N * L, axis=-1), 0.0, 1.0)
    ndoth = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0)
    spec = spec_strength * (ndoth ** shininess)

    diffuse = ndotl[..., None] * albedo * light_color
    ambient_term = ambient * albedo
    specular = spec[..., None] * light_color
    return np.clip(ambient_term + diffuse + specular, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 2. PROCEDURAL CHECKER TEXTURE
# ----------------------------------------------------------------------------

def checker_albedo(u, v, scale=8.0, color_a=np.array([0.85, 0.85, 0.9]),
                    color_b=np.array([0.1, 0.1, 0.15])):
    """Point-sampled (nearest) checkerboard lookup -- deliberately NOT
    band-limited, so it aliases/Moires under minification exactly like a
    naive texture sampler would."""
    parity = (np.floor(u * scale).astype(np.int64) + np.floor(v * scale).astype(np.int64)) % 2
    parity = parity[..., None]
    return np.where(parity == 0, color_a, color_b)


# ----------------------------------------------------------------------------
# 3. RASTERIZER
# ----------------------------------------------------------------------------

def edge_function(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


class Framebuffer:
    def __init__(self, width, height, bg_color=(1.0, 1.0, 1.0)):
        self.width = width
        self.height = height
        self.color = np.ones((height, width, 3)) * np.array(bg_color)
        self.depth = np.full((height, width), np.inf)


def rasterize_triangle(fb: Framebuffer, screen_xy, ndc_z, clip_w, world_pos, normals,
                        uv, albedo_vertex, light_dir, view_pos, shading="phong",
                        interpolation="perspective", texture_scale=None):
    """
    screen_xy   : (3,2) pixel coordinates of the 3 vertices
    ndc_z       : (3,) NDC depth (smaller = closer, standard convention here)
    clip_w      : (3,) clip-space w (equal to camera-space distance, > 0 in front of camera)
    world_pos   : (3,3) world-space position per vertex (for Phong lighting / view dir)
    normals     : (3,3) world-space normal per vertex
    uv          : (3,2) or None -- texture coordinates per vertex
    albedo_vertex: (3,3) solid per-vertex albedo (used directly if uv is None)
    shading     : "gouraud" or "phong"
    interpolation: "affine" (screen-space linear, wrong) or "perspective" (correct)
    texture_scale: if set, uv is looked up through checker_albedo at this frequency
    """
    (x0, y0), (x1, y1), (x2, y2) = screen_xy
    minx = max(int(np.floor(min(x0, x1, x2))), 0)
    maxx = min(int(np.ceil(max(x0, x1, x2))), fb.width - 1)
    miny = max(int(np.floor(min(y0, y1, y2))), 0)
    maxy = min(int(np.ceil(max(y0, y1, y2))), fb.height - 1)
    if minx > maxx or miny > maxy:
        return

    area = edge_function(x0, y0, x1, y1, x2, y2)
    if abs(area) < 1e-9:
        return

    xs = np.arange(minx, maxx + 1) + 0.5
    ys = np.arange(miny, maxy + 1) + 0.5
    PX, PY = np.meshgrid(xs, ys)

    w0 = edge_function(x1, y1, x2, y2, PX, PY)
    w1 = edge_function(x2, y2, x0, y0, PX, PY)
    w2 = edge_function(x0, y0, x1, y1, PX, PY)

    if area > 0:
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    else:
        mask = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
    if not mask.any():
        return

    b0, b1, b2 = w0 / area, w1 / area, w2 / area  # affine (screen-space) barycentric weights

    depth = b0 * ndc_z[0] + b1 * ndc_z[1] + b2 * ndc_z[2]
    sub_depth = fb.depth[miny:maxy + 1, minx:maxx + 1]
    visible = mask & (depth < sub_depth)
    if not visible.any():
        return

    if interpolation == "perspective":
        iw0, iw1, iw2 = 1.0 / clip_w[0], 1.0 / clip_w[1], 1.0 / clip_w[2]
        denom = b0 * iw0 + b1 * iw1 + b2 * iw2
        pb0 = b0 * iw0 / denom
        pb1 = b1 * iw1 / denom
        pb2 = b2 * iw2 / denom
    else:  # affine: use screen-space weights directly (the classic PS1-era distortion bug)
        pb0, pb1, pb2 = b0, b1, b2

    # interpolate world position (for Phong lighting & view direction) — always
    # perspective-correct for the position/normal regardless of the
    # `interpolation` mode being tested (that flag isolates the *texture UV*
    # distortion specifically, which is the classic teaching example).
    iw0, iw1, iw2 = 1.0 / clip_w[0], 1.0 / clip_w[1], 1.0 / clip_w[2]
    denom_pc = b0 * iw0 + b1 * iw1 + b2 * iw2
    pcb0, pcb1, pcb2 = b0 * iw0 / denom_pc, b1 * iw1 / denom_pc, b2 * iw2 / denom_pc

    Wp = (pcb0[..., None] * world_pos[0] + pcb1[..., None] * world_pos[1] + pcb2[..., None] * world_pos[2])
    Np = (pcb0[..., None] * normals[0] + pcb1[..., None] * normals[1] + pcb2[..., None] * normals[2])
    view_dir = view_pos - Wp

    if uv is not None:
        UVp = (pb0[..., None] * uv[0] + pb1[..., None] * uv[1] + pb2[..., None] * uv[2])
        albedo = checker_albedo(UVp[..., 0], UVp[..., 1], scale=texture_scale)
    else:
        albedo = (pcb0[..., None] * albedo_vertex[0] + pcb1[..., None] * albedo_vertex[1]
                  + pcb2[..., None] * albedo_vertex[2])

    if shading == "phong":
        color = blinn_phong(Np, view_dir, light_dir, albedo)
    else:  # gouraud: light at the 3 vertices, then interpolate the RESULT (not the normal)
        vcolors = []
        for i in range(3):
            c = blinn_phong(normals[i], view_pos - world_pos[i], light_dir, albedo_vertex[i])
            vcolors.append(c)
        color = pcb0[..., None] * vcolors[0] + pcb1[..., None] * vcolors[1] + pcb2[..., None] * vcolors[2]

    idx = np.where(visible)
    fb.color[miny + idx[0], minx + idx[1]] = color[idx]
    fb.depth[miny + idx[0], minx + idx[1]] = depth[idx]


def render_scene(width, height, triangles, light_dir, view_pos, shading="phong",
                  interpolation="perspective", supersample=1, bg_color=(0.55, 0.65, 0.85)):
    """
    triangles: list of dicts, each with keys screen_xy, ndc_z, clip_w, world_pos,
               normals, uv (optional), albedo_vertex, texture_scale (optional).
    supersample: render at supersample x resolution, then box-downsample --
                 the antialiasing method under test in Section 3 of the report.
    """
    W, H = width * supersample, height * supersample
    fb = Framebuffer(W, H, bg_color=bg_color)
    for tri in triangles:
        sxy = tri["screen_xy"] * supersample
        rasterize_triangle(
            fb, sxy, tri["ndc_z"], tri["clip_w"], tri["world_pos"], tri["normals"],
            tri.get("uv"), tri["albedo_vertex"], light_dir, view_pos,
            shading=shading, interpolation=interpolation,
            texture_scale=tri.get("texture_scale"),
        )
    img = fb.color
    if supersample > 1:
        img = img.reshape(height, supersample, width, supersample, 3).mean(axis=(1, 3))
    return np.clip(img, 0.0, 1.0)
