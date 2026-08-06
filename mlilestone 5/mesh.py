"""
mesh.py
-------
Procedural proxy surface mesh (tapered cylinders per bone), bound to the
skeleton with Linear Blend Skinning (LBS):

    v_skinned = sum_i  w_i * (G_i * B_i^-1) * v_rest

B_i^-1 = inverse bind matrix (rest-pose global transform, inverted),
G_i = current-frame global transform from FK, w_i = skin weight.
"""

import math
from math3d import Vec3, Mat4


def _ring(center: Vec3, axis: Vec3, radius, n=8):
    axis = axis.normalize()
    helper = Vec3(1, 0, 0) if abs(axis.x) < 0.9 else Vec3(0, 1, 0)
    u = axis.cross(helper).normalize()
    v = axis.cross(u)
    pts = []
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        p = center + u * (radius * math.cos(theta)) + v * (radius * math.sin(theta))
        pts.append(p)
    return pts


def build_bone_mesh(skeleton, bones, radius=0.045, rings_per_bone=4, n_sides=8):
    _, rest_globals = skeleton.joint_positions()
    bind_inverse = {}
    for idx, j in enumerate(skeleton.joints):
        bind_inverse[j.name] = rest_globals[idx].inverse_affine()

    vertices = []
    skin = []
    triangles = []

    for parent_name, child_name in bones:
        pi = skeleton.name_to_index[parent_name]
        ci = skeleton.name_to_index[child_name]
        p0 = rest_globals[pi]
        p1 = rest_globals[ci]
        start = Vec3(p0.get(0, 3), p0.get(1, 3), p0.get(2, 3))
        end = Vec3(p1.get(0, 3), p1.get(1, 3), p1.get(2, 3))
        axis = (end - start)
        length = axis.length()
        if length < 1e-6:
            continue
        axis_n = axis.normalize()

        ring_indices = []
        for r in range(rings_per_bone + 1):
            t = r / rings_per_bone
            center = start.lerp(end, t)
            rad = radius * (1.0 - 0.15 * math.sin(math.pi * t))
            ring_pts = _ring(center, axis_n, rad, n_sides)
            idxs = []
            for pt in ring_pts:
                w1 = t
                w0 = 1.0 - t
                vertices.append(pt)
                skin.append((pi, ci, w0, w1))
                idxs.append(len(vertices) - 1)
            ring_indices.append(idxs)

        for r in range(rings_per_bone):
            ring_a = ring_indices[r]
            ring_b = ring_indices[r + 1]
            for k in range(n_sides):
                a0 = ring_a[k]
                a1 = ring_a[(k + 1) % n_sides]
                b0 = ring_b[k]
                b1 = ring_b[(k + 1) % n_sides]
                triangles.append((a0, b0, a1))
                triangles.append((a1, b0, b1))

    return vertices, skin, triangles, bind_inverse


def skin_vertices(vertices, skin, skeleton_globals, bind_inverse, joints):
    out = []
    for v, (i0, i1, w0, w1) in zip(vertices, skin):
        j0, j1 = joints[i0], joints[i1]
        M0 = skeleton_globals[i0] * bind_inverse[j0.name]
        M1 = skeleton_globals[i1] * bind_inverse[j1.name]
        p0 = M0.transform_point(v)
        p1 = M1.transform_point(v)
        out.append(p0 * w0 + p1 * w1)
    return out
