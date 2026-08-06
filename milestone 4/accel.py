"""
Milestone 4 — Acceleration Structure: View-Frustum Culling via Bounding Volumes

The idea: projecting and rasterizing every triangle of every character in a
scene is O(total triangles), regardless of how many are actually visible.
A cheap, O(1)-per-instance bounding-volume test lets the renderer SKIP all
per-vertex and per-triangle work for anything outside the camera's view
frustum, turning the real cost into O(visible triangles) instead.

This file implements the classic Gribb/Hartmann method: the 6 frustum planes
(left, right, bottom, top, near, far) are extracted directly from the
combined view-projection matrix (no separate frustum geometry needs to be
built), and an axis-aligned bounding box (AABB) is tested against all 6 with
the standard "positive vertex" trick.
"""

import numpy as np


def extract_frustum_planes(view_proj):
    """
    Given a 4x4 view-projection matrix M (clip = M @ [x,y,z,1]), extract the
    6 clip-space planes such that a point is inside the frustum iff
    plane . [x,y,z,1] >= 0 for all 6 planes.

    Derivation: clip space considers a point inside iff -w <= x,y,z <= w.
    Each of those 6 inequalities, written as (row_w +/- row_axis).p >= 0,
    is a plane equation in whatever space p is expressed in (world space,
    since M includes both view and projection) — this is what makes the
    method convenient: no intermediate camera-space transform is needed.
    """
    r0, r1, r2, r3 = view_proj[0, :], view_proj[1, :], view_proj[2, :], view_proj[3, :]
    planes = [
        r3 + r0,  # left
        r3 - r0,  # right
        r3 + r1,  # bottom
        r3 - r1,  # top
        r3 + r2,  # near
        r3 - r2,  # far
    ]
    normed = []
    for p in planes:
        n = np.linalg.norm(p[:3])
        normed.append(p / n if n > 1e-12 else p)
    return normed


def aabb_intersects_frustum(aabb_min, aabb_max, planes):
    """
    Standard "positive vertex" AABB-frustum test: for each plane, pick the
    AABB corner that is furthest in the plane's positive-normal direction
    (the 'positive vertex' / p-vertex). If even that corner is outside one
    plane, the entire box is outside the frustum -- this is a conservative
    (never wrongly culls a partially-visible box) O(1) test per instance.
    """
    for p in planes:
        px = aabb_max[0] if p[0] >= 0 else aabb_min[0]
        py = aabb_max[1] if p[1] >= 0 else aabb_min[1]
        pz = aabb_max[2] if p[2] >= 0 else aabb_min[2]
        if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0:
            return False  # entirely outside this plane -> outside frustum
    return True


def bind_aabb(vertices):
    """Axis-aligned bounding box of a vertex array, in the mesh's own local space."""
    return vertices.min(axis=0), vertices.max(axis=0)


def translate_aabb(aabb_min, aabb_max, offset):
    return aabb_min + offset, aabb_max + offset
