"""
Milestone 4b — Monte Carlo Soft Shadows

An area light (not a point light) produces soft shadows with a penumbra --
computing that correctly requires integrating visibility over the light's
entire surface, which has no closed form for an arbitrary occluder. Monte
Carlo integration approximates it: draw K random points on the light,
average the fraction that are unoccluded.

To keep this efficient in pure NumPy (no ray-mesh intersection), the
character is approximated as a small set of spheres strung along each bone
(a standard real-time "proxy geometry" trick) and shadow rays are tested
against those spheres analytically.
"""

import numpy as np


def build_sphere_proxy(posed_world, skel, bones, radius=0.09, offset=None, samples_per_bone=3):
    """
    Approximate the skinned character as a handful of spheres along each
    bone -- enough to cast a recognizable soft shadow without needing
    triangle-mesh ray intersection.
    """
    if offset is None:
        offset = np.zeros(3)
    centers = []
    for parent, child in bones:
        pi = skel.name_to_index[parent]
        ci = skel.name_to_index[child]
        p0 = posed_world[pi][:3, 3] + offset
        p1 = posed_world[ci][:3, 3] + offset
        for k in range(samples_per_bone):
            t = (k + 0.5) / samples_per_bone
            centers.append(p0 * (1 - t) + p1 * t)
    centers = np.array(centers)
    radii = np.full(len(centers), radius)
    return centers, radii


def compute_soft_shadow(grid_points, sphere_centers, sphere_radii, light_center,
                         light_half_extents, n_samples, rng, chunk_size=64):
    """
    grid_points: (Ng,3) floor-level sample points to compute a shadow value for.
    Returns shadow_value (Ng,), 1.0 = fully lit, 0.0 = fully occluded.

    Uses INDEPENDENT random light samples per grid point (proper Monte Carlo --
    not a single shared sample set, which would correlate the noise between
    neighboring pixels and look like banding rather than genuine grain).

    Samples are processed in chunks of `chunk_size` so peak memory is
    O(Ng * chunk_size) rather than O(Ng * n_samples) -- needed once K grows
    into the thousands for a high-sample reference image.
    """
    Ng = len(grid_points)
    ox = grid_points[:, 0:1]
    oy = grid_points[:, 1:2]
    oz = grid_points[:, 2:3]

    occluded_count = np.zeros(Ng, dtype=np.int64)
    remaining = n_samples
    while remaining > 0:
        m = min(chunk_size, remaining)
        remaining -= m

        u = (rng.random((Ng, m)) * 2 - 1) * light_half_extents[0]
        v = (rng.random((Ng, m)) * 2 - 1) * light_half_extents[1]
        light_x = light_center[0] + u
        light_y = np.full((Ng, m), light_center[1])
        light_z = light_center[2] + v

        dx = light_x - ox
        dy = light_y - oy
        dz = light_z - oz

        occluded = np.zeros((Ng, m), dtype=bool)
        for c, r in zip(sphere_centers, sphere_radii):
            fx = ox - c[0]
            fy = oy - c[1]
            fz = oz - c[2]
            a = dx * dx + dy * dy + dz * dz
            b = 2 * (fx * dx + fy * dy + fz * dz)
            cc = fx * fx + fy * fy + fz * fz - r * r
            disc = b * b - 4 * a * cc
            hit = disc >= 0
            sqrt_disc = np.sqrt(np.maximum(disc, 0))
            denom = 2 * a + 1e-12
            t1 = (-b - sqrt_disc) / denom
            t2 = (-b + sqrt_disc) / denom
            tlo = np.minimum(t1, t2)
            thi = np.maximum(t1, t2)
            in_range = hit & (thi > 1e-3) & (tlo < 1.0 - 1e-3)
            occluded |= in_range

        occluded_count += occluded.sum(axis=1)

    return 1.0 - occluded_count / n_samples
