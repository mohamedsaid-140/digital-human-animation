"""
camera.py
---------
Camera model, separate from the model/world transform.

Pipeline per vertex:
    clip   = Projection * View * v_world
    ndc    = clip.xyz / clip.w           (perspective divide)
    screen = viewport(ndc)               (NDC [-1,1] -> pixels)
"""

import math
from math3d import Vec3, Mat4, Vec4


class Camera:
    def __init__(self, eye: Vec3, target: Vec3, up=None,
                 fov_deg=45.0, aspect=1.0, near=0.05, far=50.0,
                 projection="perspective"):
        self.eye = eye
        self.target = target
        self.up = up if up is not None else Vec3(0, 1, 0)
        self.fov_deg = fov_deg
        self.aspect = aspect
        self.near = near
        self.far = far
        self.projection_mode = projection

    def view_matrix(self) -> Mat4:
        return Mat4.look_at(self.eye, self.target, self.up)

    def projection_matrix(self) -> Mat4:
        if self.projection_mode == "perspective":
            return Mat4.perspective(math.radians(self.fov_deg), self.aspect, self.near, self.far)
        dist = (self.target - self.eye).length()
        half_h = dist * math.tan(math.radians(self.fov_deg) / 2.0)
        half_w = half_h * self.aspect
        return Mat4.orthographic(-half_w, half_w, -half_h, half_h, self.near, self.far)

    def view_projection(self) -> Mat4:
        return self.projection_matrix() * self.view_matrix()

    def project_to_clip(self, p: Vec3) -> Vec4:
        VP = self.view_projection()
        return VP * Vec4(p.x, p.y, p.z, 1.0)

    def project_to_screen(self, p: Vec3, width, height):
        clip = self.project_to_clip(p)
        if clip.w <= 1e-6:
            return None
        ndc_x = clip.x / clip.w
        ndc_y = clip.y / clip.w
        ndc_z = clip.z / clip.w
        sx = (ndc_x * 0.5 + 0.5) * width
        sy = (1.0 - (ndc_y * 0.5 + 0.5)) * height
        return (sx, sy, ndc_z, clip.w)
