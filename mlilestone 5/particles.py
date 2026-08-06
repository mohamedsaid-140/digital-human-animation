"""
particles.py
------------
Milestone 5, part B: physics-based deformation / secondary motion.

A short particle chain ("ponytail") is attached to the head joint and
simulated with Verlet integration + distance (stick) constraints -- the
standard cheap, unconditionally-stable-for-this-purpose method used for
cloth/hair in games (Jakobsen, "Advanced Character Physics", 2001).

State per particle: current position AND previous position (implicit
velocity = current - previous), no explicit velocity variable needed.
This is what gives Verlet its stability advantage over explicit Euler
for stiff constraint systems (see stability_analysis / motion_analysis).
"""

from math3d import Vec3


class ParticleChain:
    def __init__(self, root_pos: Vec3, n_particles=6, segment_length=0.045,
                 gravity=Vec3(0, -9.81, 0), damping=0.985, constraint_iters=4):
        self.n = n_particles
        self.segment_length = segment_length
        self.gravity = gravity
        self.damping = damping
        self.constraint_iters = constraint_iters

        # start the chain hanging straight down from the root, at rest
        self.pos = []
        self.prev = []
        for i in range(n_particles):
            p = root_pos + Vec3(0, -segment_length * i, 0)
            self.pos.append(p)
            self.prev.append(p)  # zero initial velocity

    def step(self, root_pos: Vec3, dt):
        """
        root_pos: the current world-space position of the attachment
        joint (head), supplied externally every frame -- this is how the
        rigid skeleton animation drives the soft particle simulation.
        """
        # particle 0 is pinned to the animated root joint
        self.pos[0] = root_pos

        # Verlet integration for all free particles (1..n-1):
        #   x_new = x + (x - x_prev) * damping + a * dt^2
        for i in range(1, self.n):
            vel = (self.pos[i] - self.prev[i]) * self.damping
            new_pos = self.pos[i] + vel + self.gravity * (dt * dt)
            self.prev[i] = self.pos[i]
            self.pos[i] = new_pos

        # distance constraints: enforce fixed segment length between
        # consecutive particles, iterated for a stiffer-looking chain
        for _ in range(self.constraint_iters):
            self.pos[0] = root_pos  # re-pin every iteration
            for i in range(self.n - 1):
                a, b = self.pos[i], self.pos[i + 1]
                delta = b - a
                dist = delta.length()
                if dist < 1e-9:
                    continue
                diff = (dist - self.segment_length) / dist
                if i == 0:
                    # particle 0 is pinned: particle 1 absorbs the full correction
                    self.pos[1] = b - delta * diff
                else:
                    correction = delta * (0.5 * diff)
                    self.pos[i] = a + correction
                    self.pos[i + 1] = b - correction

    def constraint_error(self):
        """Mean absolute deviation of segment lengths from rest length --
        a direct measure of constraint (physical plausibility) violation."""
        errs = []
        for i in range(self.n - 1):
            d = (self.pos[i + 1] - self.pos[i]).length()
            errs.append(abs(d - self.segment_length))
        return sum(errs) / len(errs)
