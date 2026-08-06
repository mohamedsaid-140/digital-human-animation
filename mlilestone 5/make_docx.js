const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, PageBreak, ExternalHyperlink
} = require("docx");

const TOTAL_WIDTH_DXA = 9360; // US Letter, 1" margins => 6.5in content width * 1440

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function code(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Consolas", size: 18 })],
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    spacing: { after: 160 },
  });
}
function figure(path, widthPx, heightPx, caption, maxWidthIn = 6.0) {
  const scale = Math.min(1, (maxWidthIn * 96) / widthPx);
  const w = Math.round(widthPx * scale);
  const h = Math.round(heightPx * scale);
  return [
    new Paragraph({
      children: [new ImageRun({ data: fs.readFileSync(path), transformation: { width: w, height: h }, type: "png" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
    }),
    new Paragraph({
      children: [new TextRun({ text: caption, italics: true, size: 18 })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
    }),
  ];
}

function metricTable(rows) {
  const header = new TableRow({
    tableHeader: true,
    children: ["Metric", "Value", "Verdict"].map((t, i) =>
      new TableCell({
        width: { size: [3600, 3160, 2600][i], type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: "2F4F6F" },
        children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: "FFFFFF" })] })],
      })
    ),
  });
  const body = rows.map((r) =>
    new TableRow({
      children: r.map((cellText, i) =>
        new TableCell({
          width: { size: [3600, 3160, 2600][i], type: WidthType.DXA },
          children: [new Paragraph({ text: cellText })],
        })
      ),
    })
  );
  return new Table({
    width: { size: TOTAL_WIDTH_DXA, type: WidthType.DXA },
    columnWidths: [3600, 3160, 2600],
    rows: [header, ...body],
  });
}

const doc = new Document({
  sections: [
    {
      properties: {
        page: { size: { width: 12240, height: 15840 } }, // US Letter
      },
      children: [
        new Paragraph({
          children: [new TextRun({ text: "Digital Humans & Character Animation", bold: true, size: 44 })],
          spacing: { after: 60 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Milestone 5 — Dynamics & Animation", bold: true, size: 32, color: "2F4F6F" })],
          spacing: { after: 40 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Temporal system modeling, motion realism, and procedural + physics-based control", italics: true, size: 22, color: "555555" })],
          spacing: { after: 400 },
        }),

        h1("1. Overview"),
        p(
          "This milestone extends the Milestone 1 representation (17-joint FK skeleton, LBS-skinned proxy mesh) " +
          "and Milestone 2 spatial pipeline (camera, view/projection, rasterizer) with temporal behavior. Two " +
          "independent animation sources drive the same skeleton, and a physics-based particle system adds " +
          "secondary motion, without any change to the underlying representation."
        ),
        bullet("Animation system: hand-authored keyframe clip (SLERP-interpolated) + closed-form procedural walk cycle."),
        bullet("Motion consistency: measured, not assumed — a foot-sliding metric quantifies a real locomotion artifact."),
        bullet("Deformation/particle system: a Verlet-integrated, constraint-solved particle chain (ponytail) for secondary motion."),

        h1("2. Animation System"),
        h2("2.1 Keyframe clip (SLERP)"),
        p(
          "KeyframeClip stores sparse (time, quaternion) keys per joint and evaluates arbitrary t by SLERP between " +
          "the two surrounding keys. This is the standard representation for hand-authored or motion-captured animation."
        ),
        code("q(t) = slerp(q_i, q_{i+1}, (t - t_i)/(t_{i+1} - t_i))   for t_i <= t <= t_{i+1}"),
        ...figure("out/m5_wave_still.png", 640, 640, "Figure 1. Frame from the keyframe wave clip (full sequence: out/m5_wave.gif)."),

        h2("2.2 Procedural walk cycle"),
        p(
          "procedural_walk_cycle() is a closed-form function of a single phase variable phi = 2*pi*stride_hz*t. " +
          "Hips, knees, and shoulders are driven by sinusoids of phi with anatomically correct phase offsets " +
          "(left/right legs at pi, contralateral arm swing, rectified-sine knee flexion during swing only)."
        ),
        p(
          "A real bug was found and fixed while building this: rotating the upperarm joint about the swing axis did " +
          "nothing, because in the T-pose rest skeleton the upperarm's bone offset is parallel to that rotation axis " +
          "— a joint's own rotation only ever moves its children, not the segment connecting it to its parent. " +
          "The fix drives the swing (plus a fixed bias that lowers the arm from T-pose to a hanging A-pose) from the " +
          "clavicle joint instead, which is the parent of upperarm.",
          { }
        ),
        ...figure("out/m5_walk_still.png", 640, 640, "Figure 2. Frame from the procedural walk cycle, chase camera (full sequence: out/m5_walk.gif)."),

        h1("3. Motion Analysis"),
        p(
          "Four time-series were extracted directly from the running system (not hand-plotted) and are shown below: " +
          "right-hand height through the wave clip, gait vertical trajectories, left-foot horizontal velocity, and " +
          "ponytail constraint error."
        ),
        ...figure("out/m5_motion_analysis.png", 1300, 910, "Figure 3. Motion analysis: keyframe continuity, gait pattern, foot sliding, ponytail stability."),
        h2("3.1 Keyframe interpolation quality"),
        p(
          "The hand-height curve is continuous and shows no velocity discontinuity ('popping') at any keyframe " +
          "boundary — the expected result of SLERP between unit quaternions, versus naive per-component linear " +
          "interpolation of Euler angles, which does not preserve constant angular velocity."
        ),
        h2("3.2 Gait pattern"),
        p(
          "Left/right foot height traces alternate correctly with a half-cycle phase offset, and the pelvis bob " +
          "shows the expected double-frequency component (rising during both left- and right-stance sub-phases). " +
          "This confirms the phase-offset math is internally consistent, not just visually plausible."
        ),
        h2("3.3 Foot sliding — a measured, disclosed limitation"),
        p(
          "The left foot's horizontal velocity is flat at approximately 0.55 m/s (equal to the root's walking " +
          "speed) for the entire cycle, including while the foot is at its lowest point. A physically correct walk " +
          "would hold this near zero during stance. The cause: the system has no foot-plant / IK layer — the " +
          "pelvis translates forward at a constant rate independent of gait phase. This is reported here as a " +
          "genuine, quantified defect and a concrete Milestone 6 target, not smoothed over."
        ),

        h1("4. Physics-Based Deformation: Ponytail Particle System"),
        p(
          "A 6-particle chain is attached to the head joint and simulated with Verlet integration (position + " +
          "previous-position, no explicit velocity state) followed by 4 iterations of distance-constraint " +
          "relaxation per frame."
        ),
        code("x_new(i) = x(i) + damping*(x(i) - x_prev(i)) + g*dt^2      [free particles, i>0]\nx(0) = head_joint_world_position(t)                          [pinned particle]"),
        p(
          "Verlet + position-based constraints is unconditionally stable for this class of stiff chain at real-time " +
          "frame rates, unlike explicit-Euler velocity integration, which tends to gain energy and blow up on stiff " +
          "constraints unless the timestep is very small."
        ),
        h2("4.1 Stability result"),
        bullet("Mean constraint error: 0.0073 m against a 0.045 m rest segment length (~16%)."),
        bullet("Maximum transient error: 0.0107 m (~24%), during the first ~0.15s settling period only."),
        bullet("Error does not grow over the sequence — it oscillates in a bounded envelope tracking the gait bob, rather than drifting upward: the signature of a stable simulation, not an accumulating one."),
        p(
          "Four constraint iterations under-relaxes a fairly stiff chain, hence the double-digit percent transient " +
          "stretch. This is a disclosed accuracy/performance trade-off — more iterations tighten the error at " +
          "linear extra per-frame cost — rather than a hidden bug."
        ),

        h1("5. Numerical Precision Notes"),
        p(
          "Two effects carried over from Milestone 2 remain relevant now that poses change every frame:"
        ),
        bullet("Quaternion drift does not accumulate frame-to-frame here, because every set_pose() call replaces a joint's quaternion outright rather than compounding an incremental rotation onto the previous frame's value — the failure mode demonstrated separately in the Milestone 2 stability experiments."),
        bullet("Quaternion.slerp falls back to normalized linear interpolation when dot > 0.9995, avoiding a 0/0 from sin(theta_0) in the denominator as two keys approach each other — exercised whenever adjacent keyframes are nearly identical, e.g. the wave clip's return-to-rest segments."),

        h1("6. Summary Evaluation"),
        metricTable([
          ["Ponytail mean constraint error", "0.0073 m (16% of segment)", "Bounded, non-growing -> stable"],
          ["Ponytail max transient error", "0.0107 m (24%)", "Settling transient only, decays -> acceptable"],
          ["Foot sliding (stance)", "0.55 m/s (should be ~0)", "Known limitation -> needs foot IK (Milestone 6)"],
          ["Keyframe interpolation continuity", "No visible popping at key boundaries", "Correct"],
        ]),
        new Paragraph({ text: "", spacing: { before: 200 } }),

        h1("7. Deliverable Files"),
        bullet("animation.py — KeyframeClip (SLERP) + procedural_walk_cycle"),
        bullet("particles.py — Verlet-integrated ParticleChain with distance constraints"),
        bullet("render_m5.py — driver: renders both sequences, runs motion + stability analysis"),
        bullet("out/m5_wave.gif, out/m5_walk.gif — animated system output"),
        bullet("out/m5_motion_analysis.png — the four analysis plots reproduced in Section 3"),
        bullet("motion_analysis.md, stability_analysis.md — source analysis notes for this document"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("out/Milestone5_DigitalHumans.docx", buf);
  console.log("wrote out/Milestone5_DigitalHumans.docx");
});
