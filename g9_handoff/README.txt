G9 Flying Kick - Sim2Real Handoff

Run:
  2026-08-17_21-34-16_G9_tail_direct_v2_scratch_20001

Checkpoint source:
  model_15000.pt
  NOTE: .pt is intentionally NOT included so the handoff stays below 50 MB.

Deployment model:
  deploy/policy.onnx

Reference motion:
  motion/G9_rightfeiti_tail_direct_v2.npz

Default/reset stance:
  motion/G9_default_stance.csv (when present)

Task:
  Tracking-Flat-G1-Wo-State-Estimation-G9-Stable-v0

Training:
  Scratch training, no checkpoint resume.
  Motion tail removes the old default-pose return and goes directly back
  to the locomotion stance.

Before hardware:
  Verify joint order, observation order, action scaling, default joint pose,
  stiffness/damping and emergency stop behavior. Test sim2sim first.

Package layout:
  deploy/   deployment ONNX
  motion/   final NPZ + CSV
  config/   run params
  source/   relevant task/config source
  sim2sim/  MuJoCo runner + alignment diagnostics
