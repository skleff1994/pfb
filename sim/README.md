# Drake simulation study (`sim/`)

These scripts check the certified load factor of `pfb` on simulated contacts. They use Drake's hydroelastic contact model, with pressure fields on compliant pads. Drake lives in its own pixi environment, so every command runs with `-e sim`.

## Watch it

| Command | What you see |
|---|---|
| `pixi run -e sim sim-smoke` | A block settling into a compliant pad. Meshcat runs live at http://localhost:7000 and the run is saved as `out/smoke.html`. The pressure map and the CoP are saved as `out/smoke.png`. |
| `pixi run -e sim sim-show --seed 0 --policy MT16` | One closed-loop grip episode. Meshcat shows the bar, the pads, the contact force and moment arrows, and the applied load. Saved files: `out/episode_*.html` (replay), `out/episode_*.png` (dashboard), `out/episode_*.mp4` (animated dashboard). |
| `pixi run -e sim sim-show --seed 0 --policy mask_uniform` | The same seed with the F/T + contact-mask baseline. Its estimate sits above the true capacity, and the bar slips silently. |

**Viewer options** (`sim-show`):
- `--margin` (default 0.05) and `--scenario A|B`;
- `--no-meshcat`, `--no-video`;
- `--no-wait`: without it, the script waits for Enter so that the live viewer stays open.

The policies are listed in `policies.py`. The saved `.html` replays open in any browser without Drake: use the *Animations* panel to play or scrub.

**Dashboard** (`sim-show`):
- (a) The applied demand D, the policy's estimate s_hat and the true capacity s_true of the current contact. A sound estimate stays below s_true.
- (b) The measured, commanded and requested squeeze, with the force budget, and the flag when the policy cannot certify the load within the budget.
- (c) Slip.
- (d, e) Both pads, seen from +y, showing:
  - Drake's contact faces, filled with their pressure;
  - the policy's taxel grid;
  - the CoP;
  - for a certificate, its worst-case point loads: the compatible pressure that attains s_hat (Prop. `prop:load_factor`(iii)).

## Pipeline

| Stage | Script | Command | Output |
|---|---|---|---|
| S1 | `smoke_contact.py` | `sim-smoke` | Face forces add up to the weight and to Drake's resultant |
| S2 | `bar_breakaway.py` | `sim-breakaway` | Break-away of a bar in a fixed grip, compared with the true and certified load factors |
| S2 | `sweep_breakaway.py` | `sim-sweep` | The same experiment over solver settings (the S2 table) |
| S3 | `grip_study.py` | `sim-grip` (`--quick`, `--recompute`) | Closed-loop study, 3780 episodes, cached in `data/grip_study_v2.npz` |
| S3 | `plot_grip.py` | `sim-grip-plot` | `examples/figures/s3_grip.png`, `$PFB_DOC_FIG/grip_pareto.pdf` |
| S4 | `delay_study.py` | `sim-delay` (`--quick`, `--recompute`) | Sensing delay × load rate × load preview, at the S3 margins; cached in `data/delay_study.npz` |
| S4 | `plot_delay.py` | `sim-delay-plot` | `examples/figures/s4_delay.png`, `$PFB_DOC_FIG/delay_envelope.pdf` |
| — | `show_episode.py` | `sim-show` (`--delay`, `--rate`, `--no-preview` for S4) | One episode, visualized |

Vector figures go to `$PFB_DOC_FIG` (see `examples/_style.py`).

## Modules

| Module | Role |
|---|---|
| `sensing.py` | Drake contact surface → measure. Each face is one point load (pressure × area at its centroid). Also contains the taxel binning, the common frame, and `PadContact`. |
| `grip.py` | S3 scene (two pads on prismatic joints, the bar on a planar joint), load profiles, controller, `episode()` |
| `policies.py` | Capacity estimators: oracle, certified M_T / M_F, one-pad, plug-in, contact mask, uniform |
| `grip_study.py` | Runs the study, the outcome definitions, matched seeds, tuned-margin transfer |
| `viz.py` | Meshcat setup and recording, the load arrow, pad pressure maps |

## Conventions

- **Grasp scenes:**
  - the grip axis is world y;
  - the grasp plane is world (x, z);
  - common coordinates u = (z, x), so the planar cross product is the torque about +y;
  - reference point o = 0;
  - loads w = (f_z, f_x, τ_y).
- **Pads:** pad 1 is at −y (index 0 of `contact_state`) and is the sensed pad of `MT16_onepad`. Each pad has a 32 × 32 mm taxel footprint centred at o.
- **Controller** (all policies): f_cmd = f_meas · D(1 + m)/s_hat every 50 ms, clipped to the force budget. Linear scaling of capacity with squeeze is a declared heuristic. The full logic and its reasons are in the `grip.py` docstring.
- **Outcomes:**
  - success;
  - infeasible: the request exceeded the budget before any slip;
  - silent slip: slip > 0.2 mm without an earlier flag.
