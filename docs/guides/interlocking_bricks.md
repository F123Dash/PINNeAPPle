# Self-modifying brick builds

`pinneapple_design.bricks` builds structures from interlocking toy bricks, judges every build with physics and
redesigns it until it meets a goal: the longest cantilever past a table edge, a bridge that carries a load with the
fewest parts, the tallest tower that survives a push. The geometry follows the LDraw unit system, so builds open in
LDraw-compatible editors and real part shapes can be loaded from the LDraw parts library.

```python
from pinneapple_design.bricks import Assembly, Ground, analyse_assembly

a = Assembly(ground=Ground("table", [(-1000, 0, -1000, 1000)]))   # a table that ends at x = 0
a.add("brick_1x8", -4, 0, 0, colour="blue")                         # half of it past the edge
a.add("brick_1x8", 0, 0, 3, colour="yellow")                        # on top, reaching 64 mm further
r = analyse_assembly(a)
print(r.status, r.explanation)                                      # mechanism: where the centre of mass left
```

## The parts

| quantity | value | source |
|---|---|---|
| stud pitch | 8.0 mm (20 LDU) | LDraw unit system |
| plate / brick height | 3.2 mm / 9.6 mm (8 / 24 LDU) | LDraw unit system |
| stud | 4.8 mm diameter, 1.6 mm high | LDraw `stud.dat` primitive |
| part width | 8 n - 0.2 mm (0.1 mm clearance per side) | StableLego configuration (7.8 mm unit) |
| brick masses | e.g. 2x4: 2.16 g, 1x8: 3.03 g | StableLego brick library (Liu et al. 2024) |
| plate masses | shell model fitted to the brick masses (worst residual +28 % on the 1x1 brick) | `parts.fit_mass_model` |

The catalogue (`parts.CATALOG`) holds rectangular bricks and plates with their LDraw part numbers. Assemblies live on
the stud grid (x, y in stud pitches, z in plate heights). Two parts are joined where the top of one overlaps the
bottom of the other: each overlapping cell is one stud-tube contact. `Assembly.validate()` reports overlaps and parts
with no contact path to the ground; the ground is a table (smooth) or a baseplate (studs), optionally limited to
regions such as a table edge or two banks.

## The physics

**Statics** (`statics.analyse`). Every part is a rigid body. Every contact patch carries compression at its four
corners, and a stud joint also carries a clutch tension at each stud (at most `JointModel.clutch_n`) and a
horizontal force at each stud; smooth contacts carry Coulomb friction. A linear programme finds the force
distribution with the smallest peak stud utilisation. If one exists within the clutch, the build stands (the
lower-bound theorem of plasticity); if none exists even with unbreakable studs, the build is a mechanism and the
report says which support polygon the resultant left and by how much. `capacity` gives the largest factor on the
external loads, and `progressive_failure` releases the critical joints round by round.

**Joint model.** The clutch is 0.98 N per stud, the friction capacity reported for the StableLego analysis (Liu,
Deng, Wang, Liu, IEEE RA-L 9(11) 2024) and reused by BrickGPT (Pun et al., arXiv:2505.05469). No uncertainty is
published, so the sensitivity study scales it by an assumed 0.5x to 1.5x. Limits of the model:

- tension only: a joint fails when its pull-out force exceeds the clutch;
- horizontal stud forces are unbounded by default (as in StableLego); `JointModel(shear_n=...)` caps them;
- a single stud is a pivot about the vertical (no torsion);
- rigid parts and no joint compliance: the force distribution is the most favourable one, so the verdict is
  optimistic for brittle clutch;
- side-by-side parts do not push on each other (there is a 0.2 mm gap);
- contact forces act at least `edge_inset_mm` (0.5 mm, assumed) inside a patch edge. Without it the search drives
  builds onto a knife edge (centre of mass exactly over a table edge), which the statics accepts with zero margin
  and Newton tips; the closed-form checks use sharp edges (`edge_inset_mm=0`).

**Dynamics** (`dynamics.replay`). NVIDIA Newton replays a build: every cluster held together by intact studs is one
rigid body made of its boxes, released joints become box-box contacts with friction, tables have edges and a floor
below, and vertical loads are dead weights. Newton has no breakable joints, so the statics decides which joints let
go and Newton shows what happens next.

**FEA** (`fea.solve`). The voxelised build as one Hex8 solid (incompatible modes), for load paths and stresses in
the plastic. It is an upper bound on stiffness: joint compliance is not modelled.

## The loop

`search.evolve` and `search.surrogate_search` mutate builds with plain edits (add, remove, move, swap, rotate,
recolour) and grammar rules that read the last evaluation (extend toward the goal, add a counterweight, brace the
joint that failed, widen the base, mirror a half of a span). Every candidate is simulated; duplicates are looked
up. Evolution selects survivors with NSGA-II and Deb's constraint handling on the goal's objectives (for example
reach and part count). The surrogate search proposes eight times more children than it can afford, ranks them with
a Gaussian process trained on every build simulated so far (upper confidence bound on a penalised score) and
simulates only the best. `search.compare` runs both with the same evaluation budget and seeds.

## Visuals

- `visuals.evolution_timeline` + `timeline_scene(...).export(folder)`: the twin3d three.js viewer plays the
  evolution generation by generation. Build frames are coloured by stud utilisation (the load path); rejected
  candidates are replayed by Newton with the joints that let go in red. Serve the folder over http.
- `visuals.render_frames`, `write_gif`, `write_mp4`: the same frames from the software renderer.
- `visuals.render_blender`, `turntable_blender`: Blender Cycles renders (`pip install bpy`).
- `visuals.write_instructions`: step images per layer, a part list and an LDraw file with one STEP per layer.

## LDraw files and the parts library

`ldraw.write_ldr` / `write_mpd` write builds; `ldraw.read_ldr` reads .ldr and .mpd files (sub-models expanded,
quarter-turn rotations; parts off the grid or outside the catalogue are listed in `skipped`).
`ldraw.fetch_parts_library()` downloads the LDraw parts library at run time into a cache folder; it is licensed
CC BY 2.0 ("LDraw.org Parts Library") and is not copied into this repository. `ldraw.PartsLibrary(root)`
triangulates real parts for renders.

## The experiment

```bash
python examples/bricks/self_modifying_bricks.py --full       # 5 seeds x 400 evaluations per strategy and goal
python -m pinneapple_lab sweep bricks -g case=cantilever,bridge,tower
```

RESULTS
