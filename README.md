# electron_solver

This repo simulates the number of electrons versus plunger voltage for a single-hole quantum dot
of electrons on helium.

- **Electrostatics:** ZeroHeliumKit (gmsh + FreeFEM).
- **Electron configurations:** quantum_electron.
- **Image-charge interactions:** a small custom FreeFEM Green's-function solve on the same mesh
  (neither package does this).

## Device

```
            e⁻  e⁻                         <- electrons, 11 nm above the He surface
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   <- He film on top plate (~38 nm, bulk He 5 mm below)
 ██████████████████        ██████████████  top metal, 50 nm, hole 150-200 nm
 ░░░░░░░░░░░░░░░░░░  He    ░░░░░░░░░░░░░░  dielectric, 650 nm, same hole
 ░░░░░░░░░░░░░░░░░░        ░░░░░░░░░░░░░░
 ██████████████████████████████████████    plunger (bottom metal), 25 nm
```

Assumptions (all can be changed with command-line flags):

- **Helium level.** The hole is completely filled with superfluid helium by capillary action. For
  bulk helium 5 mm below, the helium surface is flat to within 0.04 nm over the hole. This uses
  ZeroHeliumKit's meniscus physics, `rho g h R^2 / (4 alpha)`.
- **Film thickness.** The film thickness on the top plate uses the van der Waals estimate
  `30 nm * (1 cm / h)^(1/3)`, which gives 38 nm. Use `--film` if you know the real value.
- **Electron height.** Electrons are placed 11.4 nm above the helium surface. That is 1.5 a_B, the
  mean height of the image-bound ground state.
- **Dielectric.** ε = 3.9 (SiO₂). The plunger is a continuous sheet. Both metals are ideal
  conductors. A different work function only shifts that electrode's voltage (see `--mu-offset`).

## Setup

```bash
./setup_env.sh                                   # apt packages, venv, zeroheliumkit, quantum_electron
. .venv/bin/activate && export FF_LOADPATH=/usr/lib/freefem++
```

## Run

```bash
python device_fem.py --hole-diameter 0.175                   # -> fem_data/hole_175nm.npz  (~10 min on 4 cores)
python n_vs_plunger.py fem_data/hole_175nm.npz --dv-max 20   # -> results/hole_175nm*.png, .npz
```

## How it works

### `device_fem.py`

It uses ZeroHeliumKit's FEM tool chain:
`Structure/Layer` → `GMSHmaker` (3D mesh) → `FFconfigurator`/`FreeFEM` (Laplace solve per
electrode) → `FreeFemResultParser` (coupling maps).

1. **Coupling constants.** For each electrode, α_k(x, y) on the electron plane is the potential
   when that electrode is at 1 V and all others are at 0 V. Everything is enclosed by the two
   electrodes, so α_plunger + α_top = 1. Only **ΔV = V_plunger − V_top** matters.
2. **Image-charge Green's function.** For a unit charge at radius s in the electron plane, with
   both electrodes grounded, the script computes the induced potential g_s(x, y). Because the
   device is rotationally symmetric, this gives the self-image energy of an electron and the
   image-mediated part of every electron–electron interaction.
   - This matters here because the electrons sit only ~50 nm above the top plate. The self-image
     energy is −7.3 meV over the metal and about −4.7 meV at the hole centre, so images pull
     electrons out of the dot. That is comparable to the plunger well, which is only a few meV
     per volt deep.
   - Check: far from the hole the FEM self-energy agrees with the infinite-plane value `-e²/(16πε₀h)`
     to better than 1%.

### `n_vs_plunger.py`

This is a grand-canonical, zero-temperature calculation.

1. For each ΔV and each N, it finds the classical ground state with quantum_electron's `FullModel`.
   The energy includes the external potential and the Coulomb repulsion, and here also the image
   energy. It uses several initial conditions, continuation in voltage, and seeding from the
   (N−1)-electron solution.
2. The dot holds the N that minimises `E(N) − N μ_res`. μ_res is the reservoir's electrochemical
   potential. By default the reservoir is an electron on the helium film over the top plate. Use
   `--reservoir-density` to account for a charged reservoir sheet, or `--mu-offset` for anything
   else.
3. It reports the transition voltages and addition voltages, and plots N(ΔV), the addition
   energies μ(N) against μ_res, the radial potential profile, and the ground-state configurations.

## Notes on ZeroHeliumKit (as of v0.5.5)

Found while building this:

- **Polygons with interior holes.** `GMSHmaker` ignores polygon interiors: exterior and interior
  coordinates are joined into one broken loop. Make holes with the `cut` option of
  `ExtrudeSettings` instead, as done here.
- **Underscores in names.** `ExtractConfig` names become FreeFEM identifiers, so they can't contain
  underscores.
- **`mshmet` plugin.** The generated scripts always `load "mshmet"`. The Debian/Ubuntu FreeFEM
  packages don't ship it. It's only needed for mesh adaptation, so this repo strips that line.
- **False "FreeFEM error".** The runner logs "FreeFEM error" whenever the output doesn't end with
  `Ok: Normal End`. The Ubuntu build never prints that line, so check the `GC: converge` lines in
  `dump/*/logs` instead.
- **No point-charge solver.** There is nothing for point charges or Green's functions
  (`script_include_charge` is a stub), hence the custom image solve.

## Limitations

- **Classical electrons.** The in-plane zero-point energy (ħω ≈ 0.4–1.3 meV for ΔV = 2–20 V, ignoring the
  image contribution to the curvature)
  and tunnelling are not included. Temperature is not included either.
- **Helium dielectric.** In the image calculation the helium (ε = 1.057) is treated as vacuum,
  which is a ~1% effect.
- **Reservoir.** The reservoir is idealised as a fixed electrochemical potential. The field of
  reservoir electrons near the hole is ignored.
