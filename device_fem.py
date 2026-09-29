"""
FEM model of a single-hole electrons-on-helium quantum dot, built with ZeroHeliumKit
(gmsh meshing + FreeFEM Laplace solve).

Stack (bottom to top, z in micron, z=0 at the bottom of the plunger):

    vacuum
    ---------------------------  z_e = z_He + electron_height   <- electron plane
    ---------------------------  z_He = top of the helium film
    superfluid He film (van der Waals film on the top plate; He also fills the hole)
    top metal plate  (with hole)       t_top
    dielectric       (with same hole)  t_diel
    plunger metal    (full sheet)      t_plunger
    substrate

The hole is completely filled with superfluid helium by capillary action (the capillary
rise for a 100 nm radius is metres). The meniscus sag over the hole is <0.1 nm for bulk
helium 5 mm below (see `meniscus_sag_nm`), so the electron plane is treated as flat.

Two calculations are done on the same mesh:

1. Coupling constants (ZeroHeliumKit): alpha_k(x, y) on the electron plane, the potential in V
   when electrode k is at 1 V and all others at 0 V.

2. Image-charge Green's function (a small custom FreeFEM script, since ZHK has no point-charge
   solver): for a unit point charge at (s, 0, z_e) with all electrodes grounded, the induced
   potential g_s(x, y) on the electron plane. By rotational symmetry of the device this gives the
   full image interaction between any two electrons, and the self-image energy of one electron.
   Units: lengths in um with eps0 = 1, i.e. g is in 1/um; multiply by e/eps0 * 1e6 = 0.018095 eV*um
   to get the energy of a second electron in eV.

Everything is saved in one .npz; the 2D maps use the [ix, iy] orientation quantum_electron expects.

Usage:
    python device_fem.py --hole-diameter 0.175 --cores 4
"""

import argparse
import asyncio
import re
import subprocess
from pathlib import Path

import numpy as np
import psutil
import yaml

from zeroheliumkit import Structure, Layer, Square, Circle
from zeroheliumkit.fem import GMSHmaker, ExtrudeSettings, PECSettings, MeshSettings, BoxFieldMeshSettings
from zeroheliumkit.fem.freefemer import FFconfigurator, ExtractConfig, FreeFEM, scaling_size, detect_freefem
from zeroheliumkit.fem.fieldreader import FreeFemResultParser


def vdw_film_thickness_um(bulk_distance_m: float) -> float:
    """Saturated van der Waals helium film thickness, d ~ 30 nm * (1 cm / h)^(1/3).
    Order-of-magnitude estimate: the real value depends on the substrate, surface cleanliness
    and the electron density on the film. Override with --film if you know better."""
    return 0.030 * (1e-2 / bulk_distance_m) ** (1 / 3)


def meniscus_sag_nm(hole_diameter_um: float, bulk_distance_m: float) -> float:
    """Sag at the centre of a helium meniscus pinned on the rim of a circular hole:
    laplacian(u) = rho g h / alpha, u = 0 on the rim  ->  u(0) = (rho g h / alpha) R^2 / 4.
    Same physics and constants as ZeroHeliumKit's HeliumSurfaceFreeFEM."""
    R = hole_diameter_um / 2
    return scaling_size(bulk_distance_m) * R**2 / 4 * 1e3  # scaling_size is in 1/um -> u in um -> nm


def build_mesh(hole_diameter: float, t_plunger: float, t_diel: float, t_top: float, film: float,
               electron_height: float, domain: float, vacuum: float, substrate: float,
               fine_half_width: float, extract_half_width: float, mesh_fine: float, mesh_coarse: float,
               workdir: Path) -> dict:
    z_diel = t_plunger
    z_top = z_diel + t_diel
    z_He = z_top + t_top + film
    z_e = z_He + electron_height

    device = Structure()
    device.add(Layer("plane", Square(domain)))
    device.add(Layer("hole", Circle(hole_diameter / 2, num_edges=64)))
    device.add(Layer("holey", Square(domain)))
    device.holey.cut(device.hole.polygons)

    # ZHK's extruder ignores polygon interiors, so the hole in the dielectric and the top plate is
    # made with a boolean cut. `holey` is only used to find a point inside the top metal when
    # tagging the electrode. The plunger is a full sheet, so the substrate is irrelevant.
    volumes = {
        "substrate":   ExtrudeSettings(device.plane.polygons, -substrate, substrate, "SUBSTRATE"),
        "plunger":     ExtrudeSettings(device.plane.polygons, 0.0, t_plunger, "METAL"),
        "helium_hole": ExtrudeSettings(device.hole.polygons, z_diel, z_top + t_top - z_diel, "HELIUM"),
        "helium_film": ExtrudeSettings(device.plane.polygons, z_top + t_top, film, "HELIUM"),
        "dielectric":  ExtrudeSettings(device.plane.polygons, z_diel, t_diel, "DIELECTRIC", ("helium_hole",)),
        "top":         ExtrudeSettings(device.plane.polygons, z_top, t_top, "METAL", ("helium_hole",)),
        "vacuum":      ExtrudeSettings(device.plane.polygons, z_He, vacuum, "VACUUM"),
    }
    pecs = {
        "plunger": PECSettings(device.plane.polygons, [0], volume=volumes["plunger"]),
        "top":     PECSettings(device.holey.polygons, [0], volume=volumes["top"]),
    }

    w = fine_half_width
    mesh = MeshSettings(dim=3, fields={"Box": [
        # fine around the hole mouth and the electron plane, where the electrons live
        BoxFieldMeshSettings(Thickness=0.2, VIn=mesh_fine, VOut=mesh_coarse,
                             box=[-w, w, -w, w, z_top - 0.1, z_e + 0.15]),
        # moderately fine in the extraction window and down the hole
        BoxFieldMeshSettings(Thickness=0.5, VIn=3 * mesh_fine, VOut=mesh_coarse,
                             box=[-extract_half_width, extract_half_width, -extract_half_width, extract_half_width,
                                  z_diel, z_e + 0.4]),
    ]})

    GMSHmaker(extrude=volumes, pecs=pecs, mesh=mesh, save={"dir": str(workdir), "filename": "device"})
    return dict(z_diel=z_diel, z_top=z_top, z_He=z_He, z_e=z_e)


def solve_couplings(workdir: Path, z_e: float, eps_diel: float, extract_half_width: float,
                    extract_points: int, cores: int, freefem_path: str | None) -> dict:
    FFconfigurator(
        config_file=str(workdir / "device.yaml"),
        dielectric_constants={"SUBSTRATE": 11.7, "DIELECTRIC": eps_diel, "METAL": 1.0, "HELIUM": 1.057, "VACUUM": 1.0},
        ff_polynomial=2,
        # extraction names become FreeFEM identifiers: no underscores allowed
        extract_opt=[
            ExtractConfig("eplane", "phi", "xy",
                          (-extract_half_width, extract_half_width, extract_points),
                          (-extract_half_width, extract_half_width, extract_points), z_e),
            ExtractConfig("xzcut", "phi", "xz", (-0.5, 0.5, 201), (0.0, z_e + 0.5, 201), 0.0),
        ],
    )

    ff = FreeFEM(config_file=str(workdir / "device.yaml"))
    # ZHK always loads "mshmet" (only needed for mesh adaptation, which we don't use);
    # the Debian/Ubuntu FreeFEM packages don't ship it.
    for edp in ff.ffrunner.edp_files:
        text = Path(edp).read_text()
        Path(edp).write_text(re.sub(r'load "mshmet"\n', "", text))
    asyncio.run(ff.run(cores=cores, freefem_path=freefem_path, remove=True))

    parser = FreeFemResultParser(str(workdir / "metadata.yaml"), show=False)
    parser.load_data(str(workdir / "results"), "eplane")
    cc = parser.get_coupling_constants(slice_value=z_e)
    parser_xz = FreeFemResultParser(str(workdir / "metadata.yaml"), show=False)
    parser_xz.load_data(str(workdir / "results"), "xzcut")
    cc_xz = parser_xz.get_coupling_constants(slice_value=0.0)

    # ZHK stores maps as [iy, ix]; quantum_electron expects [ix, iy].
    out = {"xlist": np.asarray(cc.x), "ylist": np.asarray(cc.y)}
    for name, arr in cc.data.items():
        out[name] = np.asarray(arr).T
    out["xz_x"], out["xz_z"] = np.asarray(cc_xz.x), np.asarray(cc_xz.y)
    for name, arr in cc_xz.data.items():
        out[f"xz_{name}"] = np.asarray(arr)  # [iz, ix], for plotting only
    out["capacitance_matrix"] = np.asarray(parser.get_capacitance_matrix())
    return out


IMAGE_EDP = """\
load "msh3"
load "gmsh"
mesh3 Th = gmshload3("{mesh}");
fespace Vh(Th, P2);
fespace Ph(Th, P03d);
Ph epsm = 1.0 + {deps} * (region == {diel_id});

real sx = 0., sy = 0., sz = {z_e};
macro RR sqrt((x-sx)^2 + (y-sy)^2 + (z-sz)^2) //
// free-space potential of a unit charge (eps0 = 1) and its gradient
func uf = 1. / (4. * pi * RR);
func gx = -(x-sx) / (4. * pi * RR^3);
func gy = -(y-sy) / (4. * pi * RR^3);
func gz = -(z-sz) / (4. * pi * RR^3);

// u = uf + w with div(eps grad u) = -delta and u = 0 on all electrodes
// -> div(eps grad w) = -div((eps - 1) grad uf),  w = -uf on the electrodes.
// The helium (eps = 1.057) is treated as vacuum here; its image effect is ~1% of the metal's.
Vh w, v;
int initflag = 0;
problem Img(w, v, solver=CG, eps=1e-10, init=initflag) =
      int3d(Th)(epsm * (dx(w)*dx(v) + dy(w)*dy(v) + dz(w)*dz(v)))
    + int3d(Th, {diel_id})({deps} * (gx*dx(v) + gy*dy(v) + gz*dz(v)))
    + on({electrode_ids}, w = -uf);

real[int] sources = {sources};
int n = {n};
real h = {half_width};
ofstream out("{outfile}");
for (int m = 0; m < sources.n; m++) {{
    sx = sources[m];
    Img;
    initflag = 1;
    out << sx << " " << w(sx, 0., sz) << endl;
    for (int j = 0; j < n; j++) {{
        real Y = -h + j * 2. * h / (n - 1);
        for (int i = 0; i < n; i++) {{
            real X = -h + i * 2. * h / (n - 1);
            out << w(X, Y, sz) << endl;
        }}
    }}
}}
"""


def solve_images(workdir: Path, z_e: float, eps_diel: float, sources: np.ndarray, half_width: float,
                 n: int, cores: int, freefem_path: str | None) -> dict:
    with open(workdir / "device.yaml") as f:
        cfg = yaml.safe_load(f)
    exe = Path(freefem_path or detect_freefem()) / "FreeFem++"
    edp_dir = workdir / "edp"
    edp_dir.mkdir(exist_ok=True)

    procs, outfiles = [], []
    for k, chunk in enumerate(np.array_split(sources, cores)):
        outfile = edp_dir / f"images_{k}.txt"
        edp = edp_dir / f"images_{k}.edp"
        edp.write_text(IMAGE_EDP.format(
            mesh=cfg["meshfile"], deps=eps_diel - 1.0, diel_id=cfg["physicalVolumes"]["DIELECTRIC"],
            z_e=z_e, electrode_ids=", ".join(str(v) for v in cfg["physicalSurfaces"].values()),
            sources="[" + ", ".join(f"{s:.6f}" for s in chunk) + "]", n=n, half_width=half_width,
            outfile=outfile))
        log = open(edp_dir / f"images_{k}.log", "w")
        procs.append(subprocess.Popen([str(exe), "-nw", "-ns", str(edp)], stdout=log, stderr=subprocess.STDOUT))
        outfiles.append(outfile)
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError(f"FreeFEM image solve failed, see {edp_dir}/images_*.log")

    s_list, self_list, maps = [], [], []
    for outfile in outfiles:
        lines = outfile.read_text().split("\n")
        block = n * n + 1
        for b in range(len(lines) // block):
            sx, wself = map(float, lines[b * block].split())
            vals = np.array(lines[b * block + 1:(b + 1) * block], dtype=float)
            s_list.append(sx)
            self_list.append(wself)
            maps.append(vals.reshape(n, n).T)  # written as [iy][ix] -> [ix, iy]
    order = np.argsort(s_list)
    grid = np.linspace(-half_width, half_width, n)
    return {"img_sources": np.asarray(s_list)[order], "img_self": np.asarray(self_list)[order],
            "img_g": np.asarray(maps)[order], "img_x": grid, "img_y": grid}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hole-diameter", type=float, default=0.175, help="um")
    ap.add_argument("--t-diel", type=float, default=0.650, help="um")
    ap.add_argument("--t-top", type=float, default=0.050, help="um")
    ap.add_argument("--t-plunger", type=float, default=0.025, help="um")
    ap.add_argument("--eps-diel", type=float, default=3.9, help="relative permittivity of the dielectric (3.9 = SiO2)")
    ap.add_argument("--bulk-helium", type=float, default=5e-3, help="distance from the sample surface to bulk helium, m")
    ap.add_argument("--film", type=float, default=None, help="He film thickness on the top plate, um (default: vdW estimate)")
    ap.add_argument("--electron-height", type=float, default=0.0114,
                    help="electron height above the He surface, um (default 1.5 a_B = <z> of the ground state)")
    ap.add_argument("--mesh-fine", type=float, default=0.012, help="finest mesh size, um")
    ap.add_argument("--image-rmax", type=float, default=0.2, help="largest radius of image-charge sources, um")
    ap.add_argument("--image-step", type=float, default=0.01, help="radial spacing of image-charge sources, um")
    ap.add_argument("--no-images", action="store_true", help="skip the image-charge Green's function")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"hole_{round(args.hole_diameter * 1e3)}nm"
    workdir = Path(args.workdir or f"dump/{tag}")
    out = Path(args.out or f"fem_data/{tag}.npz")
    film = args.film if args.film is not None else vdw_film_thickness_um(args.bulk_helium)
    freefem_path = detect_freefem()

    print(f"helium film on top plate: {film * 1e3:.1f} nm")
    print(f"meniscus sag at hole centre: {meniscus_sag_nm(args.hole_diameter, args.bulk_helium):.3f} nm (neglected)")

    extract_half_width, extract_points = 0.6, 241
    geo = build_mesh(args.hole_diameter, args.t_plunger, args.t_diel, args.t_top, film, args.electron_height,
                     domain=3.0, vacuum=2.0, substrate=0.5,
                     fine_half_width=max(args.hole_diameter, args.image_rmax + 0.05),
                     extract_half_width=extract_half_width, mesh_fine=args.mesh_fine, mesh_coarse=0.3,
                     workdir=workdir)
    res = solve_couplings(workdir, geo["z_e"], args.eps_diel, extract_half_width, extract_points,
                          min(args.cores, psutil.cpu_count(logical=False) or 1), freefem_path)  # ZHK caps at physical cores
    if not args.no_images:
        sources = np.arange(0, args.image_rmax + 1e-9, args.image_step)
        res |= solve_images(workdir, geo["z_e"], args.eps_diel, sources, half_width=0.3, n=121,
                            cores=args.cores, freefem_path=freefem_path)

    res["meta"] = dict(hole_diameter=args.hole_diameter, t_plunger=args.t_plunger, t_diel=args.t_diel,
                       t_top=args.t_top, film=film, electron_height=args.electron_height,
                       eps_diel=args.eps_diel, bulk_helium_distance=args.bulk_helium, **geo)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res)

    i0 = np.argmin(np.abs(res["xlist"]))
    print(f"saved {out}")
    print(f"coupling at hole centre: plunger {res['plunger'][i0, i0]:.5f}, top {res['top'][i0, i0]:.5f}")
    if not args.no_images:
        C = 0.018095  # e / eps0 * 1e6, eV*um
        h = film + args.electron_height
        print(f"self-image energy at hole centre: {0.5 * C * res['img_self'][0] * 1e3:.2f} meV")
        print(f"self-image energy at r = {res['img_sources'][-1]:.2f} um: {0.5 * C * res['img_self'][-1] * 1e3:.2f} meV "
              f"(infinite metal plane at {h * 1e3:.0f} nm: {-C / (16 * np.pi * h) * 1e3:.2f} meV)")


if __name__ == "__main__":
    main()
