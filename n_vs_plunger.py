"""
Number of electrons in the hole dot versus plunger voltage, using quantum_electron for the
classical ground-state configurations and the FEM data from device_fem.py.

Method (grand canonical, T = 0):
  For each plunger voltage and each N, find the lowest-energy configuration of N electrons,
      E(N) = sum_i -e phi(r_i)  +  sum_{i<j} e^2 / (4 pi eps0 r_ij)  +  E_image,
  where phi = V_plunger * alpha_plunger + V_top * alpha_top and E_image is the interaction of the
  electrons with their own images in the grounded electrodes (self-image of each electron + the
  image-mediated part of every pair interaction), from the FEM Green's function.
  The dot is in equilibrium with a reservoir at electrochemical potential mu_res, so the number of
  electrons is the N that minimises E(N) - N mu_res. Transitions happen where the addition energy
  mu(N) = E(N) - E(N-1) crosses mu_res.

Because alpha_plunger + alpha_top = 1 everywhere, only dV = V_plunger - V_top matters; energies are
reported relative to -e V_top per electron.

Reservoir: by default an electron on the helium film far out over the top plate, i.e.
  mu_res = -e V_top + (self-image energy over the metal) + e^2 n_res h / eps0,
where the last term is the charging of a reservoir sheet of density n_res at height h above the top
plate (0 for a dilute reservoir). Use --mu-offset to add anything else (e.g. a different metal work
function, or a reservoir that is not on this film).

Usage:
    python n_vs_plunger.py fem_data/hole_175nm.npz --dv-max 10 --n-max 5
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import elementary_charge as q_e, epsilon_0 as eps0
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates, spline_filter

from quantum_electron import FullModel
from quantum_electron.utils import r2xy, xy2r

C_IMG = q_e / eps0 * 1e6  # eV*um: energy of charge e in the induced potential of a unit charge (eps0=1, um)


class ImageInteraction:
    """Image-charge energy of electrons above the (rotationally symmetric) electrodes.
    G(rho_i, rho_j, dtheta) is the induced potential at electron i due to electron j (unit charges),
    tabulated from the FEM maps g_s(x, y) for sources at (s, 0) and extended by rotation."""

    def __init__(self, fem: dict, n_theta: int = 73):
        self.rho = fem["img_sources"]  # um, uniform spacing starting at 0
        self.drho = self.rho[1] - self.rho[0]
        self.rmax = self.rho[-1]
        self.dtheta = np.pi / (n_theta - 1)
        theta = np.linspace(0, np.pi, n_theta)

        G = np.empty((len(self.rho), len(self.rho), n_theta))
        for m, g in enumerate(fem["img_g"]):
            spline = RectBivariateSpline(fem["img_x"], fem["img_y"], g)
            P, T = np.meshgrid(self.rho, theta, indexing="ij")
            G[:, m, :] = spline.ev(P * np.cos(T), P * np.sin(T))
        G = 0.5 * (G + G.transpose(1, 0, 2))  # reciprocity G(a, b) = G(b, a) removes numerical asymmetry
        self.coeffs = spline_filter(G, order=3, mode="mirror")  # mirror = even in theta at 0 and pi

    def pair_table(self, x, y):
        """Matrix G_ij in 1/um for electron positions in um."""
        rho = np.hypot(x, y)
        th = np.arctan2(y, x)
        dth = np.abs(np.angle(np.exp(1j * (th[:, None] - th[None, :]))))  # in [0, pi]
        ri, rj = np.broadcast_arrays(rho[:, None], rho[None, :])
        coords = np.array([ri.ravel() / self.drho, rj.ravel() / self.drho, dth.ravel() / self.dtheta])
        return map_coordinates(self.coeffs, coords, order=3, mode="mirror", prefilter=False).reshape(ri.shape)

    def energy(self, x, y):
        """Total image energy in eV (x, y in um), including the self-image of each electron."""
        return 0.5 * C_IMG * np.sum(self.pair_table(x, y))

    def self_energy(self, rho):
        rho = np.atleast_1d(rho)
        return 0.5 * C_IMG * np.diag(self.pair_table(rho, np.zeros_like(rho)))


class DotModel(FullModel):
    """quantum_electron's FullModel with the image-charge energy added to the cost function."""

    def __init__(self, *args, images: ImageInteraction | None = None, fd_step: float = 0.1e-9, **kwargs):
        super().__init__(*args, **kwargs)
        self.images = images
        self.fd_step = fd_step

    def Vtotal(self, r):
        E = super().Vtotal(r)
        if self.images is not None:
            E += self.images.energy(r[::2] * 1e6, r[1::2] * 1e6)
        return E

    def grad_total(self, r):
        g = super().grad_total(r)
        if self.images is not None:
            h = self.fd_step
            for k in range(len(r)):
                rp, rm = r.copy(), r.copy()
                rp[k] += h
                rm[k] -= h
                g[k] += (self.images.energy(rp[::2] * 1e6, rp[1::2] * 1e6)
                         - self.images.energy(rm[::2] * 1e6, rm[1::2] * 1e6)) / (2 * h)
        return g


def initial_conditions(n: int, n_trials: int, rng: np.random.Generator, previous=None):
    """A few starting configurations (in m): rings of different radii with random jitter, plus the
    solution at the previous voltage (continuation)."""
    ics = [] if previous is None else [previous]
    if not ics and n_trials == 0:
        n_trials = 1
    for k in range(n_trials):
        radius = (0.01 + 0.05 * k / max(n_trials - 1, 1)) * 1e-6 if n > 1 else 0.0
        phi = 2 * np.pi * np.arange(n) / n + rng.uniform(0, 2 * np.pi)
        x = radius * np.cos(phi) + rng.normal(0, 3e-9, n)
        y = radius * np.sin(phi) + rng.normal(0, 3e-9, n)
        ics.append(xy2r(x, y))
    return ics


def ground_state(fem: dict, images, dV: float, n: int, r_bound: float, n_trials: int,
                 rng: np.random.Generator, previous=None):
    """Lowest energy (eV, relative to -e V_top per electron) and positions of n electrons.
    Returns (inf, positions) if an electron is not bound to the dot (|r| > r_bound)."""
    potential_dict = {"xlist": fem["xlist"], "ylist": fem["ylist"], "plunger": fem["plunger"], "top": fem["top"]}
    model = DotModel(potential_dict, {"plunger": dV, "top": 0.0}, images=images,
                     potential_smoothing=0.0, trap_annealing_steps=[0.1] * 3,
                     max_x_displacement=5e-9, max_y_displacement=5e-9)
    best = None
    for ic in initial_conditions(n, n_trials, rng, previous):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.get_electron_positions(n_electrons=n, electron_initial_positions=ic, suppress_warnings=True)
        if best is None or res["fun"] < best["fun"]:
            best = res
    x, y = r2xy(best["x"])
    if np.max(np.hypot(x, y)) > r_bound * 1e-6:
        return np.inf, best["x"]
    return best["fun"], best["x"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fem", help=".npz from device_fem.py")
    ap.add_argument("--dv-min", type=float, default=0.0, help="min V_plunger - V_top (V)")
    ap.add_argument("--dv-max", type=float, default=10.0, help="max V_plunger - V_top (V)")
    ap.add_argument("--dv-points", type=int, default=41)
    ap.add_argument("--n-max", type=int, default=5)
    ap.add_argument("--trials", type=int, default=4, help="random initial conditions per (N, V)")
    ap.add_argument("--no-images", action="store_true", help="ignore image charges (bare Coulomb only)")
    ap.add_argument("--reservoir-density", type=float, default=0.0, help="reservoir sheet density, cm^-2")
    ap.add_argument("--mu-offset", type=float, default=0.0, help="extra reservoir electrochemical potential, eV")
    ap.add_argument("--out", default=None, help="output prefix (default results/<fem name>)")
    args = ap.parse_args()

    fem = dict(np.load(args.fem, allow_pickle=True))
    meta = fem["meta"].item()
    images = None if args.no_images or "img_g" not in fem else ImageInteraction(fem)
    r_bound = images.rmax - 0.02 if images else 0.6 * meta["hole_diameter"] + 0.1

    # reservoir electrochemical potential (relative to -e V_top)
    h = meta["film"] + meta["electron_height"]  # um, electron height above the top plate
    mu_res = args.mu_offset + q_e * args.reservoir_density * 1e4 * h * 1e-6 / eps0
    if images is not None:
        mu_res += -C_IMG / (16 * np.pi * h)  # self-image over the (infinite) top plate

    dVs = np.linspace(args.dv_min, args.dv_max, args.dv_points)
    E = np.full((len(dVs), args.n_max + 1), np.inf)
    E[:, 0] = 0.0
    configs = {}
    rng = np.random.default_rng(0)
    for n in range(1, args.n_max + 1):
        # sweep down and then up in voltage, continuing from the neighbouring solution, and also try
        # the (n-1)-electron ground state plus one extra electron near the centre
        for order in (range(len(dVs) - 1, -1, -1), range(len(dVs))):
            previous = None
            for k in order:
                seeds = [previous]
                if n > 1 and (n - 1, k) in configs and np.isfinite(E[k, n - 1]):
                    x, y = r2xy(configs[(n - 1, k)])
                    seeds.append(xy2r(np.append(x, rng.normal(0, 20e-9)), np.append(y, rng.normal(0, 20e-9))))
                for seed in seeds:
                    e, pos = ground_state(fem, images, dVs[k], n, r_bound, args.trials if seed is previous else 0,
                                          rng, seed)
                    if e < E[k, n] or (n, k) not in configs:
                        E[k, n], configs[(n, k)] = e, pos
                previous = configs[(n, k)] if np.isfinite(E[k, n]) else None
        print(f"N = {n} done")

    # addition energies and ground-state N
    with np.errstate(invalid="ignore"):
        mu = np.diff(E, axis=1)  # mu[:, n-1] = E(n) - E(n-1); inf if n electrons are not bound
    N_gs = np.argmin(E - np.arange(args.n_max + 1)[None, :] * mu_res, axis=1)

    # transition voltages: where the ground-state N changes. Inside that interval mu(N) crosses mu_res;
    # interpolate linearly when mu(N) is finite on both sides, otherwise (N electrons not bound on the
    # low-voltage side) take the midpoint and report the grid spacing as the uncertainty.
    transitions = []
    for i in np.where(np.diff(N_gs) != 0)[0]:
        lo, hi = sorted((N_gs[i], N_gs[i + 1]))
        for n in range(lo + 1, hi + 1):
            m0, m1 = mu[i, n - 1] - mu_res, mu[i + 1, n - 1] - mu_res
            if np.isfinite(m0) and np.isfinite(m1) and m0 != m1:
                transitions.append((n, dVs[i] - m0 * (dVs[i + 1] - dVs[i]) / (m1 - m0), 0.0))
            else:
                transitions.append((n, 0.5 * (dVs[i] + dVs[i + 1]), 0.5 * (dVs[i + 1] - dVs[i])))

    out = Path(args.out or f"results/{Path(args.fem).stem}{'_noimg' if images is None else ''}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{out}.npz", dV=dVs, E=E, mu=mu, mu_res=mu_res, N=N_gs, transitions=np.array(transitions))

    print(f"\nreservoir mu_res = {mu_res * 1e3:.2f} meV (relative to -e V_top)")
    for n, v, dv in transitions:
        print(f"N = {n - 1} -> {n} at V_plunger - V_top = {v:.3f}" + (f" +/- {dv:.3f}" if dv else "") + " V")
    if len(transitions) > 1:
        vt = np.array([v for _, v, _ in transitions])
        print("addition voltages (V):", np.round(np.diff(vt), 3))

    # --- plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    ax.step(dVs, N_gs, where="mid", color="k")
    for n, v, _ in transitions:
        ax.axvline(v, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(r"$V_\mathrm{plunger} - V_\mathrm{top}$ (V)")
    ax.set_ylabel("electrons in dot")
    ax.set_title(f"hole {meta['hole_diameter'] * 1e3:.0f} nm, images {'on' if images else 'off'}")

    ax = axs[1]
    for n in range(1, args.n_max + 1):
        ax.plot(dVs, mu[:, n - 1] * 1e3, label=rf"$\mu({n})$")
    ax.axhline(mu_res * 1e3, color="k", ls="--", label=r"$\mu_\mathrm{res}$")
    ax.set_xlabel(r"$V_\mathrm{plunger} - V_\mathrm{top}$ (V)")
    ax.set_ylabel("addition energy (meV)")
    ax.legend(fontsize=8)

    ax = axs[2]
    x_um = fem["xlist"]
    i0 = np.argmin(np.abs(fem["ylist"]))
    dV_show = dVs[-1]
    U = -dV_show * fem["plunger"][:, i0] * 1e3
    ax.plot(x_um * 1e3, U, label=f"plunger well, dV = {dV_show:.1f} V")
    if images is not None:
        r = np.abs(x_um)
        inside = r <= images.rmax
        Uimg = images.self_energy(r[inside]) * 1e3
        ax.plot(x_um[inside] * 1e3, Uimg, label="self-image energy")
        ax.plot(x_um[inside] * 1e3, U[inside] + Uimg, "k", label="total, 1 electron")
    ax.axvspan(-meta["hole_diameter"] * 500, meta["hole_diameter"] * 500, color="0.93", zorder=0)
    ax.set_xlim(-300, 300)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("potential energy (meV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=150)

    # ground-state configurations at the largest dV
    fig, axs = plt.subplots(1, args.n_max, figsize=(2.4 * args.n_max, 2.6))
    for n, ax in zip(range(1, args.n_max + 1), np.atleast_1d(axs)):
        x, y = r2xy(configs[(n, len(dVs) - 1)])
        ax.add_patch(plt.Circle((0, 0), meta["hole_diameter"] * 500, color="0.9"))
        ax.plot(x * 1e9, y * 1e9, "o", color="tab:green", mec="k")
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_aspect("equal")
        ax.set_title(f"N = {n}" + ("" if np.isfinite(E[-1, n]) else " (unbound)"), fontsize=9)
        ax.set_xlabel("x (nm)")
    fig.suptitle(f"ground states at dV = {dVs[-1]:.1f} V", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}_configs.png", dpi=150)
    print(f"saved {out}.npz, {out}.png, {out}_configs.png")


if __name__ == "__main__":
    main()
