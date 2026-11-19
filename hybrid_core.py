"""
hybrid_core.py
==============
Computational engine for the Hybrid MILP-ACO electric bus charging scheduler.

This is the same algorithm as the original script (H1 LP-guided pheromone,
H2 MILP fix-and-optimise LNS, H3 LP lower bound / stopping rule), refactored
so that:
  * there are NO global variables - all data and parameters are passed in,
  * the Streamlit interface (app.py) can change the input data at run time,
  * a progress callback can report each iteration to the user interface.

The engine has no Streamlit code, so it can still be used from a notebook
or a plain script for the thesis experiments.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


# =============================================================
# 1. INPUT DATA AND PARAMETERS
# =============================================================

@dataclass
class ProblemData:
    """All problem inputs (one row per bus, one column per time slot)."""
    initial_energy: np.ndarray          # (B,)   kWh
    target_energy: np.ndarray           # (B,)   kWh
    energy_consumption: np.ndarray      # (B, T) kWh per slot
    bus_availability: np.ndarray        # (B, T) 1 = can charge, 0 = on route
    electricity_price: np.ndarray       # (T,)   RM/kWh
    num_chargers: int = 6
    charger_power: float = 30.0         # kWh per charging slot
    battery_capacity: float = 120.0     # kWh
    min_energy: float = 20.0            # kWh
    degradation_cost_per_kWh: float = 0.05
    bus_names: list[str] | None = None
    slot_names: list[str] | None = None

    def __post_init__(self):
        self.initial_energy = np.asarray(self.initial_energy, dtype=float)
        self.target_energy = np.asarray(self.target_energy, dtype=float)
        self.energy_consumption = np.asarray(self.energy_consumption, dtype=float)
        self.bus_availability = np.asarray(self.bus_availability, dtype=int)
        self.electricity_price = np.asarray(self.electricity_price, dtype=float)
        if self.bus_names is None:
            self.bus_names = [f"EB{i+1}" for i in range(self.num_buses)]
        if self.slot_names is None:
            self.slot_names = [f"t{t+1}" for t in range(self.num_slots)]

    @property
    def num_buses(self) -> int:
        return self.energy_consumption.shape[0]

    @property
    def num_slots(self) -> int:
        return self.energy_consumption.shape[1]

    @property
    def energy_needed(self) -> np.ndarray:
        return (self.energy_consumption.sum(axis=1)
                + self.target_energy - self.initial_energy)

    @property
    def required_slots(self) -> np.ndarray:
        return np.maximum(0, np.ceil(self.energy_needed / self.charger_power)).astype(int)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems with the data (empty = OK)."""
        errors = []
        B, T = self.num_buses, self.num_slots
        if self.bus_availability.shape != (B, T):
            errors.append(f"Availability matrix must be {B} x {T}.")
        if self.electricity_price.shape != (T,):
            errors.append(f"Electricity price must have {T} values.")
        if self.initial_energy.shape != (B,) or self.target_energy.shape != (B,):
            errors.append(f"Initial/target energy must have {B} values.")
        if not np.isin(self.bus_availability, [0, 1]).all():
            errors.append("Availability values must be 0 or 1.")
        if (self.energy_consumption < 0).any():
            errors.append("Energy consumption cannot be negative.")
        if (self.initial_energy > self.battery_capacity).any():
            errors.append("Some initial energy values exceed battery capacity.")
        if errors:
            return errors
        req, avail = self.required_slots, self.bus_availability.sum(axis=1)
        for i in np.where(req > avail)[0]:
            errors.append(f"{self.bus_names[i]} needs {req[i]} slots but is "
                          f"available in only {avail[i]}.")
        if req.sum() > self.num_chargers * T:
            errors.append(f"Total required slots ({req.sum()}) exceed charger "
                          f"capacity ({self.num_chargers * T}).")
        return errors


@dataclass
class HybridParams:
    """ACO + hybrid settings (defaults = values used in the thesis)."""
    num_ants: int = 30
    num_iterations: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.3
    Q: float = 100.0
    penalty_weight: float = 10000.0
    use_lp_init: bool = True          # (H1)
    use_milp_lns: bool = True         # (H2)
    lp_weight: float = 2.0
    tau_min: float = 0.05
    tau_max: float = 5.0
    lns_every: int = 1
    lns_moves: int = 3
    lns_bus_fraction: float = 0.3     # lns_buses = max(3, ceil(f * B))
    sub_milp_time_limit: float = 5.0
    patience: int = 20
    gap_tolerance: float = 1e-6

    @property
    def mode(self) -> str:
        return "HYBRID MILP-ACO" if (self.use_lp_init or self.use_milp_lns) else "PLAIN ACO"


# =============================================================
# 2. THE SCHEDULER
# =============================================================

class HybridScheduler:
    def __init__(self, data: ProblemData, params: HybridParams | None = None):
        self.d = data
        self.p = params or HybridParams()
        d = self.d
        self.B, self.T = d.num_buses, d.num_slots
        self.required_slots = d.required_slots
        self.heuristic = 1.0 / d.electricity_price
        self.n_var = self.B * self.T
        self.cost_per_slot = (d.electricity_price + d.degradation_cost_per_kWh) * d.charger_power
        self.c_vec = np.tile(self.cost_per_slot, self.B)
        self.lns_buses = min(self.B, max(3, math.ceil(self.p.lns_bus_fraction * self.B)))
        self.constraints = self._build_constraints()
        self.ub_avail = d.bus_availability.reshape(-1).astype(float)
        self.X_LP = self.LP_BOUND = self.LP_TIME = None

    # ---------- MILP model (section 4 of the original) ----------
    def _vi(self, i, t):
        return i * self.T + t

    def _build_constraints(self):
        d, B, T = self.d, self.B, self.T
        rows, lo, hi = [], [], []
        cum = np.cumsum(d.energy_consumption, axis=1)
        # (a) SOC limits after every slot + final target energy
        for i in range(B):
            for t in range(T):
                row = np.zeros(self.n_var)
                row[self._vi(i, 0):self._vi(i, t) + 1] = d.charger_power
                lower = d.min_energy
                if t == T - 1:
                    lower = max(d.min_energy, d.target_energy[i])
                rows.append(row)
                lo.append(lower - d.initial_energy[i] + cum[i, t])
                hi.append(d.battery_capacity - d.initial_energy[i] + cum[i, t])
        # (b) charger capacity per slot
        for t in range(T):
            row = np.zeros(self.n_var)
            row[[self._vi(i, t) for i in range(B)]] = 1
            rows.append(row); lo.append(-np.inf); hi.append(d.num_chargers)
        # (c) required charging slots per bus
        for i in range(B):
            row = np.zeros(self.n_var)
            row[self._vi(i, 0):self._vi(i, T - 1) + 1] = 1
            rows.append(row); lo.append(self.required_slots[i]); hi.append(np.inf)
        return LinearConstraint(csr_matrix(np.array(rows)), np.array(lo), np.array(hi))

    def solve_milp(self, fixed=None, relax=False, time_limit=None):
        lb = np.zeros(self.n_var)
        ub = self.ub_avail.copy()
        if fixed is not None:
            f = fixed.reshape(-1)
            mask = f >= 0
            lb[mask] = f[mask]
            ub[mask] = f[mask]
        integrality = np.zeros(self.n_var) if relax else np.ones(self.n_var)
        options = {"disp": False, "mip_rel_gap": 0.0}
        if time_limit is not None:
            options["time_limit"] = time_limit
        res = milp(c=self.c_vec, integrality=integrality, bounds=Bounds(lb, ub),
                   constraints=self.constraints, options=options)
        if res.x is None:
            return None, np.inf, res.status
        x = res.x.reshape(self.B, self.T)
        if not relax:
            x = np.rint(x).astype(int)
        return x, res.fun, res.status

    def solve_lp_bound(self):
        t0 = time.perf_counter()
        self.X_LP, self.LP_BOUND, _ = self.solve_milp(relax=True)
        self.LP_TIME = time.perf_counter() - t0
        return self.LP_BOUND

    def solve_full_milp(self):
        t0 = time.perf_counter()
        x, obj, status = self.solve_milp()
        return x, obj, status, time.perf_counter() - t0

    # ---------- Evaluation (section 5) ----------
    def evaluate(self, schedule):
        d, p = self.d, self.p
        B, T = self.B, self.T
        charging_cost = degradation_cost = penalty = 0.0
        feasible = True
        ep = np.zeros((B, T + 1))
        ep[:, 0] = d.initial_energy
        for t in range(T):
            for i in range(B):
                ep[i, t + 1] = ep[i, t] + d.charger_power * schedule[i, t] - d.energy_consumption[i, t]
                if schedule[i, t] == 1:
                    charging_cost += d.charger_power * d.electricity_price[t]
                    degradation_cost += d.degradation_cost_per_kWh * d.charger_power
                if ep[i, t + 1] < d.min_energy:
                    penalty += p.penalty_weight * (d.min_energy - ep[i, t + 1]); feasible = False
                if ep[i, t + 1] > d.battery_capacity:
                    penalty += p.penalty_weight * (ep[i, t + 1] - d.battery_capacity); feasible = False
        fe = ep[:, -1]
        for i in range(B):
            if fe[i] < d.target_energy[i]:
                penalty += p.penalty_weight * (d.target_energy[i] - fe[i]); feasible = False
            actual = schedule[i].sum()
            if actual < self.required_slots[i]:
                penalty += p.penalty_weight * (self.required_slots[i] - actual); feasible = False
        for t in range(T):
            count = schedule[:, t].sum()
            if count > d.num_chargers:
                penalty += p.penalty_weight * (count - d.num_chargers); feasible = False
        viol = schedule > d.bus_availability
        if np.any(viol):
            penalty += p.penalty_weight * viol.sum(); feasible = False
        obj = charging_cost + degradation_cost
        return (obj + penalty, charging_cost, degradation_cost, obj, fe, ep, penalty, feasible)

    def bus_is_feasible(self, schedule, i):
        d = self.d
        e = d.initial_energy[i]
        for t in range(self.T):
            e = e + d.charger_power * schedule[i, t] - d.energy_consumption[i, t]
            if e < d.min_energy or e > d.battery_capacity:
                return False
        if e < d.target_energy[i] or schedule[i].sum() < self.required_slots[i]:
            return False
        return not np.any(schedule[i] > d.bus_availability[i])

    # ---------- Repair (section 6) ----------
    def repair(self, schedule):
        d = self.d
        B, T = self.B, self.T
        rep = schedule.copy()
        usage = rep.sum(axis=0).astype(int)
        for i in range(B):
            missing = max(0, self.required_slots[i] - rep[i].sum())
            if missing > 0:
                cand = [t for t in range(T) if d.bus_availability[i, t] == 1
                        and rep[i, t] == 0 and usage[t] < d.num_chargers]
                cand = sorted(cand, key=lambda t: d.electricity_price[t])
                for t in cand[:missing]:
                    rep[i, t] = 1; usage[t] += 1
        for _ in range(10):
            change = False
            for i in range(B):
                energy = d.initial_energy[i]
                for t in range(T):
                    energy = energy + d.charger_power * rep[i, t] - d.energy_consumption[i, t]
                    if energy < d.min_energy:
                        cand = [k for k in range(t + 1) if d.bus_availability[i, k] == 1
                                and rep[i, k] == 0 and usage[k] < d.num_chargers]
                        if cand:
                            new = min(cand, key=lambda k: d.electricity_price[k])
                            later = [s for s in range(t + 1, T) if rep[i, s] == 1]
                            if later:
                                old = max(later, key=lambda s: d.electricity_price[s])
                                rep[i, old] = 0; usage[old] -= 1
                            rep[i, new] = 1; usage[new] += 1
                            change = True
                            break
                if change:
                    break
            if not change:
                break
        return rep

    # ---------- ACO construction (section 7) ----------
    def construct(self, pher):
        d, p = self.d, self.p
        sched = np.zeros((self.B, self.T), dtype=int)
        usage = np.zeros(self.T, dtype=int)
        order = list(range(self.B))
        random.shuffle(order)
        for i in order:
            selected = []
            for _ in range(self.required_slots[i]):
                feas = [t for t in range(self.T) if d.bus_availability[i, t] == 1
                        and t not in selected and usage[t] < d.num_chargers]
                if not feas:
                    break
                attr = np.array([(pher[i, t] ** p.alpha) * (self.heuristic[t] ** p.beta) for t in feas])
                slot = int(np.random.choice(feas, p=attr / attr.sum()))
                sched[i, slot] = 1; usage[slot] += 1; selected.append(slot)
        return self.repair(sched)

    # ---------- (H1) pheromone initialisation (section 8) ----------
    def initial_pheromone(self):
        p = self.p
        if p.use_lp_init and self.X_LP is not None:
            tau = 1.0 + p.lp_weight * self.X_LP
        else:
            tau = np.ones((self.B, self.T))
        return np.clip(tau, p.tau_min, p.tau_max)

    # ---------- (H2) MILP-LNS (section 9) ----------
    def choose_buses_to_free(self, schedule, k):
        bad = [i for i in range(self.B) if not self.bus_is_feasible(schedule, i)]
        chosen = bad[:k]
        rem = [i for i in range(self.B) if i not in chosen]
        if len(chosen) < k and rem:
            cost = np.array([(schedule[i] * self.cost_per_slot).sum() for i in rem]) + 1e-6
            extra = np.random.choice(rem, size=min(k - len(chosen), len(rem)),
                                     replace=False, p=cost / cost.sum())
            chosen += list(extra)
        return chosen

    def milp_lns(self, schedule):
        best = schedule.copy()
        best_fit = self.evaluate(best)[0]
        n = 0
        for _ in range(self.p.lns_moves):
            free = self.choose_buses_to_free(best, self.lns_buses)
            fixed = best.copy()
            fixed[free, :] = -1
            x, _, _ = self.solve_milp(fixed=fixed, time_limit=self.p.sub_milp_time_limit)
            n += 1
            if x is None:
                continue
            fit = self.evaluate(x)[0]
            if fit < best_fit - 1e-9:
                best, best_fit = x, fit
        return best, n

    # ---------- pheromone update (section 10) ----------
    def update_pheromone(self, pher, ib, ib_obj, gb, gb_obj):
        p = self.p
        pher = (1 - p.rho) * pher
        if ib is not None:
            pher += ib * (p.Q / max(ib_obj, 1e-10))
        if gb is not None:
            pher += gb * (p.Q / max(gb_obj, 1e-10))
        return np.clip(pher, p.tau_min, p.tau_max)

    # ---------- one complete hybrid run (section 11) ----------
    def run(self, seed: int, progress_callback=None):
        """progress_callback(iteration, num_iterations, best_objective, gap) is
        called after every iteration so the UI can show live progress."""
        p = self.p
        if self.LP_BOUND is None:
            self.solve_lp_bound()
        np.random.seed(seed); random.seed(seed)
        pher = self.initial_pheromone()
        best = dict(objective=np.inf, schedule=None, charging_cost=None,
                    degradation_cost=None, energy_profile=None,
                    final_energy=None, penalty=None)
        history, sub_milps, no_improve = [], 0, 0
        start = time.perf_counter()

        for it in range(p.num_iterations):
            ants = [self.construct(pher) for _ in range(p.num_ants)]
            evals = [self.evaluate(s) for s in ants]
            ib = ants[int(np.argmin([e[0] for e in evals]))]
            if p.use_milp_lns and it % p.lns_every == 0:
                ib, n = self.milp_lns(ib)
                sub_milps += n
            fit, cc, dc, obj, fe, ep, pen, feas = self.evaluate(ib)
            improved = False
            if feas and obj < best["objective"] - 1e-9:
                best.update(objective=obj, schedule=ib.copy(), charging_cost=cc,
                            degradation_cost=dc, energy_profile=ep.copy(),
                            final_energy=fe.copy(), penalty=pen)
                improved = True
            pher = self.update_pheromone(pher, ib if feas else None, obj,
                                         best["schedule"], best["objective"])
            history.append(best["objective"] if best["schedule"] is not None else np.nan)

            no_improve = 0 if improved else no_improve + 1
            gap = np.nan
            if best["schedule"] is not None:
                gap = (best["objective"] - self.LP_BOUND) / best["objective"]
            if progress_callback:
                progress_callback(it + 1, p.num_iterations, best["objective"], gap)
            if best["schedule"] is not None and gap <= p.gap_tolerance:
                break
            if no_improve >= p.patience:
                break

        best.update(seed=seed,
                    computation_time=time.perf_counter() - start + self.LP_TIME,
                    history=history, iterations_used=len(history),
                    sub_milps=sub_milps,
                    lb_gap=(best["objective"] - self.LP_BOUND) / best["objective"] * 100
                    if best["schedule"] is not None else np.nan)
        return best

    # ---------- feasibility report (section 17) ----------
    def feasibility_checks(self, schedule, energy_profile):
        d = self.d
        util = schedule.sum(axis=0)
        return {
            "Minimum energy satisfied": bool(energy_profile.min() >= d.min_energy),
            "Maximum energy satisfied": bool(energy_profile.max() <= d.battery_capacity),
            "Final energy satisfied": bool(np.all(energy_profile[:, -1] >= d.target_energy)),
            "Charger capacity satisfied": bool(np.all(util <= d.num_chargers)),
            "Bus availability satisfied": bool(np.all(schedule <= d.bus_availability)),
            "Required charging slots satisfied": bool(np.all(schedule.sum(axis=1) >= self.required_slots)),
        }


# =============================================================
# 3. DEFAULT DATA (identical to the MILP and ACO chapters)
# =============================================================

def default_data() -> ProblemData:
    return ProblemData(
        initial_energy=[40, 50, 35, 60, 45, 55, 38, 42, 48, 52, 36, 58, 41, 47, 53],
        target_energy=[40, 50, 35, 60, 45, 55, 38, 42, 48, 52, 36, 58, 41, 47, 53],
        energy_consumption=[
            [0, 0, 15, 15, 20, 20, 0, 0], [0, 0, 20, 20, 15, 15, 0, 0],
            [0, 10, 10, 15, 15, 10, 0, 0], [0, 15, 15, 20, 20, 15, 0, 0],
            [0, 10, 15, 15, 15, 10, 0, 0], [0, 0, 18, 18, 20, 18, 0, 0],
            [0, 8, 12, 14, 18, 12, 0, 0], [0, 10, 15, 18, 18, 10, 0, 0],
            [0, 12, 14, 16, 18, 12, 0, 0], [0, 15, 18, 18, 20, 15, 0, 0],
            [0, 10, 12, 14, 16, 10, 0, 0], [0, 15, 15, 20, 20, 15, 0, 0],
            [0, 8, 10, 15, 15, 10, 0, 0], [0, 10, 14, 16, 18, 12, 0, 0],
            [0, 12, 15, 18, 20, 15, 0, 0]],
        bus_availability=[
            [1, 1, 1, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 0, 1, 1]],
        electricity_price=[0.50, 0.55, 0.65, 0.70, 0.45, 0.40, 0.39, 0.38],
        num_chargers=6, charger_power=30, battery_capacity=120,
        min_energy=20, degradation_cost_per_kWh=0.05,
    )
