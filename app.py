"""
app.py - Streamlit interface for the Hybrid MILP-ACO electric bus charging scheduler.

Run with:   streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

import charts
from hybrid_core import HybridParams, HybridScheduler, ProblemData, default_data

st.set_page_config(page_title="E-Bus Charging Scheduler", page_icon="⚡", layout="wide")


# =============================================================
# 1. SESSION STATE: holds the input tables between reruns
# =============================================================
# Streamlit re-runs this whole script on every click, so anything that must
# survive (input tables, results) lives in st.session_state.

def data_to_frames(d: ProblemData) -> dict:
    return {
        "buses": pd.DataFrame({"Bus": d.bus_names,
                               "Initial Energy (kWh)": d.initial_energy,
                               "Target Energy (kWh)": d.target_energy}),
        "prices": pd.DataFrame({"Slot": d.slot_names,
                                "Price (RM/kWh)": d.electricity_price}),
        "consumption": pd.DataFrame(d.energy_consumption, index=d.bus_names,
                                    columns=d.slot_names),
        "availability": pd.DataFrame(d.bus_availability.astype(bool), index=d.bus_names,
                                     columns=d.slot_names),
    }


def load_into_state(d: ProblemData):
    st.session_state.frames = data_to_frames(d)
    st.session_state.update({
        "n_buses": d.num_buses, "n_slots": d.num_slots,
        "sys_num_chargers": int(d.num_chargers), "sys_charger_power": float(d.charger_power),
        "sys_battery_capacity": float(d.battery_capacity), "sys_min_energy": float(d.min_energy),
        "sys_degradation_cost_per_kWh": float(d.degradation_cost_per_kWh)})
    # Changing the editor key forces the data editors to show the new tables
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.pop("results", None)


def resize(frames: dict, n_buses: int, n_slots: int) -> dict:
    """Grow/shrink the input tables when the number of buses or slots changes.
    Existing values are kept; new buses/slots get simple default values."""
    old_b = frames["buses"]["Bus"].astype(str).tolist()
    old_s = frames["prices"]["Slot"].astype(str).tolist()
    buses = (old_b + [f"EB{i+1}" for i in range(len(old_b), n_buses)])[:n_buses]
    slots = (old_s + [f"t{t+1}" for t in range(len(old_s), n_slots)])[:n_slots]

    b = frames["buses"].reset_index(drop=True).reindex(range(n_buses))
    b["Bus"] = buses
    b = b.fillna({"Initial Energy (kWh)": 50.0, "Target Energy (kWh)": 50.0})
    p = frames["prices"].reset_index(drop=True).reindex(range(n_slots))
    p["Slot"] = slots
    p = p.fillna({"Price (RM/kWh)": 0.50})

    def grid(df, fill):
        g = pd.DataFrame(fill, index=buses, columns=slots, dtype=type(fill))
        r, k = min(len(df), n_buses), min(df.shape[1], n_slots)
        g.iloc[:r, :k] = df.to_numpy()[:r, :k]
        return g

    return {"buses": b, "prices": p,
            "consumption": grid(frames["consumption"], 0.0),
            "availability": grid(frames["availability"].astype(bool), True)}


def frames_to_data(f: dict, s: dict) -> ProblemData:
    return ProblemData(
        initial_energy=f["buses"]["Initial Energy (kWh)"].to_numpy(float),
        target_energy=f["buses"]["Target Energy (kWh)"].to_numpy(float),
        energy_consumption=f["consumption"].to_numpy(float),
        bus_availability=f["availability"].to_numpy().astype(int),
        electricity_price=f["prices"]["Price (RM/kWh)"].to_numpy(float),
        bus_names=f["buses"]["Bus"].astype(str).tolist(),
        slot_names=f["prices"]["Slot"].astype(str).tolist(),
        **s)


def frames_to_excel(f: dict, s: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(list(s.items()), columns=["Parameter", "Value"]).to_excel(xw, sheet_name="System", index=False)
        f["buses"].to_excel(xw, sheet_name="Buses", index=False)
        f["prices"].to_excel(xw, sheet_name="Prices", index=False)
        f["consumption"].to_excel(xw, sheet_name="Consumption")
        f["availability"].astype(int).to_excel(xw, sheet_name="Availability")
    return buf.getvalue()


def excel_to_data(file) -> ProblemData:
    x = pd.read_excel(file, sheet_name=None)
    sysd = dict(zip(x["System"]["Parameter"], x["System"]["Value"]))
    buses, prices = x["Buses"], x["Prices"]
    cons = x["Consumption"].set_index(x["Consumption"].columns[0])
    avail = x["Availability"].set_index(x["Availability"].columns[0])
    return ProblemData(
        initial_energy=buses.iloc[:, 1], target_energy=buses.iloc[:, 2],
        energy_consumption=cons.to_numpy(float), bus_availability=avail.to_numpy(int),
        electricity_price=prices.iloc[:, 1],
        bus_names=buses.iloc[:, 0].astype(str).tolist(),
        slot_names=prices.iloc[:, 0].astype(str).tolist(),
        num_chargers=int(sysd["num_chargers"]), charger_power=float(sysd["charger_power"]),
        battery_capacity=float(sysd["battery_capacity"]), min_energy=float(sysd["min_energy"]),
        degradation_cost_per_kWh=float(sysd["degradation_cost_per_kWh"]))


if "frames" not in st.session_state:
    load_into_state(default_data())


# =============================================================
# 2. SIDEBAR: algorithm settings
# =============================================================
with st.sidebar:
    st.header("⚙️ Algorithm settings")
    method = st.radio("Method", ["Hybrid MILP-ACO", "Plain ACO", "Custom (ablation)"],
                      help="Plain ACO switches off both hybrid components (H1 and H2).")
    if method == "Custom (ablation)":
        use_lp = st.checkbox("H1: LP-guided pheromone initialisation", True)
        use_lns = st.checkbox("H2: MILP fix-and-optimise LNS", True)
    else:
        use_lp = use_lns = method == "Hybrid MILP-ACO"

    st.subheader("ACO")
    num_ants = st.slider("Number of ants", 5, 100, 30, 5)
    num_iterations = st.slider("Max iterations", 10, 300, 100, 10)
    c1, c2 = st.columns(2)
    alpha = c1.number_input("α (pheromone)", 0.0, 5.0, 1.0, 0.1)
    beta = c2.number_input("β (heuristic)", 0.0, 5.0, 2.0, 0.1)
    rho = c1.number_input("ρ (evaporation)", 0.01, 0.99, 0.3, 0.05)
    Q = c2.number_input("Q", 1.0, 10000.0, 100.0, 10.0)
    patience = st.number_input("Patience (iterations without improvement)", 1, 200, 20)

    with st.expander("Hybrid parameters"):
        lp_weight = st.number_input("LP weight (H1)", 0.0, 10.0, 2.0, 0.5)
        lns_moves = st.number_input("Sub-MILPs per LNS call (H2)", 1, 20, 3)
        lns_frac = st.slider("Share of buses freed per sub-MILP", 0.1, 1.0, 0.3, 0.05)
        sub_tl = st.number_input("Sub-MILP time limit (s)", 0.5, 60.0, 5.0, 0.5)
        tau_min = st.number_input("τ min", 0.001, 1.0, 0.05, 0.01)
        tau_max = st.number_input("τ max", 1.0, 50.0, 5.0, 0.5)

    st.subheader("Experiment")
    num_runs = st.number_input("Independent runs", 1, 50, 5,
                               help="The thesis used 30 runs; fewer runs is quicker for a demo.")
    base_seed = st.number_input("Base seed", 0, 10_000, 100)
    compare_milp = st.checkbox("Also solve the full MILP (exact optimum)", True)

params = HybridParams(num_ants=num_ants, num_iterations=num_iterations, alpha=alpha,
                      beta=beta, rho=rho, Q=Q, patience=int(patience),
                      use_lp_init=use_lp, use_milp_lns=use_lns, lp_weight=lp_weight,
                      lns_moves=int(lns_moves), lns_bus_fraction=lns_frac,
                      sub_milp_time_limit=sub_tl, tau_min=tau_min, tau_max=tau_max)


# =============================================================
# 3. MAIN PAGE
# =============================================================
st.title("⚡ Electric Bus Charging Scheduler")
st.caption("Hybrid MILP–ACO matheuristic · LP-guided pheromone (H1) · "
           "MILP fix-and-optimise LNS (H2) · LP lower-bound certificate (H3)")

tab_input, tab_schedule, tab_stats, tab_about = st.tabs(
    ["📥 Data input", "🗓️ Charging schedule", "📊 Run statistics", "ℹ️ Method"])

# ---------------- TAB 1: DATA INPUT ----------------
with tab_input:
    top = st.columns([1, 1, 1, 2])
    if top[0].button("↺ Load thesis data"):
        load_into_state(default_data()); st.rerun()
    up = top[3].file_uploader("Upload Excel input (same layout as the template)", type=["xlsx"])
    if up is not None and st.session_state.get("last_upload") != up.file_id:
        try:
            load_into_state(excel_to_data(up))
            st.session_state.last_upload = up.file_id
            st.rerun()
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    st.subheader("System parameters")
    f = st.session_state.frames
    c = st.columns(7)
    c[0].number_input("Buses", 1, 100, key="n_buses")
    c[1].number_input("Time slots", 2, 96, key="n_slots")
    c[2].number_input("Chargers", 1, 100, key="sys_num_chargers")
    c[3].number_input("kWh per slot", 1.0, 500.0, key="sys_charger_power")
    c[4].number_input("Battery cap. (kWh)", 10.0, 1000.0, key="sys_battery_capacity")
    c[5].number_input("Min energy (kWh)", 0.0, 500.0, key="sys_min_energy")
    c[6].number_input("Degradation RM/kWh", 0.0, 5.0, step=0.01,
                      key="sys_degradation_cost_per_kWh")
    s = {k[4:]: st.session_state[k] for k in st.session_state.keys() if str(k).startswith("sys_")}

    v = st.session_state.editor_version
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Buses")
        buses = st.data_editor(f["buses"], key=f"buses_{v}", hide_index=True,
                               disabled=["Bus"])
    with right:
        st.subheader("Electricity price per slot")
        prices = st.data_editor(f["prices"], key=f"prices_{v}", hide_index=True)
    st.subheader("Energy consumption per slot (kWh)")
    consumption = st.data_editor(f["consumption"], key=f"cons_{v}")
    st.subheader("Bus availability for charging (✔ = at depot, can charge)")
    availability = st.data_editor(f["availability"], key=f"avail_{v}")

    edited = {"buses": buses, "prices": prices, "consumption": consumption,
              "availability": availability}
    if st.session_state.n_buses != len(buses) or st.session_state.n_slots != len(prices):
        st.session_state.frames = resize(edited, st.session_state.n_buses, st.session_state.n_slots)
        st.session_state.editor_version += 1
        st.session_state.pop("results", None)
        st.rerun()
    data = frames_to_data(edited, s)

    st.download_button("⬇️ Download these inputs as an Excel template",
                       frames_to_excel(edited, s), "ebus_input.xlsx")

    st.subheader("Data check")
    errors = data.validate()
    m = st.columns(3)
    m[0].metric("Required charging slots", int(data.required_slots.sum()))
    m[1].metric("Charger capacity (slots)", data.num_chargers * data.num_slots)
    m[2].metric("Capacity utilisation",
                f"{data.required_slots.sum() / (data.num_chargers * data.num_slots):.0%}")
    if errors:
        for e in errors:
            st.error(e)
    else:
        st.success("Input data is consistent. Go ahead and run the scheduler.")

    run_clicked = st.button(f"🚀 Run {params.mode}", type="primary",
                            disabled=bool(errors))

# ---------------- RUN ----------------
if run_clicked:
    sched = HybridScheduler(data, params)
    with st.status("Solving…", expanded=True) as status:
        st.write("Solving LP relaxation (lower bound)…")
        sched.solve_lp_bound()
        milp_res = None
        if compare_milp:
            st.write("Solving full MILP for reference…")
            milp_res = sched.solve_full_milp()
        bar = st.progress(0.0)
        info = st.empty()
        results = []
        for r in range(int(num_runs)):
            def cb(it, total, best_obj, gap, r=r):
                bar.progress((r + it / total) / num_runs)
                txt = f"RM {best_obj:.2f}, gap {gap*100:.3f}%" if np.isfinite(best_obj) else "no feasible yet"
                info.write(f"Run {r+1}/{num_runs} · iteration {it} · best {txt}")
            results.append(sched.run(int(base_seed) + r, progress_callback=cb))
        bar.progress(1.0)
        status.update(label="Done", state="complete", expanded=False)
    st.session_state.results = dict(results=results, scheduler=sched, milp=milp_res,
                                    mode=params.mode)
    st.toast("Schedule ready – open the 'Charging schedule' tab.", icon="✅")

# ---------------- TAB 2: SCHEDULE ----------------
with tab_schedule:
    R = st.session_state.get("results")
    if R is None:
        st.info("Set up the data and press **Run** in the Data input tab.")
    else:
        sched, results = R["scheduler"], R["results"]
        d = sched.d
        objs = [r["objective"] for r in results]
        if not np.isfinite(objs).any():
            st.error("No feasible schedule was found. Try more iterations or check the data.")
        else:
            bi = int(np.nanargmin(objs))
            best = results[bi]
            milp_opt = R["milp"][1] if R["milp"] else None

            k = st.columns(5)
            k[0].metric(f"Best cost ({R['mode']})", f"RM {best['objective']:.2f}")
            if milp_opt is not None and np.isfinite(milp_opt):
                k[1].metric("MILP optimum", f"RM {milp_opt:.2f}",
                            f"gap {(best['objective'] - milp_opt) / milp_opt * 100:.3f}%",
                            delta_color="inverse")
            k[2].metric("LP lower bound", f"RM {sched.LP_BOUND:.2f}",
                        f"gap {best['lb_gap']:.3f}%", delta_color="off")
            k[3].metric("Iterations used", best["iterations_used"])
            k[4].metric("Time", f"{best['computation_time']:.2f} s")

            st.plotly_chart(charts.schedule_heatmap(best["schedule"], d.bus_availability,
                                                    d.bus_names, d.slot_names,
                                                    d.electricity_price))

            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.charger_utilisation_chart(best["schedule"], d.slot_names,
                                                             d.electricity_price, d.num_chargers))
            c2.plotly_chart(charts.convergence_chart(results, bi, milp_opt, sched.LP_BOUND))
            st.plotly_chart(charts.energy_profile_chart(best["energy_profile"], d.bus_names,
                                                        d.slot_names, d.min_energy,
                                                        d.battery_capacity))

            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Per-bus summary")
                bus_df = pd.DataFrame({
                    "Bus": d.bus_names, "Initial (kWh)": d.initial_energy,
                    "Target (kWh)": d.target_energy, "Final (kWh)": best["final_energy"],
                    "Required slots": sched.required_slots,
                    "Slots used": best["schedule"].sum(axis=1),
                    "Charging slots": [", ".join(np.array(d.slot_names)[row == 1])
                                       for row in best["schedule"]]})
                st.dataframe(bus_df, hide_index=True)
            with c2:
                st.subheader("Cost summary")
                st.dataframe(pd.DataFrame({
                    "Component": ["Charging cost", "Battery degradation", "Total"],
                    "RM": [best["charging_cost"], best["degradation_cost"], best["objective"]]
                }).round(2), hide_index=True)
                st.subheader("Feasibility check")
                for name, ok in sched.feasibility_checks(best["schedule"],
                                                         best["energy_profile"]).items():
                    st.write(("✅ " if ok else "❌ ") + name)

            sched_df = pd.DataFrame(best["schedule"], index=d.bus_names, columns=d.slot_names)
            soc_df = pd.DataFrame(best["energy_profile"], index=d.bus_names,
                                  columns=["Initial"] + d.slot_names).round(2)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                sched_df.to_excel(xw, sheet_name="Schedule"); soc_df.to_excel(xw, sheet_name="Energy profile")
                bus_df.to_excel(xw, sheet_name="Bus summary", index=False)
            st.download_button("⬇️ Download schedule (Excel)", buf.getvalue(),
                               "charging_schedule.xlsx")

# ---------------- TAB 3: STATISTICS ----------------
with tab_stats:
    R = st.session_state.get("results")
    if R is None:
        st.info("No results yet.")
    else:
        results = R["results"]
        milp_opt = R["milp"][1] if R["milp"] else None
        df = pd.DataFrame([{
            "Run": n, "Seed": r["seed"], "Charging Cost RM": r["charging_cost"],
            "Degradation Cost RM": r["degradation_cost"], "Objective Cost RM": r["objective"],
            "Gap to LP bound %": r["lb_gap"], "Iterations": r["iterations_used"],
            "Sub-MILPs": r["sub_milps"], "Computation Time s": r["computation_time"],
        } for n, r in enumerate(results, start=1)])
        valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Objective Cost RM"])
        obj = valid["Objective Cost RM"]
        k = st.columns(5)
        k[0].metric("Best", f"RM {obj.min():.2f}")
        k[1].metric("Mean", f"RM {obj.mean():.2f}")
        k[2].metric("Worst", f"RM {obj.max():.2f}")
        k[3].metric("Std. dev.", f"{obj.std():.4f}" if len(obj) > 1 else "–")
        if milp_opt is not None:
            k[4].metric("Runs reaching optimum",
                        f"{np.isclose(obj, milp_opt).mean() * 100:.0f}%")
        c1, c2 = st.columns([2, 1])
        c1.dataframe(df.round(4), hide_index=True)
        c2.plotly_chart(charts.runs_boxplot(valid, milp_opt))
        if R["milp"]:
            st.caption(f"Full MILP solved in {R['milp'][3]:.3f} s · "
                       f"mean hybrid time {valid['Computation Time s'].mean():.3f} s")
        st.download_button("⬇️ Download run statistics (CSV)", df.to_csv(index=False),
                           "hybrid_runs.csv")

# ---------------- TAB 4: METHOD ----------------
with tab_about:
    st.markdown("""
**Problem.** Assign each bus to charging slots so that battery energy stays between the
minimum and capacity, each bus ends at its target energy, the number of chargers is never
exceeded and buses charge only when they are at the depot, at minimum electricity plus
degradation cost.

**Hybrid MILP–ACO.**
- **H1 – LP-guided pheromone.** The LP relaxation is solved once; its fractional values
  bias the initial pheromone towards (bus, slot) pairs the exact model prefers.
- **H2 – MILP fix-and-optimise.** Each iteration, the best ant's schedule is improved by
  freeing a few buses and re-optimising them exactly with HiGHS.
- **H3 – LP lower bound.** Gives every solution a proven optimality gap and stops the run
  once the gap is closed or after *patience* iterations without improvement.

Choose **Plain ACO** or **Custom** in the sidebar to reproduce the ablation study.
""")
