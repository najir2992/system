# Electric Bus Charging Scheduler — Hybrid MILP–ACO (Streamlit)

## 1. Run it

```bash
cd ebus_app
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The browser opens at http://localhost:8501.

## 2. Project structure

```
ebus_app/
├── hybrid_core.py   # ENGINE  – your MILP + ACO + hybrid algorithm (no UI code)
├── charts.py        # VIEW    – Plotly figures (schedule, SOC, utilisation, convergence)
├── app.py           # UI      – Streamlit pages, inputs, Run button, results
└── requirements.txt
```

The algorithm and the interface are kept apart on purpose. The engine can still be
imported in a notebook for the thesis experiments (30 runs, ablation), and the UI only
calls it.

## 3. How the original script maps to the system

| Original script section              | Where it lives now                                   |
|--------------------------------------|------------------------------------------------------|
| 1. Simulation data                   | `default_data()` + **Data input** tab (editable)     |
| 2. Energy requirement                | `ProblemData.required_slots` + "Data check" panel    |
| 3. Hybrid settings                   | `HybridParams` + **sidebar**                         |
| 4. MILP model (HiGHS)                | `HybridScheduler._build_constraints`, `solve_milp`   |
| 5–7. Evaluate / repair / construct   | `evaluate`, `repair`, `construct`                    |
| 8. H1 LP-guided pheromone            | `initial_pheromone`                                  |
| 9. H2 MILP-LNS                       | `choose_buses_to_free`, `milp_lns`                   |
| 10. Pheromone update                 | `update_pheromone`                                   |
| 11. One hybrid run (+ H3 stopping)   | `run(seed, progress_callback)`                       |
| 12. Full MILP reference              | `solve_full_milp` (sidebar checkbox)                 |
| 13–15. 30 runs + statistics          | **Run statistics** tab                               |
| 16. Best run details                 | **Charging schedule** tab                            |
| 17. Feasibility check                | `feasibility_checks` → ✅/❌ list                     |
| 18. Graphs                           | `charts.py`                                          |

The main change from the script is that **global variables were removed**. All the data
sits in a `ProblemData` object and all the settings in a `HybridParams` object, and both
are passed to `HybridScheduler`. A UI needs this, because the user changes the data
between runs.

## 4. How the interface works (Streamlit concepts used)

1. **Rerun model.** Streamlit runs `app.py` from top to bottom every time the user clicks
   or types. Anything that must survive a rerun is kept in `st.session_state`: the input
   tables, the system parameters and the last results.
2. **Data input.** `st.data_editor` turns the bus, price, consumption and availability
   tables into spreadsheet-like grids. Availability is shown as checkboxes.
3. **Validation before solving.** `ProblemData.validate()` catches impossible inputs
   before the solver runs, such as a bus that needs more slots than it is available for,
   or a total demand above charger capacity. If any are found, the Run button is disabled.
4. **Run button → live progress.** `run()` takes a `progress_callback`, which the UI uses
   to update a progress bar and show "iteration k, best RM x, gap y %" while it solves.
5. **Results.** KPI cards (best cost, MILP optimum, LP-bound gap, time), a schedule grid,
   charger utilisation against price, convergence, battery SOC, a per-bus table, a cost
   breakdown and the feasibility checks.
6. **Import/export.** You can download the inputs as an Excel template, edit it, and
   upload it back. The schedule and the run statistics can also be downloaded.

## 5. Suggested development steps (if rebuilding or extending)

1. Refactor the algorithm into functions/classes without globals and check that it
   reproduces the thesis results (RM 692.10 optimum, LP bound RM 683.10).
2. Build a minimal page: hard-coded data, one Run button, print the objective.
3. Add `session_state` and the data editors for the inputs.
4. Add validation and the progress callback.
5. Add the charts and tables one at a time.
6. Add import/export.
7. Optional: deploy free on Streamlit Community Cloud (push to GitHub → share.streamlit.io).

## 6. Ideas for further extension

- **Fuzzy MCDM tab:** rank alternative schedules (e.g. Hybrid vs ACO vs MILP), or rank
  charging policies on cost, time, robustness and charger load.
- **Ablation tab:** run Plain ACO, H1 only, H2 only and full Hybrid side by side.
- **Variable charging power** (x[i,t] continuous) or time-of-use tariffs read from a CSV.
- **Caching:** wrap the full-MILP solve in `@st.cache_data` for large instances.
