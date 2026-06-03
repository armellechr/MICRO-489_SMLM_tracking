import matplotlib
import nbformat

matplotlib.use("Agg")

NOTEBOOK = "MICRO-489_SMLM_tracking/Experiments/CostExperiment.ipynb"
CELL_ORDER = [2, 4, 6, 8, 10, 12, 14, 16]


def quiet_display(*objects, **_kwargs):
    print(f"[display] {len(objects)} object(s)", flush=True)


nb = nbformat.read(NOTEBOOK, as_version=4)
ns = {"__name__": "__main__"}

for idx in CELL_ORDER:
    print(f"Executing notebook cell {idx}", flush=True)
    exec(compile(nb.cells[idx].source, f"{NOTEBOOK}:cell{idx}", "exec"), ns)
    ns["display"] = quiet_display

print("CostExperiment execution finished.", flush=True)
