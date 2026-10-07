"""Create the Task 2B cells for the team's shared final notebook."""
import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def main():
    cells = [
        markdown("# Task 2B: peak-day allocation\n\nRun from `Datathon/` after restoring the local competition datasets and checker copies. "
                 "Copy these cells into the shared final notebook. The last cell also demonstrates Task 1/2A inference; "
                 "run their pipelines first to create local models. Task 2B uses no trained model. "
                 "Source data and notebook outputs are kept local.\n"),
        code("from pathlib import Path\nimport json\nimport subprocess\nimport sys\nimport pandas as pd\n"
             "ROOT = Path.cwd()\nassert (ROOT / 'task2b/common.py').exists(), 'Start this notebook from Datathon/'\n"
             "sys.path.insert(0, str(ROOT))\n"
             "if hasattr(sys.stdout, 'reconfigure'):\n    sys.stdout.reconfigure(encoding='utf-8')\n"
             "from task2b.common import load_orders, load_fleet, SUBMISSION, REPORT_DIR\n"
             "from task2b.policy import protected\n"
             "orders, fleet = load_orders(), load_fleet()\n"
             "print('Orders:', len(orders), 'Available vehicles:', int(fleet.available.sum()))\n"
             "print(orders.groupby(['brand', 'temp_requirement'])[['order_volume_m3', 'order_weight_kg']].sum())\n"),
        markdown("## Capacity and policy\n\nWhole orders, one brand and district per trip, compatible depot/access/temperature, "
                 "two trips total, 270 Fresh minutes and 480 combined Style/Tech minutes. "
                 "Rank protected order count, chilled volume, total orders, outlets, then fewer minutes. "
                 "Flags are order-level; protect either flag once. S1-078 cannot fit an available vehicle.\n"),
        code("subprocess.run([sys.executable, 'task2b/analysis.py'], check=True)\n"
             "print('Protected orders:', sum(protected(r) for r in orders.itertuples()))\n"),
        markdown("## Reproduce all Task 2B phases\n\nThis runs greedy, the exact solver-free chilled search, independent exhaustive "
                 "tests, window diagnostics, repaired-reefer sensitivity, the policy and submission. "
                 "Sensitivity can take a few minutes. Optimal minutes are conditional on fixed ambient packing.\n"),
        code("subprocess.run([sys.executable, 'task2b/run_all.py'], check=True)\n"
             "summary = json.loads((REPORT_DIR / 'optimization_summary.json').read_text(encoding='utf-8'))\n"
             "print('Greedy:', summary['greedy_score'])\n"
             "print('Exact:', summary['optimized']['full_score'])\n"
             "print('Gain in chilled m3:', summary['chilled_volume_gain_m3'])\n"
             "print('Priority volume cost m3:', summary['priority_volume_cost_m3'])\n"),
        markdown("## Audits and written policy\n\nWindow waiting and approximate fuel are diagnostics. The no-return simulation "
                 "does not certify physical execution or weekly fuel compliance. Raw arrival after close defines lateness; "
                 "late unloading completion is reported separately. Individual chilled deferrals are joint choices, "
                 "while the Style oversized order is individually impossible.\n"),
        code("print((REPORT_DIR / 'audit_report.txt').read_text(encoding='utf-8'))\n"
             "print((REPORT_DIR / 'sensitivity_summary.json').read_text(encoding='utf-8'))\n"
             "print(Path('docs/handover/TASK2B_POLICY.md').read_text(encoding='utf-8'))\n"),
        markdown("## Final cell: saved-model inference and allocation\n\nLoads existing Task 1 and Task 2A models, prints inputs and predictions, "
                 "then displays/checks Task 2B. No training occurs in this cell.\n"),
        code("import runpy\nnamespace = runpy.run_path('docs/handover/inference_cell.py')\n"
             "for task in ('task1', 'task2a', 'task2b'):\n"
             "    saved = pd.read_csv(f'outputs/submission_{task}.csv', float_precision='round_trip')\n"
             "    assert namespace[f'{task}_pred'].equals(saved), task\n"
             "print('All three submissions match the final cell: PASSED')\n"),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 4}
    Path("docs/handover/TASK2B_NOTEBOOK.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    script = Path("docs/handover/inference_cell.py").read_text(encoding="utf-8")
    Path("docs/handover/NOTEBOOK_SNIPPETS.md").write_text(
        "# Final notebook cell: Tasks 1, 2A and 2B\n\n"
        "Run all three pipelines from the repository root first. This cell loads saved Task 1/2A models "
        "and displays/validates the Task 2B allocation. No training occurs here.\n\n"
        "The exact executable source is `inference_cell.py`; `task2b/integration_check.py` verifies "
        "its results against all three current submission files. See `TASK2B_NOTEBOOK.ipynb` for "
        "the complete Task 2B cells, and `TASK2B_README.md` for numerical regression notes. "
        "Task 1/2A both contain `models.py`: retain the explicit path loading and import order below.\n\n"
        "```python\n" + script + "```\n", encoding="utf-8")
    print("Task 2B notebook and three-task final cell synchronised")


if __name__ == "__main__":
    main()
