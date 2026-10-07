# Datasets (not in git)

The competition datasets are confidential (see the Terms and Conditions in the
Challenge Booklet) and are **not tracked** in this repository. Download them from
the competition link and place them at the repository root, exactly like this:

```
<repo root>/
├── General Data/
│   ├── calendar.csv
│   ├── district_travel.csv
│   ├── outlets.csv
│   ├── road_conditions.csv
│   ├── service_allowance.csv
│   ├── traffic_speed.csv
│   └── vehicles.csv
├── Training Data/
│   ├── deliveries_train.csv
│   └── route_legs_train.csv
├── Test Data/
│   ├── route_legs_test.csv
│   ├── task1_test_inputs.csv
│   ├── task2a_test_inputs.csv
│   ├── task2b_peak_day_fleet.csv
│   └── task2b_peak_day_scenarios.csv
└── Submission Templates/
    ├── submission_task1.csv
    ├── submission_task2a.csv
    └── submission_task2b.csv
```

**Task 2B checker:** `check_allocation.py` (the organisers' script, unchanged) searches
only this `data/` folder. Keep a second copy of `General Data/` and `Test Data/` here:

```
<repo root>/data/General Data/...
<repo root>/data/Test Data/...
```

These copies are ignored by git too (the folder-name patterns match at any depth).

All scripts are run from the repository root and read these folders by
relative path. The folders are listed in `.gitignore`, so they stay local.

Generated intermediate files (`task1/*.csv`, `task2a/data/`) are also
untracked; re-create them by running the pipelines.

**Note for teammates:** these files were removed from tracking on branch
`datathon-task2a-task1-fixes`. When that branch is merged and you pull, git
**deletes them from your working folder**. Keep a copy of the datasets
outside the repository and copy them back in after pulling.
