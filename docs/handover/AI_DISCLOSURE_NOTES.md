# AI tool disclosure notes (Task 2A and the Task 1 changes)

Source: `WORKLOG.md`, which has an "Authorship" note for every phase. The tool was
**Claude Code** (an AI coding agent), working in the repository under a written plan
(`docs/DATATHON_PLAN.md`) and an approval protocol.

## How we worked
- **One phase at a time:** the agent started a phase only when told, reported in a fixed template, and left everything uncommitted.
- **Commits only after approval:** a team member reviewed each report and approved or asked for changes. Only then did the agent commit (and, from Phase 6, push the feature branch).
- **Decisions belonged to the team member:** anything the plan didn't cover went back to them. The agent did not choose it.
- **No prohibited tools:** no pre-trained models, AutoML, low-code tools or API-based modelling. Every model is scikit-learn, trained from scratch on the competition data.

## What the agent generated
- **Code:**
  - all Task 2A code (`task2a/*.py`): labels, features, analysis, backtest, experiments, final fit, prediction, submission, reports, `run_all.py`;
  - the Task 1 changes: the dropped-column fix, the feature groups, the label audit, the evaluation, the HGB and calibrated models, the final fit, the submission fix, `run_all.py`;
  - reproducibility checks and tests.
- **Documents:** `WORKLOG.md` entries, report files and plots, `task1/CHANGES_FOR_REVIEW.md`, and this handover package.

## What the team member decided, reviewed or changed

| Phase | Decisions |
|---|---|
| 0 | Untracked the datasets. Pinned the requirements. Dropped lightgbm instead of bypassing certificate checks. Asked for the investigation into why the original lateness model didn't reproduce. |
| 1 | Provided **independently computed reference numbers** (order counts, status breakdown, volumes by brand, 117 weeks per series, weekly means, two New Year weeks). The agent's labels had to reproduce all of them, and did. Decided to document the not_run undercount, not correct it. |
| 2 | Chose the extra features (closure timing, festival name, payday flags d0–d2) and analyses. Decided on the Style depot × weekday structure and its peak-week flag, and on festival-specific effects. |
| 3 | Set the GLM feature set, the four backtest periods and the report contents. |
| 4 | Set the selection discipline: selection on 4 periods; a confirmation period **evaluated once and never used for selection**. Chose the plain Style outlet model over the rule's near-tie pick, rejected post-hoc blending, accepted the Tech bias, and kept the HGB grid small. |
| 5 | Required the W22 check: an in-sample overlap of payday and festival build-up, with a keep rule set in advance. Required two from-scratch runs to give identical results. |
| 6–7 | Specified the label audit, the same-models rule for feature tests, the keep thresholds and the leakage assert. Decided to add road conditions (dispatch-morning assumption), and to drop the constant feature and the unscheduled-waits feature. |
| 8 | Added the season-matched test period B. Chose sigmoid calibration and the [0.001, 0.999] clip as a fixed choice. |
| 9–10 | Required the retrain on all data, the submission checks, the comparison with the original and the change report. Reduced Phase 10 to this handover. |

## How the work was verified
- **Reference numbers** for the Task 2A labels were computed independently by the team member, and asserted in code.
- **Labels were re-derived:** the Task 1 audit recomputes every label independently, and checks time parsing and midnight crossing.
- **Time-ordered validation only:** every model choice was tested on later data than it trained on. The final Task 2A configuration was also confirmed on an untouched period.
- **Reproducibility:** both pipelines were run twice from scratch, with byte-identical submissions. The inference cell reproduces both submissions exactly.
- **Regression checks:** after every refactor, earlier metrics were re-run and matched exactly. That covers the Phase 3 backtest, the Phase 4 score and the Task 1 baseline.
- **Mistakes caught in review:**
  - a label shown with the wrong feature count;
  - a feature that turned out to be constant;
  - a model file that wasn't byte-identical across runs;
  - the Phase 8 script committed early with Phase 7 (disclosed, not rewritten).

  Each was fixed and logged.
