# Formal Run Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `run` command the auditable formal workflow for the completed CT/EUS experiment while retaining command-line control of retrieval and HMM parameters.

**Architecture:** Keep orchestration in `cli.py` and data validation in `inputs.py`. The CLI opts into a strict shared 100 mm plane contract for both CT and EUS inputs, while the reusable loader APIs retain their current permissive default for callers that need legacy records. The run bundle records the workflow contract and effective parameter values; no CT/EUS generation or external segmentation capability is added.

**Tech Stack:** Python 3.12+, argparse, NumPy, pytest, existing Mamba environment.

---

### Task 1: Define the formal CT/EUS input contract

**Files:**
- Modify: `tests/test_inputs.py`
- Modify: `src/ramalhinho2021/inputs.py`

- [x] **Step 1: Write failing strict-contract tests**

Add tests proving that strict loading accepts 100 mm records, rejects an EUS record with patient-world pose, rejects inconsistent EUS status/features, rejects out-of-range feature centroids, and rejects a CT record whose plane size is not 100 mm. Explicitly preserve JSONL `organ_labels` precedence and TAR fallback through the existing tests.

```python
query = inputs.load_eus_queries(
    eus_root,
    module,
    require_formal_contract=True,
)[0]
assert query.frame_id == frame_id

with pytest.raises(ValueError, match="patient_world_pose 必须为 false"):
    inputs.load_eus_queries(eus_root, module, require_formal_contract=True)
```

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction pytest tests/test_inputs.py -q
```

Expected: FAIL because the loader functions do not yet accept `require_formal_contract` and do not validate plane geometry.

- [x] **Step 3: Implement the minimal shared validators**

Add formal constants and validators in `inputs.py`:

```python
FORMAL_PLANE_WIDTH_MM = 100.0
FORMAL_PLANE_LENGTH_MM = 100.0
FORMAL_EUS_COORDINATE_SYSTEM = "synthetic_2d_10cm_crop"
FORMAL_VESSEL_LABELS = frozenset({"artery", "vein"})
```

The strict validators must enforce finite 100 mm dimensions, finite positive two-axis spacing, allowed labels, feature centroids inside the plane, and feature areas no larger than the plane. EUS records must additionally enforce matching directory/file/frame IDs, `status` in `gallery|unindexed`, status/feature consistency, `patient_world_pose is False`, and the formal coordinate-system marker.

Add `require_formal_contract: bool = False` to both loader APIs and call the validators only when true. This keeps direct legacy API use compatible while making the CLI strict.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction pytest tests/test_inputs.py -q
```

Expected: all `tests/test_inputs.py` tests pass.

### Task 2: Make `run` auditable and parameter-adjustable

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/ramalhinho2021/cli.py`

- [x] **Step 1: Write failing CLI tests**

Extend synthetic formal fixtures with the required geometry fields. Add assertions that a run invoked with `--k 1 --search-range 0` records those exact values and prints the effective retrieval parameters. Add a failure test proving invalid formal EUS geometry exits before creating the output directory.

```python
assert metadata["workflow_contract"] == "ramalhinho2021-formal-run/v1"
assert metadata["parameters"]["k"] == 1
assert metadata["parameters"]["search_range"] == 0
assert "K=1, r=0" in completed.stdout
```

- [x] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction pytest tests/test_cli.py -q
```

Expected: FAIL because strict loading, the workflow contract, and effective-parameter output are absent.

- [x] **Step 3: Wire strict validation into all CLI paths**

Pass `require_formal_contract=True` from `validate-eus`, `validate`, and `run`. Keep the existing single `run` command and its adjustable options:

```text
--k
--search-range
--organ-filter-mode
--hmm-window-size
--sigma-x --sigma-y --sigma-z --sigma-theta
--timestamps-csv
```

Print the effective `K` and `r` before retrieval. Add `workflow_contract`, `input_contract`, and effective parameters to `run_metadata.json`; do not add a generator, TotalSegmentator call, or `resample_old` dependency.

- [x] **Step 4: Run the focused CLI tests and verify GREEN**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction pytest tests/test_cli.py -q
```

Expected: all `tests/test_cli.py` tests pass.

### Task 3: Publish the formal command contract

**Files:**
- Modify: `README.md`
- Modify: `HMM文档.md`

- [x] **Step 1: Update the user-facing contract**

Document that `run` is the independent formal command, CT gallery and EUS cropped JSONL are external inputs, JSONL `organ_labels` has priority with TAR fallback, and formal coordinate checks are mandatory. Correct the concrete EUS delivery path to `交付宋老师文件2026.7.25`.

- [x] **Step 2: Document parameter overrides**

Show a command that changes `K` and `r`, and list all adjustable retrieval/HMM flags. State that defaults reproduce the completed experiment and overrides are recorded in `run_metadata.json`.

- [x] **Step 3: Check documentation consistency**

Run:

```bash
rg -n "formal-run|--search-range|patient_world_pose|synthetic_2d_10cm_crop|交付宋老师" README.md HMM文档.md
```

Expected: both documents describe the same formal command and coordinate contract.

### Task 4: Verify and synchronize

**Files:**
- Review all files changed by Tasks 1-3

- [x] **Step 1: Run the complete test suite**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction pytest -q
```

Expected: zero failures.

- [x] **Step 2: Run static and CLI smoke checks**

Run:

```bash
mamba run -n ramalhinho-2021-reproduction python -m compileall -q src registration run_reproduction.py
mamba run -n ramalhinho-2021-reproduction python run_reproduction.py run --help
```

Expected: both commands exit 0 and the help lists every adjustable parameter.

- [x] **Step 3: Review scope and worktree safety**

Run `git diff --check`, inspect `git diff --stat` and `git status --short`, and verify that the pre-existing modified documentation spec and `.superpowers/` files are not staged or changed by this implementation.

- [ ] **Step 4: Commit and push the requested project changes**

Stage only the plan, source, tests, README, and HMM documentation changed by this implementation. Commit without including pre-existing user changes, then push the current `main` branch to `origin` without force.
