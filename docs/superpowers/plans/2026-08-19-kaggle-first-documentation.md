# Kaggle-First Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Kaggle the default training path for beginners, publish only the completed Kaggle run as a training result, and replace stiff or template-like wording with calm, direct explanations.

**Architecture:** Rework navigation from the outside in: root entry points first, then the Kaggle setup and tutorial sequence, then supporting guides and reference links. Keep local dry runs and full local training available as optional material without changing any command, configuration schema, source code, or recorded metric file.

**Tech Stack:** Markdown, existing bilingual documentation, existing `pytest` documentation checks, Ruff format checks for Python snippets and project files.

---

### Task 1: Make the Main Entry Points Kaggle-First

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/README.md`
- Modify: `docs/README.zh-CN.md`
- Modify: `docs/tutorial/README.md`
- Modify: `docs/tutorial/README.zh-CN.md`
- Modify: `docs/tutorial/learning-path.md`
- Modify: `docs/tutorial/learning-path.zh-CN.md`

- [ ] **Step 1: Record the current navigation and wording problems**

Run:

```bash
rg -n "First workflow|第一次完整运行|bounded learning|有界学习|Evidence boundary|证据边界|recipe|配方|contract|契约" README.md README.zh-CN.md docs/README*.md docs/tutorial/README*.md docs/tutorial/learning-path*.md
```

Expected: the root and tutorial pages present local bounded training as the main route and contain repeated engineering-oriented terminology.

- [ ] **Step 2: Rewrite the root README pair around one comfortable first visit**

Use this content order in both languages:

```text
Project purpose and intended learner
Kaggle T4 result in one compact table
Real prediction image and a plain explanation
Start training on Kaggle: account -> authentication -> submit -> monitor -> download
What the tutorial teaches
Optional local checks and local-GPU training
Models, project outputs, and documentation links
Contributor commands and project policy
```

Keep the exact workflow string
`download -> prepare -> inspect -> dry run -> train -> evaluate -> predict` in
the Kaggle explanation so the existing documentation check remains meaningful.
Remove the long local command walkthrough from the first half of the README.
Describe the repository as a project or tutorial in prose; retain the official
repository name `pytorch-object-detection-lab` in clone commands and links.

- [ ] **Step 3: Rewrite the documentation and tutorial indexes**

Make `docs/README*.md` offer three clear choices: start on Kaggle, understand
the code, or look up a reference. Make `docs/tutorial/README*.md` show the
chapter sequence without prerequisites that require a completed local run.
Rewrite `learning-path*.md` so the hands-on stages are:

```text
understand boxes -> inspect VOC -> understand Faster R-CNN -> prepare Kaggle
-> run training -> read the saved result -> try optional local commands
```

Replace quiz-like "completion checks" with short "Before continuing" notes.

- [ ] **Step 4: Check bilingual navigation and local links**

Run:

```bash
UV_CACHE_DIR=/tmp/pytorch-object-detection-docs-uv-cache uv run pytest tests/test_documentation.py -q
```

Expected: all documentation tests pass, including bilingual page pairs, local
links, real commands, existing configuration paths, and the published metric.

- [ ] **Step 5: Commit the entry-point rewrite**

```bash
git add README.md README.zh-CN.md docs/README.md docs/README.zh-CN.md docs/tutorial/README.md docs/tutorial/README.zh-CN.md docs/tutorial/learning-path.md docs/tutorial/learning-path.zh-CN.md
git commit -m "docs: Make Kaggle the primary learning path"
```

### Task 2: Turn the Kaggle Guide Into the Main Training Tutorial

**Files:**
- Modify: `docs/guides/kaggle.md`
- Modify: `docs/guides/kaggle.zh-CN.md`
- Modify: `docs/tutorial/01-environment.md`
- Modify: `docs/tutorial/01-environment.zh-CN.md`
- Modify: `docs/tutorial/02-data-and-boxes.md`
- Modify: `docs/tutorial/02-data-and-boxes.zh-CN.md`

- [ ] **Step 1: Rewrite the Kaggle guide from a new user's perspective**

Use the following sections and keep English and Chinese behavior identical:

```text
What the run will do and how long it took
1. Create a Kaggle account and verify the phone/account requirements
2. Install `kaggle`, authenticate, and confirm the CLI works
3. Change the kernel owner ID in `kernel-metadata.json`
4. Submit the provided runner
5. Confirm Internet and a T4-or-newer GPU in the Kaggle page
6. Read heartbeat and epoch logs without restarting a healthy run
7. Confirm COMPLETE and download only `artifacts/.*`
8. Open metrics, evaluation files, images, and checkpoints
Common failures already seen in real runs
```

Explain plainly that `kagglehub` is not required for this runner. Include the
three observed failures and their direct fixes: unattached source archive in
v4, non-interactive dataset attachment in v5, and P100 `sm_60` incompatibility
in v6. State that T4 x2 may be displayed while this single-device project uses
only one T4. Use the completed v7 run as the expected example.

- [ ] **Step 2: Simplify the environment chapter**

Lead with Kaggle as the recommended training environment. Keep local `uv sync`,
CLI discovery, configuration display, and CPU dry run under an optional local
setup heading. Replace "trust boundary", "production boundary", and "failure
boundary" with direct descriptions of downloads, devices, and common errors.

- [ ] **Step 3: Reframe the data chapter for Kaggle and conceptual learning**

Explain that the supplied Kaggle runner downloads and prepares official VOC
2007 automatically. Keep the local download, preparation, inspection, and
preview commands in one optional section. Preserve the coordinate conversion,
`difficult` object behavior, split sizes, and manifest explanation, but avoid
making integrity implementation details part of the first reading path.

- [ ] **Step 4: Validate all commands and links**

Run:

```bash
UV_CACHE_DIR=/tmp/pytorch-object-detection-docs-uv-cache uv run pytest tests/test_documentation.py -q
```

Expected: all tests pass and every documented `detect`, Python entry point,
configuration path, and local link resolves.

- [ ] **Step 5: Commit the Kaggle onboarding rewrite**

```bash
git add docs/guides/kaggle.md docs/guides/kaggle.zh-CN.md docs/tutorial/01-environment.md docs/tutorial/01-environment.zh-CN.md docs/tutorial/02-data-and-boxes.md docs/tutorial/02-data-and-boxes.zh-CN.md
git commit -m "docs(kaggle): Add a beginner training walkthrough"
```

### Task 3: Teach Training and Evaluation Through the Recorded Kaggle Run

**Files:**
- Modify: `docs/tutorial/03-faster-rcnn.md`
- Modify: `docs/tutorial/03-faster-rcnn.zh-CN.md`
- Modify: `docs/tutorial/04-training.md`
- Modify: `docs/tutorial/04-training.zh-CN.md`
- Modify: `docs/tutorial/05-evaluation-and-inference.md`
- Modify: `docs/tutorial/05-evaluation-and-inference.zh-CN.md`
- Modify: `docs/recorded-run/README.md`
- Modify: `docs/recorded-run/README.zh-CN.md`

- [ ] **Step 1: Simplify the Faster R-CNN chapter language**

Keep the train/eval API distinction, four loss names, RPN/ROI relationship,
and use of lists for variable image sizes. Replace "contract" with concrete
phrasing such as "training input and output" or "evaluation behavior". Move
fine implementation caveats after the main model explanation.

- [ ] **Step 2: Rewrite the training chapter around Kaggle**

Use this order:

```text
What training changes in the model
How to launch the supplied Kaggle run
What appears in the Kaggle logs
How `best.pt` is selected from validation mAP
What the completed 26-epoch run produced
Optional: one local CPU dry run
Optional: full local training for a compatible GPU
Resume behavior reference link
```

Do not present the two-epoch `learning_minimal.yaml` run as a training result.
It may appear only as a quick local path check. Do not repeat long checkpoint
safety rules in this beginner chapter.

- [ ] **Step 3: Rewrite evaluation and inference around real saved outputs**

Start with `docs/recorded-run/evaluation/evaluation.json`, `per_class.csv`,
`errors.csv`, and the three real visualizations. Teach validation versus test,
`map_50_95`, `map_50`, false positives, misses, and difficult objects using the
Kaggle files. Put commands that require a local checkpoint in an optional
section because the repository does not commit the 145 MB checkpoint.

- [ ] **Step 4: Make the recorded-run page readable before technical details**

Open with the result table and visual examples. Follow with "How this run was
made" and "Files you can inspect". Move hashes, exact library versions, source
archive identity, and omitted large files to a final reproducibility-details
section. Keep every recorded number unchanged and clearly state that all
published training results come from Kaggle v7.

- [ ] **Step 5: Check that no non-Kaggle run is presented as a result**

Run:

```bash
rg -n -i "training result|训练结果|recorded result|实测结果|benchmark|基准" README*.md docs --glob '*.md' --glob '!docs/superpowers/**'
```

Expected: completed-result claims point to `docs/recorded-run` and identify
Kaggle; local dry runs and bounded examples explicitly describe checks or
teaching exercises rather than results.

- [ ] **Step 6: Commit the tutorial and result rewrite**

```bash
git add docs/tutorial/03-faster-rcnn.md docs/tutorial/03-faster-rcnn.zh-CN.md docs/tutorial/04-training.md docs/tutorial/04-training.zh-CN.md docs/tutorial/05-evaluation-and-inference.md docs/tutorial/05-evaluation-and-inference.zh-CN.md docs/recorded-run/README.md docs/recorded-run/README.zh-CN.md
git commit -m "docs(tutorial): Teach from the recorded Kaggle run"
```

### Task 4: Clean Up Supporting Guides and Template-Like Language

**Files:**
- Modify: `configs/README.md`
- Modify: `configs/README.zh-CN.md`
- Modify: `examples/README.md`
- Modify: `examples/README.zh-CN.md`
- Modify: `docs/guides/experiments.md`
- Modify: `docs/guides/experiments.zh-CN.md`
- Modify: `docs/guides/troubleshooting.md`
- Modify: `docs/guides/troubleshooting.zh-CN.md`
- Modify: `docs/guides/using-models.md`
- Modify: `docs/guides/using-models.zh-CN.md`
- Modify: `docs/guides/using-your-data.md`
- Modify: `docs/guides/using-your-data.zh-CN.md`
- Modify: `docs/concepts/code-tour.md`
- Modify: `docs/concepts/code-tour.zh-CN.md`
- Modify: `docs/concepts/configuration-flow.md`
- Modify: `docs/concepts/configuration-flow.zh-CN.md`
- Modify: `docs/concepts/detection-flow.md`
- Modify: `docs/concepts/detection-flow.zh-CN.md`
- Modify: `docs/concepts/how-faster-rcnn-works.md`
- Modify: `docs/concepts/how-faster-rcnn-works.zh-CN.md`
- Review: `docs/reference/*.md`
- Review: `scripts/README*.md`
- Review: `tests/README*.md`

- [ ] **Step 1: Rename configuration prose without renaming files or fields**

Change "recipes/配方" to "configurations/配置" in the configuration and model
guides. Describe `reference_fasterrcnn.yaml` as the Kaggle training
configuration and `learning_minimal.yaml` as an optional local path check.
Keep all actual filenames, YAML keys, and commands unchanged.

- [ ] **Step 2: Make examples and experiments feel optional and approachable**

Describe examples as short programs that isolate one idea, not "probes" or
"contracts". In the experiment guide, start with a simple one-change comparison
and link to the Kaggle result for the known baseline. Keep validation/test
selection guidance, but shorten provenance and comparison restrictions to the
facts a learner needs.

- [ ] **Step 3: Rewrite troubleshooting as symptom-to-fix guidance**

Organize by visible symptom: Kaggle submission, GPU compatibility, download,
environment, data, training, resume, evaluation, and metrics. Put the observed
Kaggle failures first. Remove instructions that ask beginners to collect a
large diagnostic dossier before they can act.

- [ ] **Step 4: Clean repeated terminology in concepts and references**

Replace repeated "lab/实验室", "recipe/配方", "contract/契约", "evidence
boundary/证据边界", "probe/探针", and "gate/门槛" where a direct sentence is
clearer. Preserve "contract" only in maintainer ADRs, schema compatibility,
and exact API concepts where replacing it would reduce accuracy. Do not rename
the repository, ADR filenames, Python symbols, configuration fields, or links.

- [ ] **Step 5: Audit the entire published documentation set**

Run:

```bash
rg -n -i "lab|实验室|recipe|配方|contract|契约|evidence boundary|证据边界|probe|探针|gate|门槛" README*.md docs configs examples scripts tests --glob '*.md' --glob '!docs/superpowers/**'
```

Expected: remaining matches are proper repository names, contributor-facing
architecture language, filenames/links, or cases where the technical meaning
is necessary. Beginner pages no longer use these words as a repeated voice.

- [ ] **Step 6: Commit the supporting-document cleanup**

```bash
git add configs/README.md configs/README.zh-CN.md examples/README.md examples/README.zh-CN.md docs/guides docs/concepts docs/reference scripts/README.md scripts/README.zh-CN.md tests/README.md tests/README.zh-CN.md
git commit -m "docs: Use clearer language across learning guides"
```

### Task 5: Run the Complete Documentation Verification

**Files:**
- Verify: all changed Markdown files
- Verify: `tests/test_documentation.py`

- [ ] **Step 1: Check Markdown whitespace and accidental conflict markers**

Run:

```bash
git diff --check
rg -n "^(<<<<<<<|=======|>>>>>>>)" README*.md docs configs examples scripts tests --glob '*.md'
```

Expected: both commands report no findings.

- [ ] **Step 2: Run documentation tests**

Run:

```bash
UV_CACHE_DIR=/tmp/pytorch-object-detection-docs-uv-cache uv run pytest tests/test_documentation.py -q
```

Expected: all documentation tests pass.

- [ ] **Step 3: Run the project test suite and static checks**

Run:

```bash
UV_CACHE_DIR=/tmp/pytorch-object-detection-docs-uv-cache uv run pytest -W error::DeprecationWarning
UV_CACHE_DIR=/tmp/pytorch-object-detection-docs-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/pytorch-object-detection-docs-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/tmp/pytorch-object-detection-docs-uv-cache uv run mypy
```

Expected: 220 or more tests pass; Ruff and mypy report no issues.

- [ ] **Step 4: Review the final diff as a beginner**

Confirm the following in the rendered source order:

```text
Kaggle is the obvious first training choice.
The exact v7 result is easy to find and is the only completed training result.
Local training is available but clearly optional.
Every beginner chapter has a clear next step.
Technical details remain reachable without dominating the main path.
English and Chinese pages make the same promises.
```

- [ ] **Step 5: Confirm repository state**

Run:

```bash
git status --short
git log -5 --oneline
```

Expected: only intentional documentation-plan state remains, and the new
documentation commits are visible. Do not push unless the user requests it.
