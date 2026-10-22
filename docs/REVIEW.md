# Independent step review

Every step must be reviewed by a separate subagent before it can be marked `done`. This file has two
parts: the protocol for the implementing agent, and the instructions the reviewer follows.

## Part 1: protocol for the implementing agent

Run this after you believe all of the step's "Definition of done" items are met and your own
`uv run pytest` and `uv run ruff check .` pass.

1. Spawn **one fresh subagent** with the Agent tool (`subagent_type: "general-purpose"`, run in the
   foreground, since you need its result). Use exactly this prompt, substituting only the step ID:

   > You are an independent reviewer for step `<STEP_ID>` of the rocket-hopper repository at
   > `/var/home/main/src/rocket-hopper`. You have no other context. Read `docs/REVIEW.md` ("Part 2")
   > and follow it exactly. Do not modify any file in the repository.

2. **Do not add anything else to the prompt.** No summary of what you built, no hints about where to
   look, no claims about passing tests, no opinions about quality. Giving the reviewer your
   conclusions defeats the independence of the review.
3. Do not continue an earlier reviewer with SendMessage. Every review round uses a new subagent.
4. Act on the verdict:
   - `PASS`: record it (step 5) and finish the step.
   - `FAIL`: fix every listed blocking finding (do not argue with the reviewer in the code; if you
     believe a finding is wrong, record why under "Decisions" in `docs/PROGRESS.md` and fix the plan or
     the code so that the disagreement is resolved on paper). Re-run your tests, then spawn a **new**
     reviewer. Repeat.
   - After **3** failed rounds, stop: set the step to `blocked` and write the unresolved findings in the
     Notes column so a human can decide.
5. When the verdict is `PASS`, put `Reviewed: PASS (round N)` in the step's Notes column and in the
   session log, and copy any non-blocking suggestions into the Backlog in `docs/PROGRESS.md`.
6. If you cannot spawn subagents, do not mark the step `done`. Set it to `blocked` with the note
   "needs independent review" so a human or another agent can run Part 2.

## Part 2: instructions for the reviewer

You are the independent reviewer. You have not seen the implementation being made. Assume nothing
about it, and trust only what you observe in the repository and in command output.

### Rules

- **Read-only.** Do not edit, create, or delete any file in the repository, and do not change
  `docs/PROGRESS.md`. Do not run `git commit`, `git stash`, `git checkout` or anything else that
  changes the working tree or index. Write scratch files only under the scratchpad directory if you have
  one, otherwise do not write files.
- Use `uv run ...` for all Python commands. Never `pip install`.
- Be sceptical. A check you did not run yourself is not evidence. Do not accept comments, docstrings,
  or progress notes as proof that something works.

### Procedure

1. Read, in this order: `docs/CONVENTIONS.md`, `docs/AGENT_START.md`, `skills/test-driven-developer/SKILL.md`,
   `docs/steps/<STEP_ID>_*.md` (the step under review), and the row for the step in `docs/PROGRESS.md`.
   Also read the step files of its direct dependencies if needed to understand interfaces.
2. **Deliverables:** for every file, function, class, signature, config field and script listed in the
   step's "Deliverables", check that it exists and matches the specified interface and behaviour.
   List anything missing or different.
3. **Tests:** for every test listed in the step's "Tests" section, find the corresponding test in
   `tests/`. Check that it exists, asserts the stated behaviour with the stated tolerance, and could
   actually fail (look for vacuous assertions, tolerances loosened beyond the spec, tests skipped or
   marked `xfail`, and tests that only assert that code runs). Tests that are listed but absent are
   blocking findings.
4. **Run the checks yourself:** `uv sync`, `uv run pytest -q`, `uv run ruff check .`, and
   `uv run ruff format --check .`. Report the actual results. If the step defines other runnable
   outputs (a script, a CLI, a scenario), run them and check the output.
5. **Mutation spot check (at least 2 mutations, more for physics-heavy steps):** pick the core
   behaviours the tests claim to protect (for example a sign in a torque, a coefficient in an equation of
   motion, a threshold). Temporarily change one in a *copy* of the repo (for example
   `cp -r` to a scratch directory and edit the copy; never in the original), run the relevant tests
   against the copy with `uv run pytest` there, and confirm that at least one test fails. If a mutation
   survives, that is a finding: the tests do not pin that behaviour. Delete the copy afterwards.
6. **Conventions and rules:** check against `docs/CONVENTIONS.md` and `docs/AGENT_START.md`: frames
   and units, quaternion convention, state layout, explicit RNG usage, the controller import boundary
   (`hopper_control` and `hopper_estimation` must not import simulator modules), no global numpy RNG, and
   no hidden unit conversions. Check that any interface deviation was recorded under "Decisions" in
   `docs/PROGRESS.md` and in `docs/CONVENTIONS.md`.
7. **Scope:** note any code that implements things belonging to other steps or that is not asked for.
   Note any step requirement that was silently dropped instead of being recorded as a decision.
8. **Physics sanity** (steps with dynamics): independently check at least one number by hand or with a
   short throwaway script (for example hover thrust vs weight, a torque sign, a rocket-equation delta-v) and
   compare with what the code produces.
9. **TDD evidence:** git history may not exist, so judge by the tests: they should specify behaviour, not
   implementation details, per the TDD skill.

### Output format

Return exactly this structure, and nothing before the verdict line:

```
VERDICT: PASS | FAIL
STEP: <STEP_ID>

COMMANDS RUN
- <command>: <result summary>

BLOCKING FINDINGS (empty if PASS)
1. <file:line> <what is wrong> | <which requirement in the step file it violates> | <how to verify>

NON-BLOCKING SUGGESTIONS
- <suggestion>

MUTATION CHECKS
- <what you changed> -> <which test failed, or "SURVIVED">

REQUIREMENT CHECKLIST
- [x]/[ ] <each Deliverable and Test bullet from the step file, one line each>
```

`PASS` requires: every checklist item satisfied, all commands succeeded, no surviving mutations of
core behaviour, and no violations of the conventions. If you are unsure whether something is a
violation, treat it as a blocking finding and say what evidence would resolve it. Do not pass a step
out of politeness or because the amount of work seems large.
