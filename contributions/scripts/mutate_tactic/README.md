Scripts for generating:
1. Theorem mutation errors
2. Tactic mutation errors
3. Line/multiline mutation errors

## Build the Lean prerequisites

From the repository root, run:

```bash
python3 contributions/scripts/build_modules.py
```

The script reads the same default dataset and uses the same Lean project as
`mutate_tactic.py`. It validates that the source checkout matches the extracted
records, deduplicates repeated source files, and runs Lake in batches. The current
dataset has 208 records and 195 distinct source files, producing 25 batches with
the default batch size of 8.

The default requests each module's `:deps` facet. This builds the imports and
required plugins needed to check the full source, while leaving the source
module's own proof compilation to the mutation script. `--full` builds those
source modules' `.olean` files as well.

To preview the work without starting any builds or writing logs:

```bash
python3 contributions/scripts/build_modules.py --dry-run
```

To start with the first five dataset records and reduce CPU pressure:

```bash
python3 contributions/scripts/build_modules.py --limit 5 --threads 1 --batch-size 1
python3 contributions/scripts/mutate_tactic/mutate_tactic.py --limit 5 --seed 42
```

The default `--threads 2` sets `LEAN_NUM_THREADS` for Lake and its child processes.
This is a runtime worker-pool setting, not a strict total process or memory limit.
See [Lean's thread documentation](https://lean-lang.org/doc/reference/latest/IO/Tasks-and-Threads/).
Only one batch runs at a time.

You can interrupt with Ctrl-C and rerun the same command later. Lake checks its
incremental build traces and reuses completed artifacts, including successful
dependencies from a failed or interrupted batch. The script does not clean build
artifacts or update dependency pins. The first build can still take substantial
time; automation does not reduce the cost of individual Lean proofs.

Build output is shown live and saved to `runs/openai_math_build/batch-0001.log`,
etc. `runs/openai_math_build/report.json` records each batch's command, status,
exit code, duration, and log path. These files describe the latest run and are
replaced when the same log directory is reused. A failed batch stops the script
with a nonzero exit code; use `--keep-going` to attempt later batches too.

Other options:

- `--input PATH` and `--project-dir PATH`: choose the dataset and Lake project.
- `--get-cache`: first run `lake exe cache get` to fetch available Mathlib caches;
  this requires network access and does not provide all `OAI` build artifacts.
- `--timeout SECONDS`: set a per-batch deadline; the default has no deadline.
- `--log-dir PATH`: choose another output directory for logs and the report.

## Tactic mutation errors

From the repository root, run:

```bash
python3 contributions/scripts/mutate_tactic/mutate_tactic.py --seed 42
```

The default input is `data/openai_math/originals.jsonl` if it exists, otherwise
`data/raw/openai_math/originals.jsonl` (the extractor's current output). The default
output is `data/openai_math/tme.jsonl`. Rerunning replaces the output; each accepted
record is flushed immediately, so an interrupted run retains completed records.
Python's standard library is sufficient.

To choose paths or make a small run:

```bash
python3 contributions/scripts/mutate_tactic/mutate_tactic.py \
  --input data/raw/openai_math/originals.jsonl \
  --output data/openai_math/tme.jsonl \
  --project-dir external/openai-math/lean \
  --seed 42 --limit 10 --timeout 120
```

`--limit` counts input records, including omitted proofs. `--timeout` applies to
each Lean invocation. The project must match the records' `lean_toolchain` and
have its imported modules built. For the OpenAI checkout, first follow its setup
instructions and prepare the dependencies with:

```bash
python3 contributions/scripts/build_modules.py
```

Building every source theorem can be expensive. The script reports missing
imports as an environment error and stops, instead of recording them as tactic
mutation errors. It does not download dependencies or build modules itself.

For each input record, the script:

1. Elaborates the complete original file and checks that it has no compiler
   errors or `sorry` warnings.
2. Uses Lean's tactic information trees to find actual tactic heads inside the
   target `proof_span` (UTF-8 byte offsets). Comments, strings, proof terms such as
   `:= rfl`, and other declarations are excluded. An unambiguous `proof_text` is
   also accepted when a record has no `proof_span`.
3. Randomly selects an occurrence, then tries its other same-family tactics in
   random order. Only the tactic token changes; its arguments are preserved.
4. Accepts the first replacement with a real Lean error. If every replacement
   compiles, it selects another occurrence on a previously untried line. Proofs
   with no successful mutation are omitted. Timeouts are skipped and reported;
   they are never accepted as errors.

The role-based families are in `TACTIC_CLASSES` in `mutate_tactic.py`. The same
input, project, seed, and compiler outcomes produce the same selection order.
Syntax errors from incompatible tactic arguments are valid mutations, as are
elaboration errors and unsolved goals.

Every output object has exactly these fields:

- `correct_proof` and `incorrect_proof`: the complete original and modified files.
- `src_hash`: the original source hash, preserved from the input. If absent, a
  SHA-256 of `declaration_text` (or the complete source) is computed.
- `line` and `col`: the selected compiler diagnostic's **one-based** line and
  Unicode character column in `incorrect_proof`. These can differ from the
  mutation location when Lean reports a later error or an earlier unsolved goal.
- `error`: Lean's diagnostic message.
- `line_at_error`: the line at that diagnostic location.
- `state_at_error`: Lean's goal state associated with the failing syntax, or the
  state after the preceding tactic for a subsequent parser error. If parsing
  prevents tactic elaboration entirely, the original goal state immediately
  before the changed tactic is used; the preceding source is identical.
- `error_type`: always `"tme"`.

`contributions/scripts/mutate_tactic/tactic_probe.lean` is the compiler helper.
It loads the selected Lake environment and uses Lean's frontend to elaborate the
file, collect structured compiler diagnostics, and pretty-print actual tactic
states. Keep it next to `mutate_tactic.py`. The helper has been checked with Lean
4.28.0 and 4.34.1.

Run the unit tests with:

```bash
python3 -m unittest discover -s contributions/scripts -p 'test_*.py'
```
