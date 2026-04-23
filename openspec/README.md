# OpenSpec Workflow for This Repository

This repository is in a **closeout-oriented** phase: OpenSpec is used to keep structure, documentation, automation, and validation coherent while the project is being consolidated for long-term archival stability.

## Authoritative Layout

```text
openspec/
├── config.yaml
├── specs/                    # Stable capability specs
│   ├── architecture/spec.md
│   ├── kernel/spec.md
│   ├── project-presentation/spec.md
│   ├── repository-governance/spec.md
│   └── testing/spec.md
├── changes/
│   ├── <change>/
│   │   ├── .openspec.yaml
│   │   ├── proposal.md
│   │   ├── design.md
│   │   ├── tasks.md
│   │   └── specs/<capability>/spec.md
│   └── archive/
└── AGENTS.md
```

- `openspec/specs/` is the **stable source of truth**
- `openspec/changes/<change>/specs/` contains **delta specs for the active change**
- Archived changes preserve reasoning; they are not the stable spec source

## Default Command Flow

| Stage | Command | Notes |
|-------|---------|-------|
| Explore | `/opsx:explore` | Clarify scope, risks, and trade-offs |
| Propose | `/opsx:propose "description"` | Create proposal, design, tasks, and change-local specs |
| Apply | `/opsx:apply` | Work tasks in order and update checkboxes immediately |
| Review | `/review` | Recommended before large deletions, workflow changes, or archive |
| Archive | `/opsx:archive` | Merge delta specs into stable specs once all tasks are complete |

## Repository-Specific Expectations

- Prefer **deletion or merge** over keeping redundant placeholder files.
- Use OpenSpec for any non-trivial change affecting docs, workflow, validation, repo layout, or public positioning.
- Prefer **one long-running apply/autopilot session** over frequent `/fleet` fan-out.
- Keep README, Pages, governance docs, and stable specs aligned after any repo-wide cleanup.

## Validation Baseline

```bash
openspec validate --all
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
```

- Hosted CI is for formatting, compile-time validation, OpenSpec/repository checks, and Pages.
- GPU runtime verification and benchmarking remain local-only unless a GPU-enabled runner is explicitly added.

## References

- Root contributor guidance: [`../AGENTS.md`](../AGENTS.md)
- OpenSpec-specific agent guidance: [`AGENTS.md`](AGENTS.md)
- Human-friendly spec index: [`../specs.md`](../specs.md)
