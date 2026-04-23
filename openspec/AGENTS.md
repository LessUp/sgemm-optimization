# OpenSpec Agent Guide

Use this file for **OpenSpec-specific** behavior in this repository. Generic project rules live in the root [`AGENTS.md`](../AGENTS.md).

## Stable Capabilities

The stable authoritative specs are:

- `openspec/specs/kernel/spec.md`
- `openspec/specs/architecture/spec.md`
- `openspec/specs/testing/spec.md`
- `openspec/specs/repository-governance/spec.md`
- `openspec/specs/project-presentation/spec.md`

Do not treat archived changes or old root-level spec references as normative.

## Change Lifecycle

```text
explore -> propose -> apply -> review -> archive
```

1. Use `/opsx:explore` when scope, cleanup aggressiveness, or trade-offs are unclear.
2. Use `/opsx:propose` for any repo-wide change affecting structure, docs, workflow, or quality gates.
3. Use `/opsx:apply` to work tasks in dependency order.
4. Use `/review` before major deletions, workflow simplification, or archive.
5. Use `/opsx:archive` only after tasks, docs, specs, and validation are aligned.

## Authoring Rules for This Repository

### Proposal

- Frame the change around repository outcomes, not generic process language.
- Be explicit when the goal is consolidation, deletion, or archive-readiness.

### Design

- Capture why one authoritative source is chosen when duplicate systems exist.
- Call out validation boundaries between local GPU checks and CI-safe checks.

### Specs

- Stable capability specs belong under `openspec/specs/<capability>/spec.md`.
- Change deltas belong under `openspec/changes/<change>/specs/<capability>/spec.md`.
- Use `ADDED`, `MODIFIED`, `REMOVED`, or `RENAMED` exactly.
- Every requirement needs at least one `#### Scenario`.

### Tasks

- Group tasks by dependency and cleanup phase.
- Use trackable checkboxes only: `- [ ]`.
- Prefer tasks that can complete in one focused apply session.

## Closeout Bias

- Prefer deleting or merging low-value artifacts over preserving shells.
- Prefer one longer apply session over `/fleet` unless the work is naturally parallel.
- Keep the stable specs, README, Pages, and automation model synchronized at each major milestone.
