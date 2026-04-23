---
layout: default
title: Specifications
nav_order: 9
has_children: true
permalink: /specs/
---

# Specifications
{: .fs-8 }

Authoritative engineering rules and requirements
{: .fs-6 .fw-300 }

---

## Overview

This repository uses **OpenSpec** as its single authoritative system for requirements, workflow, and closeout-stage governance.

- **Stable specs** live under `openspec/specs/`
- **Active changes** live under `openspec/changes/<change>/`
- **Archived changes** live under `openspec/changes/archive/`

This page is a navigation surface for humans. The normative source of truth remains the OpenSpec files inside the repository.

## Document Structure

```text
openspec/
├── config.yaml
├── specs/            # Stable capability specs
├── changes/          # Active change proposals
│   ├── <change>/     # proposal.md, design.md, tasks.md, specs/
│   └── archive/      # Completed changes
├── README.md         # Repository-specific OpenSpec workflow notes
└── AGENTS.md         # OpenSpec-specific agent guidance
```

## Stable Capability Index

| Capability | Purpose | Source |
|------------|---------|--------|
| Kernel | Kernel behaviors, tolerances, benchmark scope | [`openspec/specs/kernel/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/kernel/spec.md) |
| Architecture | Architecture and engineering decisions | [`openspec/specs/architecture/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md) |
| Testing | Validation scenarios and execution boundaries | [`openspec/specs/testing/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/testing/spec.md) |
| Repository Governance | Authoritative governance, workflow ownership, automation, and tooling expectations | [`openspec/specs/repository-governance/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/repository-governance/spec.md) |
| Project Presentation | README, Pages, and repository metadata positioning requirements | [`openspec/specs/project-presentation/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/project-presentation/spec.md) |

## Change Workflow

| Step | Command | Result |
|------|---------|--------|
| Explore | `/opsx:explore` | Clarify scope, risks, and trade-offs before creating a change |
| Propose | `/opsx:propose "description"` | Create proposal, design, tasks, and delta specs |
| Apply | `/opsx:apply` | Work through `tasks.md` and update checkboxes |
| Review | `/review` | High-signal review before major consolidation or archive |
| Archive | `/opsx:archive` | Merge delta specs into stable specs and move the change to archive |

## Closeout Rules

- Prefer **consolidation over expansion**. Delete or merge low-value artifacts instead of keeping placeholder files.
- Prefer **one long apply session** over frequent `/fleet` fan-out.
- Use OpenSpec for any change that affects repository structure, public docs, workflow, or quality gates.
- Keep stable specs, governance docs, README, and Pages aligned after every significant cleanup pass.

For repository-wide contributor guidance, see [`AGENTS.md`](https://github.com/LessUp/sgemm-optimization/blob/master/AGENTS.md). For OpenSpec-specific details, see [`openspec/README.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/README.md).
