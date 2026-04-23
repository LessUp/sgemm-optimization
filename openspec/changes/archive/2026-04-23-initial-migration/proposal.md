# Proposal: Initial Migration to OpenSpec

## Summary

Migrate existing SDD specification structure to OpenSpec framework.

## Motivation

Standardize on OpenSpec for improved change tracking and AI assistant integration. The OpenSpec framework provides:

- Formalized change proposal workflow (`/opsx:propose`, `/opsx:apply`, `/opsx:archive`)
- Domain-organized specs with delta specs for incremental changes
- Built-in AI assistant integration via slash commands
- Archival system for preserving change history

## Scope

- All existing specs in `specs/` directory
- Current AGENTS.md instructions
- RFC and product documentation

## Approach

1. Created OpenSpec directory structure under `openspec/`
2. Transformed existing specs to OpenSpec domain format:
   - `specs/product/` → `openspec/specs/kernel/`
   - `specs/rfc/` → `openspec/specs/architecture/`
   - `specs/testing/` → `openspec/specs/testing/`
3. Updated AGENTS.md to reference OpenSpec workflow
4. Removed legacy `specs/` directory

## Status: Completed

Migration completed on 2026-04-23.
