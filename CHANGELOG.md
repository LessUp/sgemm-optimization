---
layout: default
title: Changelog
nav_order: 10
permalink: /CHANGELOG
---

# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Consolidated repository governance around `openspec/specs/`, updated agent instructions, and simplified documentation roles.
- Reworked README, GitHub Pages content, and supporting docs into clearer repository-entry and learning surfaces.
- Began pruning redundant release-history and engineering guidance artifacts in favor of fewer authoritative files.

## [2.1.0] - 2026-04-16

### Added
- Tensor Core WMMA SGEMM kernel with guarded FP32 fallback for unsupported dimensions
- Benchmark enhancements, including roofline data export and configurable warmup/benchmark iterations
- Google Test coverage for standard kernels, Tensor Core fast path, fallback behavior, and edge cases
- Bilingual documentation and a GitHub Pages documentation site

### Changed
- Consolidated source code into `src/kernels/`, `src/utils/`, and `tests/`
- Adopted CMake as the primary build system while retaining the Makefile for quick local runs
- Expanded supported CUDA architecture targets to cover Volta through Hopper generation GPUs

### Fixed
- Tensor Core path memory management issues
- Double-buffer synchronization issues
- Grid dimension handling for non-square matrices

## [2.0.0] - 2026-03-13

### Added
- Bank-conflict-free and double-buffer SGEMM kernels
- CUDA Events-based benchmark infrastructure
- Nsight-oriented profiling support

### Changed
- Migrated from an earlier single-file layout to the current modular structure
- Standardized on CUDA 11.0+ and C++17

### Removed
- Legacy single-file benchmark script
- SM 6.x support

## [1.0.0] - 2025-02-13

### Added
- Initial naive and tiled SGEMM kernels
- Basic cuBLAS correctness verification
- First benchmark CLI
