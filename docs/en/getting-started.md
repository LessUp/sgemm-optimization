---
title: Getting Started
---

# Getting Started

Build, run, and validate the project without guessing the toolchain



## Recommended build flow

```bash
git clone https://github.com/LessUp/sgemm-optimization.git
cd sgemm-optimization

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Run the default benchmark:

```bash
./build/bin/sgemm_benchmark
```

Run the broader benchmark set:

```bash
./build/bin/sgemm_benchmark -a
```

Run tests:

```bash
ctest --test-dir build
```



## Validation boundary

| Environment | What to run |
|-------------|-------------|
| Local GPU machine | benchmark, runtime verification, `ctest` |
| Hosted CI | formatting, compile validation, OpenSpec/repository checks, Pages |

This split is intentional: GitHub-hosted runners validate repository health, while performance and CUDA runtime correctness still require a real GPU machine.



## Where to go next

- [Learning Path](/en/learning-path)
- [Architecture](/en/architecture/)
- [Benchmark Results](/en/benchmark-results)
