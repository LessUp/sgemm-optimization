---
title: 3. Bank Conflict Free
---

# Kernel 3: Bank Conflict Free

Eliminating shared memory bank conflicts through padding



## Shared Memory Banks Explained

### Memory Organization

GPU shared memory is divided into **32 banks** (on modern architectures). Each bank can service one access per clock cycle.

```
Address → Bank Index:  address % 32

Bank 0  Bank 1  ...  Bank 31
┌─────┐ ┌─────┐     ┌─────┐
│ [0] │ │ [1] │ ... │ [31]│  ← addresses 0-31
├─────┤ ├─────┤     ├─────┤
│ [32]│ │ [33]│ ... │ [63]│  ← addresses 32-63
├─────┤ ├─────┤     ├─────┤
│ ... │ │ ... │ ... │ ... │
└─────┘ └─────┘     └─────┘
```

### Conflict Scenario

```cpp
__shared__ float tile[32][32];

// In the inner product loop:
for (int k = 0; k < 32; ++k) {
    sum += tile[ty][k] * tile[k][tx];  // All threads access column k
}
```

When threads in a warp read `tile[k][0]`, `tile[k][1]`, ..., `tile[k][31]`:
- Thread 0 accesses address: `k * 32 + 0` → Bank `(k * 32) % 32 = 0`
- Thread 1 accesses address: `k * 32 + 1` → Bank `(k * 32) % 32 = 0`
- ...
- Thread 31 accesses address: `k * 32 + 31` → Bank `(k * 32) % 32 = 0`

**Result**: All 32 threads hit **Bank 0** simultaneously → **32-way conflict**!



## The Solution: Padding

Change the shared memory declaration:

```cpp
// Before: 32-way bank conflict
__shared__ float As[TILE_SIZE][TILE_SIZE];      // 32×32

// After: No bank conflicts
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // 32×33
```

### Why This Works

With padding, the address calculation changes:

```
Address of As[row][col] = row × 33 + col

Bank index = (row × 33 + col) % 32
           = (row + col) % 32  (since 33 % 32 = 1)

Thread 0: (k + 0) % 32 = k % 32
Thread 1: (k + 1) % 32 = (k + 1) % 32
Thread 2: (k + 2) % 32 = (k + 2) % 32
...
Thread 31: (k + 31) % 32 = (k + 31) % 32
```

Each thread accesses a **different bank**!



## Performance Impact

| Metric | Tiled (32×32) | Bank-Free (32×33) | Improvement |
|--------|---------------|-------------------|-------------|
| **GFLOPS (1024³)** | 753 | 673 | Slight variation |
| **Bank Conflicts** | 32-way | None | **Eliminated** |
| **Shared Memory** | 8 KB | 8.4 KB | +5.5% overhead |
| **Access Cycles** | 32× | 1× | **32× faster** |

### Why Not Always Faster?

The bank-free kernel may show slight performance variation due to:

1. **Occupancy reduction**: Padding increases shared memory per block (8 KB → 8.4 KB), potentially reducing active blocks per SM
2. **Cache behavior**: Different memory strides affect L1 cache efficiency
3. **Latency hiding**: Bank conflicts in the tiled kernel may be partially hidden by memory latency or compute latency

The bank-free kernel provides more **consistent** performance across different scenarios and is essential for performance-critical applications where predictability matters.



## Alternative: Transposed Access

Another approach is to transpose matrix B during loading:

```cpp
// Transpose B tile in shared memory
Bs[tx][ty] = B[...];  // Note: [tx][ty] not [ty][tx]

// Then access:
sum += As[ty][k] * Bs[tx][k];  // Both row-major now
```

This also eliminates conflicts but adds complexity. Padding is simpler and widely used.



## Next Steps

Now that we have efficient shared memory access, the next optimization target is **memory latency hiding**. Even with bank-free access, threads still wait for memory loads.

→ Continue to [Double Buffer Kernel](/en/academy/kernel-double-buffer)

---
