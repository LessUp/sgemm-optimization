<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  data: { name: string; gflops: number; color?: string }[]
  baseline?: string
}>()

const maxGflops = computed(() => Math.max(...props.data.map(d => d.gflops)))
</script>

<template>
  <div class="perf-compare">
    <div v-for="item in data" :key="item.name" class="perf-row">
      <span class="perf-label">{{ item.name }}</span>
      <div class="perf-bar-container">
        <div
          class="perf-bar"
          :style="{
            width: `${(item.gflops / maxGflops) * 100}%`,
            background: item.color || 'var(--vp-c-brand-1)'
          }"
        />
        <span class="perf-value">{{ item.gflops.toFixed(1) }} GFLOPS</span>
      </div>
    </div>
  </div>
</template>
