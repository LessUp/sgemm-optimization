<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

const props = defineProps<{
  light: string
  dark: string
  alt: string
  caption?: string
  wide?: boolean
}>()

const { isDark } = useData()
const src = computed(() => withBase(isDark.value ? props.dark : props.light))
</script>

<template>
  <div :class="['figure-frame', wide && 'figure-frame-wide']">
    <img class="hero-figure" :src="src" :alt="alt" />
    <p v-if="caption" class="figure-note">{{ caption }}</p>
  </div>
</template>
