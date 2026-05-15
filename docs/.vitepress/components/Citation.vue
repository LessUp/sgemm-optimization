<script setup lang="ts">
defineProps<{
  citeKey: string
  title: string
  authors: string
  year: string
  venue?: string
  doi?: string
  url?: string
}>()
</script>

<template>
  <div class="citation">
    <span class="citation-key">[{{ citeKey }}]</span>
    <span class="citation-title">{{ title }}</span>
    <span class="citation-authors"> — {{ authors }}</span>
    <span class="citation-year"> ({{ year }})</span>
    <span v-if="venue" class="citation-venue">, {{ venue }}</span>
    <div v-if="doi || url" class="citation-links">
      <a v-if="doi" :href="`https://doi.org/${doi}`" class="doi-link" target="_blank" rel="noopener">
        DOI: {{ doi }}
      </a>
      <a v-if="url" :href="url" target="_blank" rel="noopener">{{ url }}</a>
    </div>
  </div>
</template>
