---
layout: home
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  // Auto-detect language and redirect
  const savedLang = localStorage.getItem('vitepress-locale')
  const browserLang = navigator.language.toLowerCase()
  
  if (savedLang) {
    window.location.href = savedLang === 'zh' ? '/sgemm-optimization/zh/' : '/sgemm-optimization/en/'
  } else if (browserLang.startsWith('zh')) {
    window.location.href = '/sgemm-optimization/zh/'
  } else {
    window.location.href = '/sgemm-optimization/en/'
  }
})
</script>

# SGEMM Optimization

Redirecting to your preferred language...
