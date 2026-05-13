---
layout: home
hero:
  name: SGEMM Optimization
  text: ' '
  actions:
    - theme: brand
      text: English
      link: /en/
    - theme: alt
      text: 简体中文
      link: /zh/
---

<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vitepress'

onMounted(() => {
  const router = useRouter()
  const savedLang = localStorage.getItem('vitepress-locale')
  const browserLang = navigator.language.toLowerCase()

  if (savedLang === 'zh') {
    router.go('/zh/')
  } else if (savedLang === 'en') {
    router.go('/en/')
  } else if (browserLang.startsWith('zh')) {
    router.go('/zh/')
  } else {
    router.go('/en/')
  }
})
</script>
