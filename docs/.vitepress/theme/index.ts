import DefaultTheme from 'vitepress/theme'
import './style.css'
import { watch, onMounted } from 'vue'
import { useData } from 'vitepress'
import Citation from '../components/Citation.vue'
import PerfCompare from '../components/PerfCompare.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Citation', Citation)
    app.component('PerfCompare', PerfCompare)
  },
  setup() {
    const { isDark } = useData()

    // Mermaid 深浅色主题动态切换
    const updateMermaidTheme = async (dark: boolean) => {
      // 动态导入 mermaid
      const mermaid = (window as any).mermaid
      if (!mermaid) return

      const themeVariables = dark ? {
        primaryColor: '#8bcd29',
        primaryTextColor: '#ebf0e6',
        primaryBorderColor: '#76b900',
        lineColor: '#b0b8aa',
        secondaryColor: '#141b13',
        tertiaryColor: '#1a2118',
        background: '#10140f',
        mainBkg: '#1a2118',
        fontSize: '14px',
      } : {
        primaryColor: '#76b900',
        primaryTextColor: '#1b2117',
        primaryBorderColor: '#5a9200',
        lineColor: '#4f5b47',
        secondaryColor: '#f3f5f1',
        tertiaryColor: '#f0f2ed',
        fontSize: '14px',
      }

      mermaid.initialize({
        startOnLoad: true,
        theme: dark ? 'dark' : 'base',
        themeVariables,
        flowchart: {
          curve: 'basis',
        },
        sequence: {
          mirrorActors: false,
        },
      })

      // 重新渲染所有 mermaid 图表
      const mermaidElements = document.querySelectorAll('.mermaid')
      mermaidElements.forEach((el) => {
        el.removeAttribute('data-processed')
      })

      try {
        await mermaid.run({ querySelector: '.mermaid' })
      } catch (e) {
        console.warn('Mermaid re-render warning:', e)
      }
    }

    onMounted(() => {
      // 初始化时设置主题
      updateMermaidTheme(isDark.value)
    })

    watch(isDark, (dark) => {
      updateMermaidTheme(dark)
    })
  }
}