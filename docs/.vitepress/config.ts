import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/sgemm-optimization/'

export default withMermaid(defineConfig({
  base,
  title: 'SGEMM Architecture Whitepaper',
  description: 'Bilingual SGEMM architecture guide covering CUDA kernel design, optimization methodology, validation evidence, and engineering resources',

  head: [
    ['meta', { name: 'theme-color', content: '#76b900' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/sgemm-optimization/favicon.svg' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/sgemm-optimization/favicon-32x32.png' }],
    ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: '/sgemm-optimization/apple-touch-icon.png' }],
  ],

  ignoreDeadLinks: [
    /^https?:\/\//,
  ],

  locales: {
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'SGEMM Architecture Whitepaper',
      description: 'CUDA SGEMM architecture guide with architecture walkthroughs, optimization methodology, validation boundaries, and engineering resources',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/', activeMatch: '^/en/$' },
          { text: 'Architecture', link: '/en/architecture/', activeMatch: '^/en/architecture' },
          { text: 'Methodology', link: '/en/methodology/', activeMatch: '^/en/(methodology|learning-path|kernel-.*|optimization-playbook)' },
          { text: 'Resources', link: '/en/resources/', activeMatch: '^/en/(resources|references|cuda-memory-cheatsheet|performance-casebook)' },
          { text: 'Validation', link: '/en/validation/', activeMatch: '^/en/(validation|benchmark-results)' },
          { text: 'Support', link: '/en/getting-started', activeMatch: '^/en/getting-started' },
        ],
        sidebar: {
          '/en/': [
            {
              text: 'Architecture',
              items: [
                { text: 'Architecture Overview', link: '/en/architecture/' },
                { text: 'Kernel Ladder', link: '/en/architecture/kernel-ladder' },
                { text: 'Memory Flow', link: '/en/architecture/memory-flow' },
                { text: 'Tensor Core Path', link: '/en/architecture/tensor-core-path' },
              ],
            },
            {
              text: 'Methodology',
              items: [
                { text: 'Methodology Overview', link: '/en/methodology/' },
                { text: 'Learning Path', link: '/en/learning-path' },
                { text: 'Naive Kernel', link: '/en/kernel-naive' },
                { text: 'Tiled Kernel', link: '/en/kernel-tiled' },
                { text: 'Bank Conflict Free', link: '/en/kernel-bank-free' },
                { text: 'Double Buffer', link: '/en/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/en/kernel-tensor-core' },
                { text: 'Benchmark Discipline', link: '/en/methodology/benchmark-discipline' },
                { text: 'Diagnosis Loop', link: '/en/methodology/diagnosis-loop' },
              ],
            },
            {
              text: 'Resources',
              items: [
                { text: 'Resources Hub', link: '/en/resources/' },
                { text: 'Further Reading Routes', link: '/en/resources/further-reading' },
                { text: 'Related Papers & Research', link: '/en/resources/papers' },
                { text: 'CUDA Memory Cheat Sheet', link: '/en/cuda-memory-cheatsheet' },
                { text: 'Performance Casebook', link: '/en/performance-casebook' },
                { text: 'Curated References', link: '/en/references' },
              ],
            },
            {
              text: 'Validation',
              items: [
                { text: 'Validation Overview', link: '/en/validation/' },
                { text: 'Correctness Policy', link: '/en/validation/correctness-policy' },
                { text: 'Benchmark Scope', link: '/en/validation/benchmark-scope' },
                { text: 'Reproducibility', link: '/en/validation/reproducibility' },
                { text: 'Benchmark Results', link: '/en/benchmark-results' },
              ],
            },
            {
              text: 'Support',
              items: [
                { text: 'SGEMM Architecture Whitepaper Home', link: '/en/' },
                { text: 'Getting Started', link: '/en/getting-started' },
              ],
            },
          ],
        },
      },
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      title: 'SGEMM 架构白皮书',
      description: '双语 SGEMM 架构指南，聚焦 CUDA 内核设计、优化方法、验证证据与工程资源',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/', activeMatch: '^/zh/$' },
          { text: '架构', link: '/zh/architecture/', activeMatch: '^/zh/architecture' },
          { text: '方法论', link: '/zh/methodology/', activeMatch: '^/zh/(methodology|learning-path|kernel-.*|optimization-playbook)' },
          { text: '资源', link: '/zh/resources/', activeMatch: '^/zh/(resources|references|cuda-memory-cheatsheet|performance-casebook)' },
          { text: '验证', link: '/zh/validation/', activeMatch: '^/zh/(validation|benchmark-results)' },
          { text: '支持', link: '/zh/getting-started', activeMatch: '^/zh/getting-started' },
        ],
        sidebar: {
          '/zh/': [
            {
              text: '架构',
              items: [
                { text: '架构概述', link: '/zh/architecture/' },
                { text: 'Kernel 阶梯', link: '/zh/architecture/kernel-ladder' },
                { text: 'Memory Flow', link: '/zh/architecture/memory-flow' },
                { text: 'Tensor Core 路径', link: '/zh/architecture/tensor-core-path' },
              ],
            },
            {
              text: '方法论',
              items: [
                { text: '方法论概览', link: '/zh/methodology/' },
                { text: '学习路径', link: '/zh/learning-path' },
                { text: '朴素内核', link: '/zh/kernel-naive' },
                { text: '分块内核', link: '/zh/kernel-tiled' },
                { text: '消除 Bank Conflict', link: '/zh/kernel-bank-free' },
                { text: '双缓冲', link: '/zh/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/zh/kernel-tensor-core' },
                { text: 'Benchmark 纪律', link: '/zh/methodology/benchmark-discipline' },
                { text: '诊断闭环', link: '/zh/methodology/diagnosis-loop' },
              ],
            },
            {
              text: '资源',
              items: [
                { text: '资源中心', link: '/zh/resources/' },
                { text: '延伸阅读路线', link: '/zh/resources/further-reading' },
                { text: '相关论文与研究', link: '/zh/resources/papers' },
                { text: 'CUDA 内存速查表', link: '/zh/cuda-memory-cheatsheet' },
                { text: '性能案例库', link: '/zh/performance-casebook' },
                { text: '参考资料清单', link: '/zh/references' },
              ],
            },
            {
              text: '验证',
              items: [
                { text: '验证概览', link: '/zh/validation/' },
                { text: '正确性策略', link: '/zh/validation/correctness-policy' },
                { text: 'Benchmark 范围', link: '/zh/validation/benchmark-scope' },
                { text: '可复现性', link: '/zh/validation/reproducibility' },
                { text: 'Benchmark 结果', link: '/zh/benchmark-results' },
              ],
            },
            {
              text: '支持',
              items: [
                { text: 'SGEMM 架构白皮书首页', link: '/zh/' },
                { text: '快速上手', link: '/zh/getting-started' },
              ],
            },
          ],
        },
      },
    },
  },

  themeConfig: {
    outline: [2, 3],
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/sgemm-optimization' },
    ],
    footer: {
      message: 'MIT Licensed',
      copyright: 'Copyright © 2026 LessUp',
    },
  },

  vite: {
    plugins: [llmstxt()],
  },

  // Mermaid 配置 - 通过 withMermaid 传入
  mermaid: {
    startOnLoad: true,
    theme: 'base',
    themeVariables: {
      primaryColor: '#76b900',
      primaryTextColor: '#1b2117',
      primaryBorderColor: '#5a9200',
      lineColor: '#4f5b47',
      secondaryColor: '#f3f5f1',
      tertiaryColor: '#f0f2ed',
      fontSize: '14px',
      fontFamily: 'ui-sans-serif, system-ui, sans-serif',
    },
  },
}))
