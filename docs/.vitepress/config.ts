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
  ],

  ignoreDeadLinks: [
    // External links that VitePress can't verify at build time
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
          { text: 'Architecture', link: '/en/architecture', activeMatch: '^/en/architecture' },
          { text: 'Methodology', link: '/en/learning-path', activeMatch: '^/en/(learning-path|kernel-.*|optimization-playbook)' },
          { text: 'Resources', link: '/en/references', activeMatch: '^/en/(references|cuda-memory-cheatsheet|performance-casebook)' },
          { text: 'Validation', link: '/en/benchmark-results', activeMatch: '^/en/benchmark-results' },
          { text: 'Support', link: '/en/getting-started', activeMatch: '^/en/getting-started' },
        ],
        sidebar: {
          '/en/': [
            {
              text: 'Home',
              items: [
                { text: 'SGEMM Architecture Whitepaper Home', link: '/en/' },
              ],
            },
            {
              text: 'Architecture',
              items: [
                { text: 'Architecture Overview', link: '/en/architecture' },
              ],
            },
            {
              text: 'Methodology',
              items: [
                { text: 'Learning Path', link: '/en/learning-path' },
                { text: 'Naive Kernel', link: '/en/kernel-naive' },
                { text: 'Tiled Kernel', link: '/en/kernel-tiled' },
                { text: 'Bank Conflict Free', link: '/en/kernel-bank-free' },
                { text: 'Double Buffer', link: '/en/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/en/kernel-tensor-core' },
                { text: 'Optimization Playbook', link: '/en/optimization-playbook' },
              ],
            },
            {
              text: 'Resources',
              items: [
                { text: 'CUDA Memory Cheat Sheet', link: '/en/cuda-memory-cheatsheet' },
                { text: 'Performance Casebook', link: '/en/performance-casebook' },
                { text: 'References', link: '/en/references' },
              ],
            },
            {
              text: 'Validation',
              items: [
                { text: 'Benchmark Results', link: '/en/benchmark-results' },
              ],
            },
            {
              text: 'Support',
              items: [
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
          { text: '架构', link: '/zh/architecture', activeMatch: '^/zh/architecture' },
          { text: '方法论', link: '/zh/learning-path', activeMatch: '^/zh/(learning-path|kernel-.*|optimization-playbook)' },
          { text: '资源', link: '/zh/references', activeMatch: '^/zh/(references|cuda-memory-cheatsheet|performance-casebook)' },
          { text: '验证', link: '/zh/benchmark-results', activeMatch: '^/zh/benchmark-results' },
          { text: '支持', link: '/zh/getting-started', activeMatch: '^/zh/getting-started' },
        ],
        sidebar: {
          '/zh/': [
            {
              text: '首页',
              items: [
                { text: 'SGEMM 架构白皮书首页', link: '/zh/' },
              ],
            },
            {
              text: '架构',
              items: [
                { text: '架构概述', link: '/zh/architecture' },
              ],
            },
            {
              text: '方法论',
              items: [
                { text: '学习路径', link: '/zh/learning-path' },
                { text: '朴素内核', link: '/zh/kernel-naive' },
                { text: '分块内核', link: '/zh/kernel-tiled' },
                { text: '消除 Bank Conflict', link: '/zh/kernel-bank-free' },
                { text: '双缓冲', link: '/zh/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/zh/kernel-tensor-core' },
                { text: '优化手册', link: '/zh/optimization-playbook' },
              ],
            },
            {
              text: '资源',
              items: [
                { text: 'CUDA 内存速查表', link: '/zh/cuda-memory-cheatsheet' },
                { text: '性能案例库', link: '/zh/performance-casebook' },
                { text: '参考文献', link: '/zh/references' },
              ],
            },
            {
              text: '验证',
              items: [
                { text: 'Benchmark 结果', link: '/zh/benchmark-results' },
              ],
            },
            {
              text: '支持',
              items: [
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
}))
