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
  title: 'SGEMM Optimization',
  description: 'Bilingual CUDA SGEMM optimization tutorial and reference implementation',

  ignoreDeadLinks: [
    // External links that VitePress can't verify at build time
    /^https?:\/\//,
  ],

  locales: {
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'SGEMM Optimization',
      description: 'CUDA SGEMM optimization tutorial from naive kernels to Tensor Core WMMA',
      themeConfig: {
        nav: [
          { text: 'Guide', link: '/en/getting-started', activeMatch: '/en/' },
          { text: 'Architecture', link: '/en/architecture' },
          { text: 'Learning Path', link: '/en/learning-path' },
          { text: 'Kernels', link: '/en/kernel-naive', activeMatch: '/en/kernel-' },
          { text: 'Reference', link: '/en/cuda-memory-cheatsheet', activeMatch: '/en/' },
        ],
        sidebar: {
          '/en/': [
            {
              text: 'Getting Started',
              items: [
                { text: 'Introduction', link: '/en/getting-started' },
                { text: 'Learning Path', link: '/en/learning-path' },
                { text: 'Architecture', link: '/en/architecture' },
              ],
            },
            {
              text: 'Kernel Optimizations',
              items: [
                { text: 'Naive Kernel', link: '/en/kernel-naive' },
                { text: 'Tiled Kernel', link: '/en/kernel-tiled' },
                { text: 'Bank Conflict Free', link: '/en/kernel-bank-free' },
                { text: 'Double Buffer', link: '/en/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/en/kernel-tensor-core' },
              ],
            },
            {
              text: 'Reference',
              items: [
                { text: 'CUDA Memory Cheatsheet', link: '/en/cuda-memory-cheatsheet' },
                { text: 'Optimization Playbook', link: '/en/optimization-playbook' },
                { text: 'Performance Casebook', link: '/en/performance-casebook' },
                { text: 'Benchmark Results', link: '/en/benchmark-results' },
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
      title: 'SGEMM 优化',
      description: 'CUDA SGEMM 优化教程，从朴素内核到 Tensor Core WMMA',
      themeConfig: {
        nav: [
          { text: '指南', link: '/zh/getting-started', activeMatch: '/zh/' },
          { text: '架构', link: '/zh/architecture' },
          { text: '学习路径', link: '/zh/learning-path' },
          { text: '内核', link: '/zh/kernel-naive', activeMatch: '/zh/kernel-' },
          { text: '参考', link: '/zh/cuda-memory-cheatsheet', activeMatch: '/zh/' },
        ],
        sidebar: {
          '/zh/': [
            {
              text: '入门指南',
              items: [
                { text: '入门介绍', link: '/zh/getting-started' },
                { text: '学习路径', link: '/zh/learning-path' },
                { text: '架构概述', link: '/zh/architecture' },
              ],
            },
            {
              text: '内核优化',
              items: [
                { text: '朴素内核', link: '/zh/kernel-naive' },
                { text: '分块内核', link: '/zh/kernel-tiled' },
                { text: '消除 Bank Conflict', link: '/zh/kernel-bank-free' },
                { text: '双缓冲', link: '/zh/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/zh/kernel-tensor-core' },
              ],
            },
            {
              text: '参考资料',
              items: [
                { text: 'CUDA 内存速查表', link: '/zh/cuda-memory-cheatsheet' },
                { text: '优化手册', link: '/zh/optimization-playbook' },
                { text: '性能案例集', link: '/zh/performance-casebook' },
                { text: '基准测试结果', link: '/zh/benchmark-results' },
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
