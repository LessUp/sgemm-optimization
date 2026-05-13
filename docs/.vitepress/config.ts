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
  title: 'SGEMM Optimization Lab',
  description: 'Interview-grade CUDA SGEMM engineering notebook from naive FP32 to guarded Tensor Core WMMA',

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
      title: 'SGEMM Optimization Lab',
      description: 'CUDA SGEMM project with benchmark discipline, interview storytelling, and research references',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/', activeMatch: '^/en/$' },
          { text: 'Quick Start', link: '/en/getting-started' },
          { text: 'Highlights', link: '/en/project-highlights' },
          { text: 'Interview', link: '/en/interview-playbook' },
          { text: 'Kernels', link: '/en/kernel-naive', activeMatch: '/en/kernel-' },
          { text: 'Benchmark', link: '/en/benchmark-results' },
          { text: 'References', link: '/en/references' },
        ],
        sidebar: {
          '/en/': [
            {
              text: 'Project Brief',
              items: [
                { text: 'Home', link: '/en/' },
                { text: 'Getting Started', link: '/en/getting-started' },
                { text: 'Project Highlights', link: '/en/project-highlights' },
                { text: 'Interview Playbook', link: '/en/interview-playbook' },
                { text: 'Architecture', link: '/en/architecture' },
              ],
            },
            {
              text: 'Kernel Ladder',
              items: [
                { text: 'Learning Path', link: '/en/learning-path' },
                { text: 'Naive Kernel', link: '/en/kernel-naive' },
                { text: 'Tiled Kernel', link: '/en/kernel-tiled' },
                { text: 'Bank Conflict Free', link: '/en/kernel-bank-free' },
                { text: 'Double Buffer', link: '/en/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/en/kernel-tensor-core' },
              ],
            },
            {
              text: 'Performance & Ops',
              items: [
                { text: 'Benchmark Results', link: '/en/benchmark-results' },
                { text: 'Optimization Playbook', link: '/en/optimization-playbook' },
                { text: 'Performance Casebook', link: '/en/performance-casebook' },
                { text: 'CUDA Memory Cheatsheet', link: '/en/cuda-memory-cheatsheet' },
              ],
            },
            {
              text: 'Research References',
              items: [
                { text: 'Papers & Repositories', link: '/en/references' },
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
      title: 'SGEMM 优化实验室',
      description: '面向面试展示与社区传播的 CUDA SGEMM 项目文档：从朴素内核到 Tensor Core WMMA',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/', activeMatch: '^/zh/$' },
          { text: '快速开始', link: '/zh/getting-started' },
          { text: '项目亮点', link: '/zh/project-highlights' },
          { text: '面试手册', link: '/zh/interview-playbook' },
          { text: '内核阶梯', link: '/zh/kernel-naive', activeMatch: '/zh/kernel-' },
          { text: '基准验证', link: '/zh/benchmark-results' },
          { text: '参考文献', link: '/zh/references' },
        ],
        sidebar: {
          '/zh/': [
            {
              text: '项目总览',
              items: [
                { text: '首页', link: '/zh/' },
                { text: '快速上手', link: '/zh/getting-started' },
                { text: '项目亮点', link: '/zh/project-highlights' },
                { text: '面试手册', link: '/zh/interview-playbook' },
                { text: '架构概述', link: '/zh/architecture' },
              ],
            },
            {
              text: '内核阶梯',
              items: [
                { text: '学习路径', link: '/zh/learning-path' },
                { text: '朴素内核', link: '/zh/kernel-naive' },
                { text: '分块内核', link: '/zh/kernel-tiled' },
                { text: '消除 Bank Conflict', link: '/zh/kernel-bank-free' },
                { text: '双缓冲', link: '/zh/kernel-double-buffer' },
                { text: 'Tensor Core WMMA', link: '/zh/kernel-tensor-core' },
              ],
            },
            {
              text: '性能与工程',
              items: [
                { text: 'Benchmark 结果', link: '/zh/benchmark-results' },
                { text: '优化手册', link: '/zh/optimization-playbook' },
                { text: '性能案例集', link: '/zh/performance-casebook' },
                { text: 'CUDA 内存速查表', link: '/zh/cuda-memory-cheatsheet' },
              ],
            },
            {
              text: '论文与引用',
              items: [
                { text: '论文与仓库索引', link: '/zh/references' },
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
