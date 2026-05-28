import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/sgemm-optimization/'

const asset = (relativePath: string) => `${base}${relativePath.replace(/^\/+/, '')}`

function localeNav(prefix: '/en/' | '/zh/') {
  if (prefix === '/en/') {
    return [
      { text: 'Overview', link: '/en/overview/', activeMatch: '^/en/overview/' },
      { text: 'Architecture', link: '/en/architecture/', activeMatch: '^/en/architecture/' },
      { text: 'Academy', link: '/en/academy/', activeMatch: '^/en/academy/' },
      { text: 'Validation', link: '/en/validation/', activeMatch: '^/en/validation/' },
      { text: 'Research', link: '/en/research/', activeMatch: '^/en/research/' },
    ]
  }

  return [
    { text: '导读', link: '/zh/overview/', activeMatch: '^/zh/overview/' },
    { text: '架构', link: '/zh/architecture/', activeMatch: '^/zh/architecture/' },
    { text: '学院', link: '/zh/academy/', activeMatch: '^/zh/academy/' },
    { text: '验证', link: '/zh/validation/', activeMatch: '^/zh/validation/' },
    { text: '研究', link: '/zh/research/', activeMatch: '^/zh/research/' },
  ]
}

function localeSidebar(prefix: '/en/' | '/zh/') {
  if (prefix === '/en/') {
    return {
      '/en/overview/': [
        {
          text: 'Overview',
          items: [
            { text: 'Project Guide', link: '/en/overview/' },
            { text: 'Reader Map', link: '/en/overview/reader-map' },
            { text: 'Getting Started', link: '/en/overview/getting-started' },
          ],
        },
      ],
      '/en/architecture/': [
        {
          text: 'Architecture',
          items: [
            { text: 'Architecture Overview', link: '/en/architecture/' },
            { text: 'System Blueprint', link: '/en/architecture/system-blueprint' },
            { text: 'Kernel Ladder', link: '/en/architecture/kernel-ladder' },
            { text: 'Memory Flow', link: '/en/architecture/memory-flow' },
            { text: 'Tensor Core Path', link: '/en/architecture/tensor-core-path' },
          ],
        },
      ],
      '/en/academy/': [
        {
          text: 'Academy',
          items: [
            { text: 'Academy Overview', link: '/en/academy/' },
            { text: 'Learning Path', link: '/en/academy/learning-path' },
            { text: 'Benchmark Discipline', link: '/en/academy/benchmark-discipline' },
            { text: 'Diagnosis Loop', link: '/en/academy/diagnosis-loop' },
            { text: 'Naive Kernel', link: '/en/academy/kernel-naive' },
            { text: 'Tiled Kernel', link: '/en/academy/kernel-tiled' },
            { text: 'Bank Conflict Free', link: '/en/academy/kernel-bank-free' },
            { text: 'Double Buffer', link: '/en/academy/kernel-double-buffer' },
            { text: 'Tensor Core WMMA', link: '/en/academy/kernel-tensor-core' },
            { text: 'Optimization Playbook', link: '/en/academy/optimization-playbook' },
            { text: 'CUDA Memory Cheat Sheet', link: '/en/academy/cuda-memory-cheatsheet' },
          ],
        },
      ],
      '/en/validation/': [
        {
          text: 'Validation',
          items: [
            { text: 'Validation Overview', link: '/en/validation/' },
            { text: 'Performance Model', link: '/en/validation/performance-model' },
            { text: 'Correctness Policy', link: '/en/validation/correctness-policy' },
            { text: 'Benchmark Scope', link: '/en/validation/benchmark-scope' },
            { text: 'Reproducibility', link: '/en/validation/reproducibility' },
            { text: 'Benchmark Results', link: '/en/validation/benchmark-results' },
          ],
        },
      ],
      '/en/research/': [
        {
          text: 'Research',
          items: [
            { text: 'Research Desk', link: '/en/research/' },
            { text: 'Reference Map', link: '/en/research/reference-map' },
            { text: 'Curated References', link: '/en/research/references' },
            { text: 'Related Projects', link: '/en/research/related-projects' },
            { text: 'Evolution Notes', link: '/en/research/evolution' },
            { text: 'Papers', link: '/en/research/papers' },
            { text: 'Further Reading', link: '/en/research/further-reading' },
            { text: 'Performance Casebook', link: '/en/research/performance-casebook' },
          ],
        },
      ],
    }
  }

  return {
    '/zh/overview/': [
      {
        text: '导读',
        items: [
          { text: '项目导读', link: '/zh/overview/' },
          { text: '阅读地图', link: '/zh/overview/reader-map' },
          { text: '快速上手', link: '/zh/overview/getting-started' },
        ],
      },
    ],
    '/zh/architecture/': [
      {
        text: '架构',
        items: [
          { text: '架构概述', link: '/zh/architecture/' },
          { text: '系统蓝图', link: '/zh/architecture/system-blueprint' },
          { text: 'Kernel 阶梯', link: '/zh/architecture/kernel-ladder' },
          { text: 'Memory Flow', link: '/zh/architecture/memory-flow' },
          { text: 'Tensor Core 路径', link: '/zh/architecture/tensor-core-path' },
        ],
      },
    ],
    '/zh/academy/': [
      {
        text: '学院',
        items: [
          { text: '学院导览', link: '/zh/academy/' },
          { text: '学习路径', link: '/zh/academy/learning-path' },
          { text: 'Benchmark 纪律', link: '/zh/academy/benchmark-discipline' },
          { text: '诊断闭环', link: '/zh/academy/diagnosis-loop' },
          { text: '朴素内核', link: '/zh/academy/kernel-naive' },
          { text: '分块内核', link: '/zh/academy/kernel-tiled' },
          { text: '消除 Bank Conflict', link: '/zh/academy/kernel-bank-free' },
          { text: '双缓冲', link: '/zh/academy/kernel-double-buffer' },
          { text: 'Tensor Core WMMA', link: '/zh/academy/kernel-tensor-core' },
          { text: '优化作战手册', link: '/zh/academy/optimization-playbook' },
          { text: 'CUDA 内存速查表', link: '/zh/academy/cuda-memory-cheatsheet' },
        ],
      },
    ],
    '/zh/validation/': [
      {
        text: '验证',
        items: [
          { text: '验证概览', link: '/zh/validation/' },
          { text: '性能模型', link: '/zh/validation/performance-model' },
          { text: '正确性策略', link: '/zh/validation/correctness-policy' },
          { text: 'Benchmark 范围', link: '/zh/validation/benchmark-scope' },
          { text: '可复现性', link: '/zh/validation/reproducibility' },
          { text: 'Benchmark 结果', link: '/zh/validation/benchmark-results' },
        ],
      },
    ],
    '/zh/research/': [
      {
        text: '研究',
        items: [
          { text: '研究总览', link: '/zh/research/' },
          { text: '参考文献地图', link: '/zh/research/reference-map' },
          { text: '参考资料清单', link: '/zh/research/references' },
          { text: '相关开源项目', link: '/zh/research/related-projects' },
          { text: '演进思考', link: '/zh/research/evolution' },
          { text: '论文索引', link: '/zh/research/papers' },
          { text: '延伸阅读路线', link: '/zh/research/further-reading' },
          { text: '性能案例库', link: '/zh/research/performance-casebook' },
        ],
      },
    ],
  }
}

export default withMermaid(defineConfig({
  base,
  title: 'SGEMM Whitepaper',
  description: 'Bilingual SGEMM architecture whitepaper and kernel academy for CUDA optimization study',

  head: [
    ['meta', { name: 'theme-color', content: '#76b900' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['link', { rel: 'icon', type: 'image/svg+xml', href: asset('favicon.svg') }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: asset('favicon-32x32.png') }],
    ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: asset('apple-touch-icon.png') }],
  ],

  ignoreDeadLinks: [
    /^https?:\/\//,
  ],

  locales: {
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'SGEMM Whitepaper',
      description: 'CUDA SGEMM architecture whitepaper, kernel academy, validation guide, and research desk',
      themeConfig: {
        nav: localeNav('/en/'),
        sidebar: localeSidebar('/en/'),
      },
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      title: 'SGEMM 白皮书',
      description: '双语 SGEMM 架构白皮书、Kernel 学院、验证指南与研究资料台',
      themeConfig: {
        nav: localeNav('/zh/'),
        sidebar: localeSidebar('/zh/'),
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
    build: {
      rollupOptions: {
        output: {
          manualChunks: (id) => {
            // Split mermaid and related dependencies into separate chunk
            if (id.includes('mermaid') || id.includes('cytoscape')) {
              return 'mermaid-vendor'
            }
            // Split search/fuse dependencies into separate chunk
            if (id.includes('fuse') || id.includes('minisearch')) {
              return 'search-vendor'
            }
            // Split katex/markdown-it math dependencies
            if (id.includes('katex') || id.includes('markdown-it')) {
              return 'markdown-vendor'
            }
          },
        },
      },
    },
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
