import { defineConfig } from 'vitepress'
import llmstxt from 'vitepress-plugin-llms'

export default defineConfig({
  base: '/cuda-kernel-academy/',
  title: 'CUDA Kernel Academy',
  description: 'Systematic CUDA kernel engineering from SGEMM to inference',

  lang: 'zh-CN',

  lastUpdated: true,

  ignoreDeadLinks: true,

  sitemap: {
    hostname: 'https://lessup.github.io/cuda-kernel-academy/'
  },

  head: [
    ['meta', { name: 'description', content: 'Systematic CUDA kernel engineering from SGEMM fundamentals to reusable inference components' }],
    ['meta', { name: 'keywords', content: 'CUDA, kernel, SGEMM, GEMM, Tensor Core, FlashAttention, inference engine, GPU programming' }],
    ['meta', { name: 'author', content: 'LessUp' }],
    ['meta', { name: 'theme-color', content: '#76B900' }],
    ['meta', { property: 'og:title', content: 'CUDA Kernel Academy' }],
    ['meta', { property: 'og:description', content: 'Systematic CUDA kernel engineering from SGEMM fundamentals to reusable inference components' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: 'https://lessup.github.io/cuda-kernel-academy/' }],
    ['meta', { property: 'og:site_name', content: 'CUDA Kernel Academy' }],
    ['meta', { property: 'og:image', content: 'https://lessup.github.io/cuda-kernel-academy/og-image.png' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'CUDA Kernel Academy' }],
    ['meta', { name: 'twitter:description', content: 'Systematic CUDA kernel engineering from SGEMM fundamentals to reusable inference components' }],
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    ['link', { rel: 'canonical', href: 'https://lessup.github.io/cuda-kernel-academy/' }],
  ],

  themeConfig: {
    outline: [2, 3],

    search: {
      provider: 'local'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/cuda-kernel-academy' }
    ],

    nav: [
      {
        text: '指南',
        items: [
          { text: '快速开始', link: '/zh/guides/getting-started' },
          { text: '工作流', link: '/zh/guides/workflow' }
        ]
      },
      {
        text: '模块',
        items: [
          { text: '01-SGEMM', link: '/zh/modules/01-sgemm' },
          { text: '02-TensorCraft', link: '/zh/modules/02-tensorcraft' },
          { text: '03-HPC', link: '/zh/modules/03-hpc' },
          { text: '04-Inference', link: '/zh/modules/04-inference' }
        ]
      },
      {
        text: '参考',
        items: [
          { text: '编码风格', link: '/zh/reference/coding-style' },
          { text: '故障排查', link: '/zh/reference/troubleshooting' },
          { text: '集成示例', link: '/zh/reference/integration-examples' }
        ]
      },
      { text: 'Benchmarks', link: '/zh/benchmarks/' },
      { text: '路线图', link: '/zh/roadmap' }
    ],

    sidebar: {
      '/zh/modules/01-sgemm': [
        {
          text: '01-SGEMM 教程',
          items: [
            { text: '概述', link: '/zh/modules/01-sgemm' },
            { text: 'Naive 实现', link: '/zh/modules/01/sgemm-naive' },
            { text: '分块优化', link: '/zh/modules/01/sgemm-tiled' },
            { text: 'Bank Conflict', link: '/zh/modules/01/sgemm-bank-conflict' },
            { text: '双缓冲', link: '/zh/modules/01/sgemm-double-buffer' },
            { text: 'Tensor Core', link: '/zh/modules/01/sgemm-tensor-core' }
          ]
        }
      ],
      '/zh/modules/02-tensorcraft': [
        {
          text: '02-TensorCraft Core',
          items: [
            { text: '概述', link: '/zh/modules/02-tensorcraft' },
            { text: '架构', link: '/zh/modules/02/architecture' },
            { text: '优化指南', link: '/zh/modules/02/optimization-guide' },
            { text: 'API 参考', link: '/zh/modules/02/api-reference' }
          ]
        }
      ],
      '/zh/modules/03-hpc': [
        {
          text: '03-HPC 进阶',
          items: [
            { text: '概述', link: '/zh/modules/03-hpc' },
            { text: 'GEMM 优化', link: '/zh/modules/03/gemm-optimization' },
            { text: '内存优化', link: '/zh/modules/03/memory-optimization' },
            { text: '归约优化', link: '/zh/modules/03/reduction-optimization' },
            { text: 'Flash Attention', link: '/zh/modules/03/flash-attention' },
            { text: 'CUDA 13 特性', link: '/zh/modules/03/cuda13-features' }
          ]
        }
      ],
      '/zh/modules/04-inference': [
        {
          text: '04-Inference Engine',
          items: [
            { text: '概述', link: '/zh/modules/04-inference' },
            { text: '快速开始', link: '/zh/modules/04/quick-start' },
            { text: '架构', link: '/zh/modules/04/architecture' },
            { text: '性能调优', link: '/zh/modules/04/performance-tuning' }
          ]
        }
      ],
      '/zh/': [
        {
          text: '指南',
          items: [
            { text: '快速开始', link: '/zh/guides/getting-started' },
            { text: '工作流', link: '/zh/guides/workflow' }
          ]
        },
        {
          text: '参考',
          items: [
            { text: '编码风格', link: '/zh/reference/coding-style' },
            { text: '故障排查', link: '/zh/reference/troubleshooting' },
            { text: '集成示例', link: '/zh/reference/integration-examples' },
            { text: 'Benchmarks', link: '/zh/benchmarks/' },
            { text: '路线图', link: '/zh/roadmap' }
          ]
        }
      ]
    },

    editLink: {
      pattern: 'https://github.com/LessUp/cuda-kernel-academy/edit/main/docs/:path'
    },

    lastUpdated: {
      text: '最后更新于'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    skipToContentLabel: '跳转到内容',
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026 CUDA Kernel Academy Contributors'
    }
  },

  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN'
    },
    en: {
      label: 'English',
      lang: 'en-US',
      themeConfig: {
        nav: [
          {
            text: 'Guides',
            items: [
              { text: 'Getting Started', link: '/en/guides/getting-started' },
              { text: 'Workflow', link: '/en/guides/workflow' }
            ]
          },
          {
            text: 'Modules',
            items: [
              { text: '01-SGEMM', link: '/en/modules/01-sgemm' },
              { text: '02-TensorCraft', link: '/en/modules/02-tensorcraft' },
              { text: '03-HPC', link: '/en/modules/03-hpc' },
              { text: '04-Inference', link: '/en/modules/04-inference' }
            ]
          },
          {
            text: 'Reference',
            items: [
              { text: 'Coding Style', link: '/en/reference/coding-style' },
              { text: 'Troubleshooting', link: '/en/reference/troubleshooting' },
              { text: 'Integration', link: '/en/reference/integration-examples' }
            ]
          },
          { text: 'Benchmarks', link: '/en/benchmarks/' },
          { text: 'Roadmap', link: '/en/roadmap' }
        ],
        sidebar: {
          '/en/modules/01-sgemm': [
            {
              text: '01-SGEMM Tutorial',
              items: [
                { text: 'Overview', link: '/en/modules/01-sgemm' },
                { text: 'Naive', link: '/en/modules/01/sgemm-naive' },
                { text: 'Tiled', link: '/en/modules/01/sgemm-tiled' },
                { text: 'Bank Conflict', link: '/en/modules/01/sgemm-bank-conflict' },
                { text: 'Double Buffer', link: '/en/modules/01/sgemm-double-buffer' },
                { text: 'Tensor Core', link: '/en/modules/01/sgemm-tensor-core' }
              ]
            }
          ],
          '/en/modules/02-tensorcraft': [
            {
              text: '02-TensorCraft Core',
              items: [
                { text: 'Overview', link: '/en/modules/02-tensorcraft' },
                { text: 'Architecture', link: '/en/modules/02/architecture' },
                { text: 'Optimization Guide', link: '/en/modules/02/optimization-guide' },
                { text: 'API Reference', link: '/en/modules/02/api-reference' }
              ]
            }
          ],
          '/en/modules/03-hpc': [
            {
              text: '03-HPC Advanced',
              items: [
                { text: 'Overview', link: '/en/modules/03-hpc' },
                { text: 'GEMM Optimization', link: '/en/modules/03/gemm-optimization' },
                { text: 'Memory Optimization', link: '/en/modules/03/memory-optimization' },
                { text: 'Reduction', link: '/en/modules/03/reduction-optimization' },
                { text: 'Flash Attention', link: '/en/modules/03/flash-attention' },
                { text: 'CUDA 13 Features', link: '/en/modules/03/cuda13-features' }
              ]
            }
          ],
          '/en/modules/04-inference': [
            {
              text: '04-Inference Engine',
              items: [
                { text: 'Overview', link: '/en/modules/04-inference' },
                { text: 'Quick Start', link: '/en/modules/04/quick-start' },
                { text: 'Architecture', link: '/en/modules/04/architecture' },
                { text: 'Performance Tuning', link: '/en/modules/04/performance-tuning' }
              ]
            }
          ],
          '/en/': [
            {
              text: 'Guides',
              items: [
                { text: 'Getting Started', link: '/en/guides/getting-started' },
                { text: 'Workflow', link: '/en/guides/workflow' }
              ]
            },
            {
              text: 'Reference',
              items: [
                { text: 'Coding Style', link: '/en/reference/coding-style' },
                { text: 'Troubleshooting', link: '/en/reference/troubleshooting' },
                { text: 'Integration', link: '/en/reference/integration-examples' },
                { text: 'Benchmarks', link: '/en/benchmarks/' },
                { text: 'Roadmap', link: '/en/roadmap' }
              ]
            }
          ]
        },
        editLink: {
          pattern: 'https://github.com/LessUp/cuda-kernel-academy/edit/main/docs/:path'
        },
        lastUpdated: {
          text: 'Last updated'
        },
        docFooter: {
          prev: 'Previous',
          next: 'Next'
        },
        returnToTopLabel: 'Return to top',
        sidebarMenuLabel: 'Menu',
        darkModeSwitchLabel: 'Theme',
        lightModeSwitchTitle: 'Switch to light mode',
        darkModeSwitchTitle: 'Switch to dark mode',
        skipToContentLabel: 'Skip to content',
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2026 CUDA Kernel Academy Contributors'
        }
      }
    }
  },

  vite: {
    plugins: [
      llmstxt({
        includeAll: true,
        customInfo: 'CUDA Kernel Academy: 系统性 CUDA 算子工程学习项目，涵盖 SGEMM 优化、TensorCraft 核心库、HPC 高级特性和轻量级推理引擎。适合面试展示和进阶学习。'
      })
    ]
  }
})
