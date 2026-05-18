---
layout: home
title: SGEMM 白皮书
---

<div class="home-shell">
  <div class="home-hero-grid">
    <div>
      <p class="home-eyebrow">CUDA SGEMM 白皮书 · 架构展示站 · KERNEL 学院</p>
      <h1 class="home-main-title">把一个 CUDA SGEMM 项目写成一份经得起追问的技术论证</h1>
      <p class="home-main-subtitle">
        这套站点面向严苛的面试官和高级开发者，不满足于“有一个更快的 kernel”。它把仓库重新组织成一条可辩护的技术链路：
        从项目导读开始，进入架构，进入学院，再进入验证与研究资料台。
      </p>
      <div class="home-action-row">
        <a class="btn" href="./overview/">打开项目导读</a>
        <a class="btn btn-outline" href="./architecture/">查看架构全景</a>
        <a class="btn btn-outline" href="./academy/">进入学院</a>
        <a class="btn btn-outline" href="./research/">打开研究资料台</a>
      </div>
      <div class="home-kicker-row">
        <span class="home-chip">5 级 kernel 阶梯</span>
        <span class="home-chip">cuBLAS 锚定验证</span>
        <span class="home-chip">中英镜像路由</span>
      </div>
    </div>
    <div class="figure-frame">
      <picture>
        <source srcset="/figures/whitepaper-system-dark.svg" media="(prefers-color-scheme: dark)" />
        <img class="hero-figure" src="/figures/whitepaper-system-light.svg" alt="连接项目导读、架构、学院、验证与研究资料台的 SGEMM 白皮书总图。" />
      </picture>
      <p class="figure-note">公共叙事被刻意组织成技术论证链：先给论点，再给架构、课程、证据，最后给技术谱系。</p>
    </div>
  </div>

  <div class="thesis-grid">
    <div class="signal-card">
      <div class="signal-title">项目论点</div>
      <div class="signal-value">优化必须可解释</div>
      <div class="signal-note">每个 kernel 都是为了暴露并改变某一类瓶颈，而不是为了多放一张跑分图。</div>
    </div>
    <div class="signal-card">
      <div class="signal-title">读者契约</div>
      <div class="signal-value">面试压力下也能讲清楚</div>
      <div class="signal-note">这套站点既能让评审快速审查设计，也能帮助候选人把项目讲成一条完整的工程链路。</div>
    </div>
    <div class="signal-card">
      <div class="signal-title">信任模型</div>
      <div class="signal-value">CI 负责结构，GPU 负责证据</div>
      <div class="signal-note">自动化负责仓库健康、Pages 构建与规范一致性，运行时正确性和性能仍然属于真实 GPU。</div>
    </div>
  </div>
</div>

## 按你的目标进入

<div class="route-grid">
  <div class="route-card">
    <h3>我只想先看 90 秒项目摘要</h3>
    <p>先看导读，再跳到架构页补上系统视角。</p>
    <div class="route-links">
      <a href="./overview/">项目导读</a>
      <a href="./architecture/">架构概述</a>
    </div>
  </div>
  <div class="route-card">
    <h3>我想知道每一级 kernel 为什么存在</h3>
    <p>先看阶梯和内存主线，再进入学院逐个打开深度页面。</p>
    <div class="route-links">
      <a href="./architecture/kernel-ladder">Kernel 阶梯</a>
      <a href="./academy/">学院导览</a>
    </div>
  </div>
  <div class="route-card">
    <h3>我更关心证据，不关心口号</h3>
    <p>先看验证，确认正确性策略、benchmark 范围和可复现性边界，再决定这些数字是否值得相信。</p>
    <div class="route-links">
      <a href="./validation/">验证概览</a>
      <a href="./validation/benchmark-results">Benchmark 结果</a>
    </div>
  </div>
  <div class="route-card">
    <h3>我想看技术谱系和对照材料</h3>
    <p>研究资料台负责论文、相关仓库，以及这个项目为什么长成现在这个样子的演进思考。</p>
    <div class="route-links">
      <a href="./research/">研究总览</a>
      <a href="./research/related-projects">相关项目</a>
    </div>
  </div>
</div>

## 白皮书主干

| 表面 | 它回答什么 | 为什么存在 |
|---|---|---|
| [导读](./overview/) | 这是什么项目，为什么值得看，应该怎么读？ | 给新读者和评审一个坚决的入口。 |
| [架构](./architecture/) | SGEMM 系统怎么组织，它的核心约束是什么？ | 把实现细节提升为可辩护的系统地图。 |
| [学院](./academy/) | 应该按什么顺序学习优化阶梯？ | 把仓库包装成课程，而不是散页笔记。 |
| [验证](./validation/) | 这些证据到底能证明什么，不能证明什么？ | 让项目保持技术诚实。 |
| [研究](./research/) | 这些想法来自哪里，又该和谁做对照？ | 增强学术和对比维度。 |

## 用受控图示表达架构主线

<div class="figure-frame figure-frame-wide">
  <picture>
    <source srcset="/figures/kernel-ladder-dark.svg" media="(prefers-color-scheme: dark)" />
    <img class="hero-figure" src="/figures/kernel-ladder-light.svg" alt="从 naive FP32 到 tiled、bank-free、double-buffer、Tensor Core WMMA 的 kernel 阶梯，并联接架构、验证和研究三条辅助轨。" />
  </picture>
  <p class="figure-note">这条阶梯不是奖杯陈列架，而是一张瓶颈转移、接口约束与证据要求的地图。</p>
</div>

## 为什么这套呈现方式更强

1. 它把 SGEMM 当作技术论证，而不是项目秀场。
2. 它把架构、学院、验证、研究拆成清晰分工，每页只做一件事。
3. 它把中英镜像和公共深度视为项目的一部分，而不是临时补丁。

## 如果你想从仓库入口开始

- 英文仓库摘要： [README](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
- 中文仓库入口： [README.zh-CN](https://github.com/LessUp/sgemm-optimization/blob/master/README.zh-CN.md)
- 稳定需求来源： [OpenSpec 项目呈现规范](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/project-presentation/spec.md)
