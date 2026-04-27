---
layout: default
title: 规范索引
nav_order: 9
has_children: true
permalink: /zh/specs/
lang: zh-CN
page_key: zh-specs
lang_ref: specs
---

# 规范索引
{: .fs-8 }

权威工程规则与需求
{: .fs-6 .fw-300 }

---

## 概述

本仓库使用 **OpenSpec** 作为需求、workflow 和收尾阶段治理的唯一权威系统。

- **稳定规范** 位于 `openspec/specs/`
- **活动变更** 位于 `openspec/changes/<change>/`
- **归档变更** 位于 `openspec/changes/archive/`

本页面是面向读者的导航入口。规范性事实来源仍是仓库内的 OpenSpec 文件。

## 文档结构

```text
openspec/
├── config.yaml
├── specs/            # 稳定能力规范
├── changes/          # 活动变更提案
│   ├── <change>/     # proposal.md, design.md, tasks.md, specs/
│   └── archive/      # 已完成变更
├── README.md         # 仓库特定的 OpenSpec workflow 说明
└── AGENTS.md         # OpenSpec 相关的代理指南
```

## 稳定能力索引

| 能力 | 用途 | 来源 |
|------|------|------|
| Kernel | Kernel 行为、容差、benchmark 范围 | [`openspec/specs/kernel/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/kernel/spec.md) |
| Architecture | 架构与工程决策 | [`openspec/specs/architecture/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md) |
| Testing | 验证场景与执行边界 | [`openspec/specs/testing/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/testing/spec.md) |
| Repository Governance | 权威治理、workflow 所有权、自动化与工具期望 | [`openspec/specs/repository-governance/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/repository-governance/spec.md) |
| Project Presentation | README、Pages 和仓库元数据定位需求 | [`openspec/specs/project-presentation/spec.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/project-presentation/spec.md) |

## 变更 Workflow

| 步骤 | 命令 | 结果 |
|------|------|------|
| 探索 | `/opsx:explore` | 创建变更前明确范围、风险和权衡 |
| 提案 | `/opsx:propose "描述"` | 创建 proposal、design、tasks 和 delta specs |
| 实施 | `/opsx:apply` | 执行 `tasks.md` 并更新复选框 |
| 审查 | `/review` | 大型合并或归档前的高质量审查 |
| 归档 | `/opsx:archive` | 合并 delta specs 到稳定 specs，移动变更到 archive |

## 收尾规则

- **合并优于扩展**。删除或合并低价值产物，而非保留占位文件。
- **一次长时间实施优于频繁 `/fleet` 扩散**。
- 任何影响仓库结构、公开文档、workflow 或质量门禁的变更都应使用 OpenSpec。
- 每次重大清理后保持稳定 specs、治理文档、README 和 Pages 对齐。

仓库级贡献者指南见 [`AGENTS.md`](https://github.com/LessUp/sgemm-optimization/blob/master/AGENTS.md)。OpenSpec 细节见 [`openspec/README.md`](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/README.md)。
