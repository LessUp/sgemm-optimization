# GitHub Pages 优化 (2026-03-10)

## 变更内容

### _config.yml
- 添加 SEO 元数据（url、baseurl、lang）
- 添加 exclude 列表，排除源代码、构建产物、CSV 数据等非文档文件，加速 Jekyll 构建

### index.md（GitHub Pages 首页）
- 重写为专业中文落地页，与项目整体风格一致
- 添加 CI badge
- 性能表格增加耗时列
- 优化演进路线图使用 ASCII 框图
- 新增 Roofline 模型分析表、GPU 架构参考表、技术栈表
- 新增项目结构、测试验证、快速开始（含 CMake 构建）等章节

### pages.yml
- 路径触发过滤从宽泛的 `*.md` 收窄为具体文件名（index.md、README.md、README.zh-CN.md）
- sparse-checkout 同步收窄，仅检出 Jekyll 构建所需文件

### README.md
- 添加 CI / Pages badges
- 性能表格增加耗时列
- 添加 ASCII 优化演进框图
- 新增 CMake 构建指令
- 扩展项目结构（含文件描述）
- 新增测试验证、GPU 架构参考表、Engineering Quality 章节

### .gitignore
- 添加 `_site/`、`.jekyll-cache/`、`.jekyll-metadata`、`.cache/`
