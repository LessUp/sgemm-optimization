/**
 * SGEMM Optimization - 断点配置模块
 * 单一真相源：所有断点值从此模块导出
 *
 * 设计说明：
 * - CSS 规范不允许在 @media 中使用 var()，因此 CSS 媒体查询使用字面值
 * - 此模块作为 JS 层的断点真相源，供 Vue 组件和未来扩展使用
 * - CSS 中的断点变量 (--bp-sm, --bp-lg) 作为文档化引用
 */

/** 断点常量 - 与 VitePress 核心保持一致 */
export const BREAKPOINTS = {
  /** 移动端断点：640px 以下 */
  sm: 640,
  /** 平板/桌面断点：960px 以下 */
  lg: 960,
} as const

/** 断点名称类型 */
export type BreakpointName = keyof typeof BREAKPOINTS

/**
 * 生成媒体查询字符串（min-width 策略）
 * @param name 断点名称
 * @returns 媒体查询字符串，如 "(min-width: 640px)"
 */
export function minQuery(name: BreakpointName): string {
  return `(min-width: ${BREAKPOINTS[name]}px)`
}

/**
 * 生成媒体查询字符串（max-width 策略）
 * @param name 断点名称
 * @returns 媒体查询字符串，如 "(max-width: 639px)"
 */
export function maxQuery(name: BreakpointName): string {
  return `(max-width: ${BREAKPOINTS[name] - 1}px)`
}

/**
 * VueUse 响应式断点钩子
 * 在 Vue 组件中使用，获取当前视口状态
 *
 * @example
 * ```vue
 * <script setup>
 * import { useBreakpoint } from './breakpoints'
 * const { isMobile, isDesktop } = useBreakpoint()
 * </script>
 *
 * <template>
 *   <div v-if="isMobile">移动端视图</div>
 *   <div v-else>桌面端视图</div>
 * </template>
 * ```
 */
export function useBreakpoint() {
  // 动态导入 VueUse，避免在非 Vue 环境中报错
  // VitePress 已内置 @vueuse/core
  const { useMediaQuery } = require('@vueuse/core')

  return {
    /** 是否为移动端（< 640px） */
    isMobile: useMediaQuery(maxQuery('sm')),
    /** 是否为平板/移动端（< 960px） */
    isCompact: useMediaQuery(maxQuery('lg')),
    /** 是否为桌面端（>= 960px） */
    isDesktop: useMediaQuery(minQuery('lg')),
  }
}
