import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * 一个用于条件性地合并 Tailwind CSS 类名的工具函数。
 * 它结合了 `clsx` (用于条件性地添加类) 和 `tailwind-merge` (用于解决 Tailwind 类名冲突)。
 * @param inputs - 任意数量的类名字符串、对象或数组。
 * @returns 合并后的 CSS 类名字符串。
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
