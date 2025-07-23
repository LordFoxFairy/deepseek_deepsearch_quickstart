import React from 'react';
/**
 * 将 Markdown 文本转换为 React 元素，同时处理 [text](url) 链接。
 * 这将把 Markdown 链接转换为可点击的 HTML <a> 标签，并可以添加自定义样式。
 * @param markdownText - 包含 Markdown 的文本。
 * @returns React.ReactNode 数组。
 */
export function renderMarkdownWithLinks(markdownText: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const regex = /\[([^\]]+?)]\((https?:\/\/[^\s)]+?)\)/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(markdownText)) !== null) {
    // 添加链接前的文本
    if (match.index > lastIndex) {
      parts.push(markdownText.substring(lastIndex, match.index));
    }

    // 添加链接元素
    parts.push(
      <a
        key={match.index}
        href={match[2]}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 hover:underline" // Tailwind 样式
      >
        {match[1]}
      </a>
    );
    lastIndex = regex.lastIndex;
  }

  // 添加最后一个链接后的文本
  if (lastIndex < markdownText.length) {
    parts.push(markdownText.substring(lastIndex));
  }

  return parts;
}
