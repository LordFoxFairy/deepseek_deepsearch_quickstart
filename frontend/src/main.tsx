import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './global.css';

// 获取 React 应用的根元素
const rootElement = document.getElementById('root');

// 确保根元素存在
if (rootElement) {
  // 使用 createRoot 将 React 应用渲染到 DOM
  createRoot(rootElement).render(
    <React.StrictMode>
      <App /> {/* 渲染 App 组件 */}
    </React.StrictMode>,
  );
} else {
  console.error('未找到 ID 为 "root" 的元素。请确保 index.html 中存在 <div id="root"></div>。');
}
