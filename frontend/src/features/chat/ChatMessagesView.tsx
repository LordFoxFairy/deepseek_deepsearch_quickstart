import React, { useEffect, useRef } from 'react';
// import { ScrollArea } from './components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// 定义消息和进度的类型
interface Message {
  id: string;
  sender: "user" | "ai";
  content: string;
  sources?: any[];
}

interface ProgressInfo {
  type: 'research' | 'writing';
  current: number;
  total: number;
  description: string;
}

interface ChatMessagesViewProps {
  messages: Message[];
  isLoading: boolean;
  progress: ProgressInfo | null;
}

// 进度条组件
const ProgressDisplay: React.FC<{ progress: ProgressInfo }> = ({ progress }) => {
  const progressType = progress.type === 'research' ? '研究中' : '写作中';
  const progressText = `${progressType} (${progress.current}/${progress.total}): ${progress.description}`;

  return (
    <div className="flex justify-center items-center p-2 bg-white/80 backdrop-blur-sm rounded-lg shadow-md mx-auto mb-4 border border-gray-200" style={{ maxWidth: '80%' }}>
      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
      <p className="ml-3 text-gray-700 text-sm truncate">{progressText}</p>
    </div>
  );
};

const ChatMessagesView: React.FC<ChatMessagesViewProps> = ({ messages, isLoading, progress }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null); // 新增一个 ref 用于聊天容器

  // 使用 useEffect 确保新消息到来时自动滚动到底部
  useEffect(() => {
    // 确保滚动发生在聊天容器内部，而不是整个窗口
    if (messagesEndRef.current && chatContainerRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages.length]); // 监听消息数组长度变化，确保新消息添加时滚动

  return (
    <div ref={chatContainerRef} className="flex-1 flex flex-col relative overflow-y-auto">
      <div className="p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {/* AI 头像 */}
            {message.sender === 'ai' && <span className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-400" />}
            {/* 聊天气泡本身：
                AI 消息气泡内部仍然保留 max-h-[70vh] 和 overflow-y-auto，
                以处理单个长消息内部的滚动。这与整个聊天窗口的滚动是独立的。 */}
            <div
              className={`p-4 rounded-lg max-w-4xl prose prose-sm ${
                message.sender === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-800 border border-gray-200 shadow-sm max-h-[70vh] overflow-y-auto' 
              }`}
            >
              {message.sender === 'ai' ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    h1: ({node, ...props}) => <h1 className="text-2xl font-bold my-3 border-b pb-2" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-xl font-semibold my-2" {...props} />,
                    a: ({node, ...props}) => <a className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer" {...props} />,
                    // 可以根据需要添加更多 Markdown 元素的样式
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              ) : (
                <p className="whitespace-pre-wrap">{message.content}</p>
              )}
            </div>
            {/* 用户头像 */}
            {message.sender === 'user' && <span className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500" />}
          </div>
        ))}
        {/* 用于自动滚动的空 div，确保新消息总能滚动到视图内 */}
        <div ref={messagesEndRef} />
      </div>
      {/* 加载进度条显示在底部，并确保它不影响滚动区域 */}
      {isLoading && progress && (
        <div className="absolute bottom-4 left-0 right-0 flex justify-center pointer-events-none">
           <ProgressDisplay progress={progress} />
        </div>
      )}
    </div>
  );
};

export default ChatMessagesView;
