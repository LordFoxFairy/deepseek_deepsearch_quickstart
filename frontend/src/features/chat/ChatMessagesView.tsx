import React, { useEffect, useRef } from 'react';
import { ScrollArea } from '../../components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';

// 定义单条聊天消息的结构
interface Message {
  id: string;
  sender: "user" | "ai";
  content: string;
  sources?: string[];
}

interface ChatMessagesViewProps {
  messages: Message[];
}

const ChatMessagesView: React.FC<ChatMessagesViewProps> = ({ messages }) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // 消息列表更新时，自动滚动到底部
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({
        top: scrollAreaRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  return (
    <ScrollArea className="flex-1 w-full" ref={scrollAreaRef}>
      <div className="p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start gap-3 ${
              message.sender === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {message.sender === 'ai' && <span className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-400" />}

            <div
              className={`p-4 rounded-lg max-w-3xl ${
                message.sender === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
              }`}
            >
              {message.sender === 'ai' ? (
                <ReactMarkdown
                  components={{
                    h1: ({node, ...props}) => <h1 className="text-2xl font-bold my-3 border-b pb-2" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-xl font-semibold my-2" {...props} />,
                    h3: ({node, ...props}) => <h3 className="text-lg font-semibold my-1" {...props} />,
                    p: ({node, ...props}) => <p className="leading-relaxed my-2" {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc pl-5 space-y-1" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal pl-5 space-y-1" {...props} />,
                    li: ({node, ...props}) => <li className="ml-2" {...props} />,
                    a: ({node, ...props}) => <a className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer" {...props} />,
                    // 修复：正确处理 code 元素的 props，特别是 'inline' 和 'children'
                    code({ node, inline, className, children, ...props }) {
                      return !inline ? (
                        <pre className="bg-gray-800 text-white p-3 rounded-md my-2 overflow-x-auto">
                          <code className={className} {...props}>
                            {children}
                          </code>
                        </pre>
                      ) : (
                        <code className="bg-gray-200 text-red-600 px-1 py-0.5 rounded" {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              ) : (
                <p className="whitespace-pre-wrap">{message.content}</p>
              )}

              {message.sources && message.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-300/50">
                  <h4 className="text-sm font-semibold mb-1">参考来源:</h4>
                  <ul className="list-disc list-inside text-sm space-y-1">
                    {message.sources.map((source, index) => (
                      <li key={index}>
                        <a
                          href={source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline break-all"
                        >
                          {source}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {message.sender === 'user' && <span className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500" />}
          </div>
        ))}
      </div>
    </ScrollArea>
  );
};

export default ChatMessagesView;
