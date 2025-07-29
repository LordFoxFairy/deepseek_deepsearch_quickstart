import { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';

import InputForm from './features/chat/InputForm';
import ChatMessagesView from './features/chat/ChatMessagesView';
import ActivityTimeline from './components/ActivityTimeline';

// 类型定义
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

interface ActivityLog {
  id: string;
  message: string;
  timestamp: string;
}

interface TypingQueueItem {
  type: 'chapter' | 'references';
  content: string;
  title?: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([]);
  const [currentAiMessageId, setCurrentAiMessageId] = useState<string | null>(null);
  const [typingQueue, setTypingQueue] = useState<TypingQueueItem[]>([]);

  // 使用useRef来避免闭包问题
  const queueRef = useRef(typingQueue);
  queueRef.current = typingQueue;

  // 初始化会话ID
  useEffect(() => {
    let id = localStorage.getItem('deepsearch_session_id') || uuidv4();
    localStorage.setItem('deepsearch_session_id', id);
    setSessionId(id);
  }, []);

  // 处理打字机效果的useEffect
  useEffect(() => {
    if (isLoading || typingQueue.length === 0) {
      return; // 如果不在加载或队列为空，则不执行
    }

    const currentItem = typingQueue[0];
    let fullContent = '';
    if (currentItem.type === 'chapter' && currentItem.title) {
      fullContent = `\n\n## ${currentItem.title}\n\n${currentItem.content}`;
    } else {
      fullContent = currentItem.content;
    }

    let charIndex = 0;
    let timeoutId: NodeJS.Timeout;

    const typeChar = () => {
      if (charIndex < fullContent.length) {
        // 为了性能，一次性追加多个字符
        const nextChunk = fullContent.substring(charIndex, charIndex + 5);
        setMessages(prev => prev.map(msg =>
          msg.id === currentAiMessageId
            ? { ...msg, content: msg.content + nextChunk }
            : msg
        ));
        charIndex += nextChunk.length;
        // 加快打字速度，这里保持原有的 5ms，如果需要可以进一步调整
        timeoutId = setTimeout(typeChar, 5);
      } else {
        setTypingQueue(prev => prev.slice(1));
      }
    };

    typeChar(); // 启动打字

    return () => clearTimeout(timeoutId); // 清理函数
  }, [typingQueue, isLoading, currentAiMessageId]);


  const handleSendMessage = async (userMessage: string) => {
    if (!sessionId) return;

    setIsLoading(true);
    setProgress(null);
    setActivityLogs([]);
    setTypingQueue([]);

    const newUserMessage: Message = { id: uuidv4(), sender: "user", content: userMessage };
    setMessages([newUserMessage]); // 开始新会话时，清空旧消息

    const newAiMessageId = uuidv4();
    setCurrentAiMessageId(newAiMessageId);
    const newAiMessage: Message = { id: newAiMessageId, sender: "ai", content: "", sources: [] };
    setMessages(prev => [...prev, newAiMessage]);

    try {
      const response = await fetch('http://localhost:8000/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, session_id: sessionId }),
      });

      if (!response.ok) throw new Error(`API 错误: ${response.statusText}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("无法获取响应流阅读器。");

      const decoder = new TextDecoder();
      let buffer = '';
      let eventName = '';
      let eventData = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventName = line.substring('event:'.length).trim();
          } else if (line.startsWith('data:')) {
            eventData += line.substring('data:'.length).trim();
          } else if (line === '' && eventName) {
            if (eventData === '[DONE]') {
              setIsLoading(false);
              setProgress(null);
              break;
            }
            try {
              const jsonData = JSON.parse(eventData);
              switch (eventName) {
                case 'progress':
                  const progressInfo = jsonData as ProgressInfo;
                  setProgress(progressInfo);
                  const progressMessage = `${progressInfo.type === 'research' ? '研究中' : '写作中'} (${progressInfo.current}/${progressInfo.total}): ${progressInfo.description}`;
                  setActivityLogs(prev => [ { id: uuidv4(), message: progressMessage, timestamp: new Date().toLocaleTimeString() }, ...prev]);
                  break;
                case 'chapter':
                  setTypingQueue(prev => [...prev, { type: 'chapter', title: jsonData.title, content: jsonData.content }]);
                  break;
                case 'references':
                  setTypingQueue(prev => [...prev, { type: 'references', content: jsonData.content }]);
                  break;
                case 'sources':
                  setMessages(prev => prev.map(msg => msg.id === newAiMessageId ? { ...msg, sources: jsonData.sources } : msg));
                  break;
                case 'error':
                   setMessages(prev => prev.map(msg => msg.id === newAiMessageId ? { ...msg, content: `后端错误: ${jsonData.error}` } : msg));
                   setIsLoading(false);
                   setProgress(null);
                   break;
              }
            } catch (e) { console.error("解析SSE数据失败:", e, "原始数据:", eventData); }
            eventName = '';
            eventData = '';
          }
        }
      }
    } catch (error) {
       setMessages(prev => prev.map(msg => msg.id === currentAiMessageId ? { ...msg, content: `请求失败: ${(error as Error).message}` } : msg));
    } finally {
      setIsLoading(false);
      setProgress(null);
    }
  };

  return (
    // 整个应用容器，确保其高度为屏幕高度，并使用 flex 布局
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="flex items-center justify-between p-4 bg-white shadow-md">
        <h1 className="text-xl font-bold text-gray-800">DeepSearch AI Assistant</h1>
        {sessionId && (<span className="text-sm text-gray-500">Session ID: {sessionId.substring(0, 8)}...</span>)}
      </header>
      {/* 主内容区域，弹性填充可用空间并处理溢出 */}
      <main className="flex-1 flex overflow-hidden">
        {/* 聊天消息视图区域 (ChatMessagesView) 弹性填充并处理溢出。
            这个 div 的 flex-1 属性确保它会占据 main 标签内所有可用空间，
            并将其内部的溢出内容隐藏，为 ChatMessagesView 提供正确的尺寸。*/}
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatMessagesView messages={messages} isLoading={isLoading} progress={progress} />
        </div>
        {/* 活动时间线区域，固定宽度并处理溢出 */}
        <div className="w-1/3 max-w-md p-4 border-l border-gray-200 bg-white overflow-y-auto">
          <ActivityTimeline activities={activityLogs} />
        </div>
      </main>
      {/* 输入表单区域 */}
      <footer>
        <InputForm onSendMessage={handleSendMessage} isLoading={isLoading} />
      </footer>
    </div>
  );
}

export default App;
