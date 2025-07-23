import { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

import InputForm from './features/chat/InputForm';
import ChatMessagesView from './features/chat/ChatMessagesView';
import ActivityTimeline from './components/ActivityTimeline';

interface Message {
  id: string;
  sender: "user" | "ai";
  content: string;
  sources?: string[];
}

interface ActivityLog {
  step_name: string;
  output: string;
  timestamp: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([]);
  const [currentAiMessageId, setCurrentAiMessageId] = useState<string | null>(null);

  useEffect(() => {
    let currentSessionId = localStorage.getItem('deepsearch_session_id');
    if (!currentSessionId) {
      currentSessionId = uuidv4();
      localStorage.setItem('deepsearch_session_id', currentSessionId);
    }
    setSessionId(currentSessionId);
  }, []);

  const handleSendMessage = async (userMessage: string) => {
    if (!sessionId) {
      console.error("会话ID未初始化。");
      return;
    }

    setIsLoading(true);
    setActivityLogs([]);

    const newUserMessage: Message = { id: uuidv4(), sender: "user", content: userMessage };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);

    const newAiMessageId = uuidv4();
    setCurrentAiMessageId(newAiMessageId);
    const newAiMessage: Message = { id: newAiMessageId, sender: "ai", content: "思考中...", sources: [] };
    setMessages((prevMessages) => [...prevMessages, newAiMessage]);

    try {
      const response = await fetch('http://localhost:8000/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, session_id: sessionId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API 错误: ${errorData.detail || response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("无法获取响应流阅读器。");

      let buffer = "";
      let currentEvent: { event?: string; data?: string } = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          setIsLoading(false);
          break;
        }

        buffer += new TextDecoder().decode(value);
        const lines = buffer.split('\n');
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent.event = line.substring('event: '.length).trim();
            currentEvent.data = '';
          } else if (line.startsWith('data: ')) {
            const dataContent = line.substring('data: '.length).trim();
            if (dataContent === '[DONE]') {
                setIsLoading(false);
                break;
            }
            currentEvent.data = (currentEvent.data || '') + dataContent;
          } else if (line === '' && currentEvent.event && currentEvent.data !== undefined) {
            const { event: eventType, data: eventData } = currentEvent;

            if (eventType === 'activity_update' && eventData) {
              try {
                const activityData = JSON.parse(eventData);
                const newActivity: ActivityLog = {
                  step_name: activityData.step_name,
                  output: activityData.output,
                  timestamp: new Date().toLocaleTimeString(),
                };
                setActivityLogs((prevLogs) => [...prevLogs, newActivity]);
              } catch (e) {
                console.error("解析活动数据失败:", e, "原始数据:", eventData);
              }
            } else if (eventType === 'final_response' && eventData) {
              try {
                const finalData = JSON.parse(eventData);
                const finalAnswer = finalData.answer;
                const finalSources = finalData.sources || [];

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === newAiMessageId
                      ? { ...msg, content: finalAnswer, sources: finalSources }
                      : msg
                  )
                );
                setIsLoading(false); // 收到最终答案，停止加载
              } catch (e) {
                console.error("解析最终响应失败:", e, "原始数据:", eventData);
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === newAiMessageId
                      ? { ...msg, content: "解析最终响应失败。" }
                      : msg
                  )
                );
                setIsLoading(false);
              }
            }
            currentEvent = {};
          }
        }
      }
    } catch (error) {
      const errorMessage = `错误: ${(error as Error).message}`;
      console.error("发送消息失败:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === currentAiMessageId ? { ...msg, content: errorMessage } : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="flex items-center justify-between p-4 bg-white shadow-md">
        <h1 className="text-xl font-bold text-gray-800">DeepSearch AI Assistant</h1>
        {sessionId && (
          <span className="text-sm text-gray-500">Session ID: {sessionId.substring(0, 8)}...</span>
        )}
      </header>
      <main className="flex-1 flex overflow-hidden">
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatMessagesView messages={messages} />
          {isLoading && (
            <div className="flex justify-center items-center p-2">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              <p className="ml-2 text-gray-600">代理思考中...</p>
            </div>
          )}
        </div>
        <div className="w-1/3 p-4 border-l border-gray-200 bg-white overflow-hidden">
          <ActivityTimeline activities={activityLogs} />
        </div>
      </main>
      <footer>
        <InputForm onSendMessage={handleSendMessage} isLoading={isLoading} />
      </footer>
    </div>
  );
}

export default App;
