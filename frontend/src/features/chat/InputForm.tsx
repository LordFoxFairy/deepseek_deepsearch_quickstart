import * as React from "react";
import { SendHorizonalIcon } from "lucide-react";

import { Button } from "../../components/ui/button.tsx";
import { Textarea } from "../../components/ui/textarea";

// 定义 InputForm 组件的 props 类型
interface InputFormProps {
  onSendMessage: (message: string) => void; // 发送消息的回调函数
  isLoading: boolean; // 指示是否正在加载（发送中）
}

// InputForm 组件
const InputForm: React.FC<InputFormProps> = ({ onSendMessage, isLoading }) => {
  const [message, setMessage] = React.useState<string>(""); // 存储用户输入的消息

  // 处理文本域内容变化
  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(event.target.value);
  };

  // 处理发送按钮点击或回车键
  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); // 阻止表单默认提交行为
    if (message.trim() && !isLoading) {
      onSendMessage(message.trim()); // 调用父组件传入的发送消息函数
      setMessage(""); // 清空输入框
    }
  };

  // 允许按 Enter 键发送消息，Shift + Enter 换行
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      handleSubmit(event as unknown as React.FormEvent<HTMLFormElement>); // 触发提交
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center space-x-2 p-4 border-t border-gray-200 bg-white">
      <Textarea
        placeholder="输入你的问题..."
        value={message}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        disabled={isLoading} // 发送中时禁用输入
        className="flex-1 resize-none pr-10" // pr-10 为图标留出空间
        rows={1} // 初始行数
      />
      <Button type="submit" size="icon" disabled={!message.trim() || isLoading}>
        {isLoading ? (
          <span className="animate-spin h-5 w-5 border-t-2 border-b-2 border-primary-foreground rounded-full"></span> // 加载动画
        ) : (
          <SendHorizonalIcon className="h-5 w-5" /> // 发送图标
        )}
      </Button>
    </form>
  );
};

export default InputForm;
