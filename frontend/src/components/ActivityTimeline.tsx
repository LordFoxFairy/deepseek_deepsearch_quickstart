import React from 'react';
import { ScrollArea } from '../components/ui/scroll-area';
import { CheckCircle, Zap } from 'lucide-react';

// 定义单个日志条目的类型
interface ActivityLog {
  id: string;
  message: string;
  timestamp: string;
}

interface ActivityTimelineProps {
  activities: ActivityLog[];
}

const ActivityTimeline: React.FC<ActivityTimelineProps> = ({ activities }) => {
  return (
    <div className="flex flex-col h-full">
      <h3 className="text-lg font-semibold text-gray-800 mb-4 pb-2 border-b">全局实时日志</h3>
      <ScrollArea className="flex-1 -mr-4 pr-4">
        <div className="space-y-4">
          {activities.length === 0 ? (
            <div className="text-center text-gray-500 mt-8">
              <Zap className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm">等待任务开始...</p>
              <p className="text-xs text-gray-400">AI 的详细思考过程将在这里实时显示。</p>
            </div>
          ) : (
            activities.map((activity, index) => (
              <div key={activity.id} className="flex items-start">
                <div className="flex flex-col items-center mr-4">
                  <div className="flex items-center justify-center w-8 h-8 bg-blue-100 rounded-full">
                    <CheckCircle className="w-5 h-5 text-blue-600" />
                  </div>
                  {index < activities.length - 1 && (
                    <div className="w-px h-6 bg-gray-200 mt-1"></div>
                  )}
                </div>
                <div className="flex-1 pt-1">
                  <p className="text-sm text-gray-700 break-words">
                    {activity.message}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {activity.timestamp}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default ActivityTimeline;
