import * as React from "react";
import { ScrollArea } from "./ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Dot } from "lucide-react";
import { renderMarkdownWithLinks } from "../utils/markdown_parser";

// Define activity log type
interface ActivityLog {
  step_name: string; // Step name (e.g., "supervisor", "planner", "executor")
  output: string;    // Output or brief description of the step
  timestamp: string; // Timestamp of the activity
}

// Define ActivityTimeline component props type
interface ActivityTimelineProps {
  activities: ActivityLog[]; // List of activity logs
}

// ActivityTimeline component
const ActivityTimeline: React.FC<ActivityTimelineProps> = ({ activities }) => {
  const timelineEndRef = React.useRef<HTMLDivElement>(null); // Used for auto-scrolling to bottom

  // Auto-scroll to bottom when activities update
  React.useEffect(() => {
    if (timelineEndRef.current) {
      timelineEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [activities]);

  return (
    <Card className="w-full h-full flex flex-col">
      <CardHeader className="p-4 border-b">
        <CardTitle className="text-lg">代理活动时间线</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 p-4 overflow-hidden">
        <ScrollArea className="h-full pr-4"> {/* Add right padding to avoid scrollbar obscuring content */}
          <div className="relative pl-6"> {/* Left padding for timeline markers */}
            {activities.map((activity, index) => (
              <div key={index} className="mb-4 relative">
                {/* 时间线连接线 */}
                {index < activities.length - 1 && (
                  <div className="absolute left-2 top-0 h-full w-0.5 bg-gray-300"></div>
                )}
                {/* 时间线圆点标记 */}
                <div className="absolute left-0 top-1.5 flex items-center justify-center w-5 h-5 rounded-full bg-blue-500 text-white z-10">
                  <Dot className="h-4 w-4" />
                </div>

                <div className="ml-6">
                  <p className="text-sm font-semibold text-gray-800">{activity.step_name}</p>
                  <p className="text-xs text-gray-500">{activity.timestamp}</p>
                  {/* Use renderMarkdownWithLinks to parse and render Markdown content, preserving whitespace */}
                  <div className="text-sm text-gray-700 mt-1 whitespace-pre-wrap">
                    {renderMarkdownWithLinks(activity.output)}
                  </div>
                </div>
              </div>
            ))}
            <div ref={timelineEndRef} /> {/* Scroll target */}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default ActivityTimeline;