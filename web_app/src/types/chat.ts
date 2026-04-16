export type MessageRole = "assistant" | "system" | "user";
export type DeliveryState = "error" | "idle" | "pending";

export interface SourceReference {
  id: string;
  title: string;
  excerpt: string;
  url?: string;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  sources?: SourceReference[];
  state: DeliveryState;
}

export interface ConversationSummary {
  id: string;
  title: string;
  preview: string;
  updatedAt: string;
}

export interface ChatRequest {
  conversationId: string;
  message: string;
  history: Array<{
    role: Exclude<MessageRole, "system">;
    content: string;
  }>;
}

export interface ChatResponse {
  conversationId: string;
  message: ChatMessage;
  sources?: SourceReference[];
}

export interface ChatService {
  sendMessage: (request: ChatRequest) => Promise<ChatResponse>;
}

