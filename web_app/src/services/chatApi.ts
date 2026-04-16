import { appConfig } from "../config/env";
import type { ChatRequest, ChatResponse, ChatService, SourceReference } from "../types/chat";

const defaultFetch: typeof fetch = (input, init) => globalThis.fetch(input, init);

function createId(prefix: string) {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}_${crypto.randomUUID()}`;
  }

  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function normalizeSources(value: unknown): SourceReference[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const normalizedSources: SourceReference[] = [];

  value.forEach((item, index) => {
    if (!isRecord(item) || typeof item.title !== "string" || typeof item.excerpt !== "string") {
      return;
    }

    normalizedSources.push({
      id: typeof item.id === "string" ? item.id : `source_${index + 1}`,
      title: item.title,
      excerpt: item.excerpt,
      url: typeof item.url === "string" ? item.url : undefined,
    });
  });

  return normalizedSources;
}

function normalizeChatResponse(payload: unknown, conversationId: string): ChatResponse {
  if (!isRecord(payload)) {
    throw new Error("The API returned an invalid response payload.");
  }

  const rawMessage = isRecord(payload.message) ? payload.message : null;
  const rawText =
    typeof payload.reply === "string"
      ? payload.reply
      : typeof payload.answer === "string"
        ? payload.answer
        : typeof rawMessage?.content === "string"
          ? rawMessage.content
          : typeof payload.message === "string"
            ? payload.message
            : null;

  if (!rawText || !rawText.trim()) {
    throw new Error("The API returned an empty answer.");
  }

  const sources = normalizeSources(payload.sources ?? rawMessage?.sources);

  return {
    conversationId:
      typeof payload.conversation_id === "string"
        ? payload.conversation_id
        : typeof payload.conversationId === "string"
          ? payload.conversationId
          : conversationId,
    message: {
      id: typeof rawMessage?.id === "string" ? rawMessage.id : createId("assistant"),
      role: "assistant",
      content: rawText.trim(),
      createdAt:
        typeof rawMessage?.created_at === "string"
          ? rawMessage.created_at
          : typeof rawMessage?.createdAt === "string"
            ? rawMessage.createdAt
            : new Date().toISOString(),
      sources,
      state: "idle",
    },
    sources,
  };
}

class FastApiChatService implements ChatService {
  constructor(
    private readonly baseUrl: string,
    private readonly fetcher: typeof fetch = defaultFetch,
  ) {}

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.fetcher(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        conversation_id: request.conversationId,
        history: request.history,
        message: request.message,
      }),
    });

    if (!response.ok) {
      const responseText = await response.text();
      throw new Error(responseText || `Request failed with status ${response.status}.`);
    }

    return normalizeChatResponse(await response.json(), request.conversationId);
  }
}

function generateMockReply(message: string) {
  const normalizedMessage = message.toLowerCase();

  if (normalizedMessage.includes("summarize")) {
    return "Here is a concise summary: identify the core claim, extract supporting evidence, and surface any open questions that should be validated against the underlying documents.";
  }

  if (normalizedMessage.includes("compare")) {
    return "A strong compare-and-contrast response should line up both sources on claims, evidence quality, terminology, and any unresolved disagreement that would matter to the user.";
  }

  if (normalizedMessage.includes("lecture") || normalizedMessage.includes("study")) {
    return "A study-oriented answer would highlight the main topic, list the core concepts, then propose a few targeted follow-up questions to test comprehension.";
  }

  return "The UI is currently backed by a mock service.";
}

class MockChatService implements ChatService {
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    await new Promise((resolve) => {
      window.setTimeout(resolve, 900);
    });

    return {
      conversationId: request.conversationId,
      message: {
        id: createId("assistant"),
        role: "assistant",
        content: generateMockReply(request.message),
        createdAt: new Date().toISOString(),
        sources: [
          {
            id: "source_mock_1",
            title: "Retrieval trace preview",
            excerpt: "Source chips render here so the UI is ready for backend citations.",
            url: undefined,
          },
          {
            id: "source_mock_2",
            title: "FastAPI handoff",
            excerpt: "Switch `VITE_USE_MOCK_API` to false to target the real API endpoint.",
            url: undefined,
          }
        ],
        state: "idle",
      },
      sources: [],
    };
  }
}

export function createChatService(): ChatService {
  if (appConfig.useMockApi) {
    return new MockChatService();
  }

  return new FastApiChatService(appConfig.apiBaseUrl);
}

export const chatService = createChatService();
