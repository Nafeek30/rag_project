import { startTransition, useEffect, useMemo, useReducer, useRef } from "react";

import type { ChatMessage, ChatRequest, ChatService, ConversationSummary } from "../types/chat";

interface ConversationRecord {
  id: string;
  messages: ChatMessage[];
  summary: ConversationSummary;
}

interface ChatState {
  activeConversationId: string;
  conversations: Record<string, ConversationRecord>;
  errorMessage: string | null;
  isSending: boolean;
  orderedConversationIds: string[];
}

type ChatAction =
  | { type: "conversation/create"; payload: { conversationId: string; greeting: ChatMessage } }
  | { type: "conversation/select"; payload: { conversationId: string } }
  | {
      type: "conversation/queueTurn";
      payload: { assistantPlaceholder: ChatMessage; conversationId: string; userMessage: ChatMessage };
    }
  | {
      type: "conversation/resolveTurn";
      payload: { conversationId: string; placeholderId: string; reply: ChatMessage };
    }
  | {
      type: "conversation/rejectTurn";
      payload: { conversationId: string; errorMessage: string; placeholderId: string; reply: ChatMessage };
    };

function createId(prefix: string) {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}_${crypto.randomUUID()}`;
  }

  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function truncate(value: string, length: number) {
  if (value.length <= length) {
    return value;
  }

  return `${value.slice(0, length - 1)}...`;
}

function createMessage(
  role: ChatMessage["role"],
  content: string,
  state: ChatMessage["state"] = "idle",
): ChatMessage {
  return {
    id: createId(role),
    role,
    content,
    createdAt: new Date().toISOString(),
    state,
  };
}

function createGreetingMessage() {
  return createMessage(
    "assistant",
    "Hello. I can help with course content, research notes, and retrieval-backed questions. Ask a question to start the session.",
  );
}

function buildConversationSummary(conversationId: string, messages: ChatMessage[]): ConversationSummary {
  const firstUserMessage = messages.find((message) => message.role === "user");
  const lastMessage = messages[messages.length - 1];

  return {
    id: conversationId,
    title: truncate(firstUserMessage?.content ?? "New conversation", 40),
    preview: truncate(lastMessage?.content ?? "Ready for the first prompt.", 78),
    updatedAt: lastMessage?.createdAt ?? new Date().toISOString(),
  };
}

function buildConversationRecord(conversationId: string, messages: ChatMessage[]): ConversationRecord {
  return {
    id: conversationId,
    messages,
    summary: buildConversationSummary(conversationId, messages),
  };
}

function moveConversationToFront(orderedConversationIds: string[], conversationId: string) {
  return [conversationId, ...orderedConversationIds.filter((currentId) => currentId !== conversationId)];
}

const initialConversationId = createId("conversation");
const initialGreeting = createGreetingMessage();
const initialConversation = buildConversationRecord(initialConversationId, [initialGreeting]);

const initialState: ChatState = {
  activeConversationId: initialConversationId,
  conversations: {
    [initialConversationId]: initialConversation,
  },
  errorMessage: null,
  isSending: false,
  orderedConversationIds: [initialConversationId],
};

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "conversation/create": {
      const { conversationId, greeting } = action.payload;
      const nextRecord = buildConversationRecord(conversationId, [greeting]);

      return {
        ...state,
        activeConversationId: conversationId,
        conversations: {
          ...state.conversations,
          [conversationId]: nextRecord,
        },
        errorMessage: null,
        orderedConversationIds: moveConversationToFront(state.orderedConversationIds, conversationId),
      };
    }

    case "conversation/select":
      return {
        ...state,
        activeConversationId: action.payload.conversationId,
        errorMessage: null,
      };

    case "conversation/queueTurn": {
      const { assistantPlaceholder, conversationId, userMessage } = action.payload;
      const conversation = state.conversations[conversationId];

      if (!conversation) {
        return state;
      }

      const nextMessages = [...conversation.messages, userMessage, assistantPlaceholder];
      const nextRecord = buildConversationRecord(conversationId, nextMessages);

      return {
        ...state,
        conversations: {
          ...state.conversations,
          [conversationId]: nextRecord,
        },
        errorMessage: null,
        isSending: true,
        orderedConversationIds: moveConversationToFront(state.orderedConversationIds, conversationId),
      };
    }

    case "conversation/resolveTurn": {
      const { conversationId, placeholderId, reply } = action.payload;
      const conversation = state.conversations[conversationId];

      if (!conversation) {
        return {
          ...state,
          isSending: false,
        };
      }

      const nextMessages = conversation.messages.map((message) =>
        message.id === placeholderId ? reply : message,
      );
      const nextRecord = buildConversationRecord(conversationId, nextMessages);

      return {
        ...state,
        conversations: {
          ...state.conversations,
          [conversationId]: nextRecord,
        },
        isSending: false,
        orderedConversationIds: moveConversationToFront(state.orderedConversationIds, conversationId),
      };
    }

    case "conversation/rejectTurn": {
      const { conversationId, errorMessage, placeholderId, reply } = action.payload;
      const conversation = state.conversations[conversationId];

      if (!conversation) {
        return {
          ...state,
          errorMessage,
          isSending: false,
        };
      }

      const nextMessages = conversation.messages.map((message) =>
        message.id === placeholderId ? reply : message,
      );
      const nextRecord = buildConversationRecord(conversationId, nextMessages);

      return {
        ...state,
        conversations: {
          ...state.conversations,
          [conversationId]: nextRecord,
        },
        errorMessage,
        isSending: false,
        orderedConversationIds: moveConversationToFront(state.orderedConversationIds, conversationId),
      };
    }

    default:
      return state;
  }
}

export function useChatSession(chatService: ChatService) {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const messages = state.conversations[state.activeConversationId]?.messages ?? [];
  const conversations = useMemo(
    () =>
      state.orderedConversationIds
        .map((conversationId) => state.conversations[conversationId])
        .filter(Boolean)
        .map((conversation) => conversation.summary),
    [state.conversations, state.orderedConversationIds],
  );

  function startNewConversation() {
    const conversationId = createId("conversation");
    const greeting = createGreetingMessage();

    dispatch({
      type: "conversation/create",
      payload: {
        conversationId,
        greeting,
      },
    });
  }

  function selectConversation(conversationId: string) {
    if (!state.conversations[conversationId]) {
      return;
    }

    dispatch({
      type: "conversation/select",
      payload: {
        conversationId,
      },
    });
  }

  async function sendMessage(value: string) {
    const trimmedValue = value.trim();
    const currentState = stateRef.current;

    if (!trimmedValue || currentState.isSending) {
      return;
    }

    const conversationId = currentState.activeConversationId;
    const userMessage = createMessage("user", trimmedValue);
    const assistantPlaceholder = createMessage("assistant", "", "pending");

    startTransition(() => {
      dispatch({
        type: "conversation/queueTurn",
        payload: {
          assistantPlaceholder,
          conversationId,
          userMessage,
        },
      });
    });

    const currentConversation = currentState.conversations[conversationId];

    if (!currentConversation) {
      return;
    }

    const history: ChatRequest["history"] = [...currentConversation.messages, userMessage]
      .filter((message) => message.role !== "system" && message.state !== "pending")
      .map((message) => ({
        role: message.role === "assistant" ? "assistant" : "user",
        content: message.content,
      }));

    try {
      const response = await chatService.sendMessage({
        conversationId,
        history,
        message: trimmedValue,
      });

      const reply: ChatMessage = {
        ...response.message,
        id: response.message.id || createId("assistant"),
        role: "assistant",
        state: "idle",
      };

      startTransition(() => {
        dispatch({
          type: "conversation/resolveTurn",
          payload: {
            conversationId,
            placeholderId: assistantPlaceholder.id,
            reply,
          },
        });
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to reach the chat service.";
      const failedReply: ChatMessage = {
        ...createMessage(
          "assistant",
          "The assistant could not complete that request. Check the backend configuration and try again.",
          "error",
        ),
      };

      startTransition(() => {
        dispatch({
          type: "conversation/rejectTurn",
          payload: {
            conversationId,
            errorMessage: message,
            placeholderId: assistantPlaceholder.id,
            reply: failedReply,
          },
        });
      });
    }
  }

  return {
    activeConversationId: state.activeConversationId,
    conversations,
    errorMessage: state.errorMessage,
    isSending: state.isSending,
    messages,
    selectConversation,
    sendMessage,
    startNewConversation,
  };
}
