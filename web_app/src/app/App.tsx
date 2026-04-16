import { useState } from "react";

import { ChatSurface } from "../components/chat/ChatSurface";
import { ConversationSidebar } from "../components/layout/ConversationSidebar";
import { appConfig } from "../config/env";
import { useChatSession } from "../hooks/useChatSession";
import { chatService } from "../services/chatApi";
import styles from "./App.module.css";

export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const {
    activeConversationId,
    conversations,
    errorMessage,
    isSending,
    messages,
    selectConversation,
    sendMessage,
    startNewConversation,
  } = useChatSession(chatService);

  function handleSelectConversation(conversationId: string) {
    selectConversation(conversationId);
    setIsSidebarOpen(false);
  }

  function handleStartNewConversation() {
    startNewConversation();
    setIsSidebarOpen(false);
  }

  return (
    <div className={styles.appShell}>
      <div className={styles.frame}>
        <ConversationSidebar
          activeConversationId={activeConversationId}
          conversations={conversations}
          isOpen={isSidebarOpen}
          onClose={() => setIsSidebarOpen(false)}
          onSelectConversation={handleSelectConversation}
          onStartNewConversation={handleStartNewConversation}
        />

        <main className={styles.main}>
          <ChatSurface
            appTitle={appConfig.appTitle}
            errorMessage={errorMessage}
            isSending={isSending}
            messages={messages}
            onSendMessage={sendMessage}
            onToggleSidebar={() => setIsSidebarOpen((currentValue) => !currentValue)}
          />
        </main>
      </div>
    </div>
  );
}
