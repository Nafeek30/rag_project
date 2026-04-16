import { useDeferredValue } from "react";

import type { ChatMessage } from "../../types/chat";
import { ChatTranscript } from "./ChatTranscript";
import { MessageComposer } from "./MessageComposer";
import styles from "./ChatSurface.module.css";

interface ChatSurfaceProps {
  appTitle: string;
  errorMessage: string | null;
  isSending: boolean;
  messages: ChatMessage[];
  onSendMessage: (value: string) => Promise<void>;
  onToggleSidebar: () => void;
}

export function ChatSurface({
  appTitle,
  errorMessage,
  isSending,
  messages,
  onSendMessage,
  onToggleSidebar,
}: ChatSurfaceProps) {
  const deferredMessages = useDeferredValue(messages);

  return (
    <section className={styles.surface}>
      <header className={styles.header}>
        <div className={styles.headerGroup}>
          <button className={styles.menuButton} onClick={onToggleSidebar} type="button">
            Menu
          </button>
          <div>
            <p className={styles.eyebrow}>Conversational retrieval UI</p>
            <h2 className={styles.title}>{appTitle}</h2>
          </div>
        </div>
      </header>

      <div className={styles.content}>
        <ChatTranscript messages={deferredMessages} />
      </div>

      <footer className={styles.composerArea}>
        {errorMessage ? <div className={styles.errorBanner}>{errorMessage}</div> : null}
        <MessageComposer isSending={isSending} onSubmit={onSendMessage} />
        <p className={styles.footerNote}>Enter sends the message. Shift+Enter inserts a new line.</p>
      </footer>
    </section>
  );
}
