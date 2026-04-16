import { useEffect, useRef } from "react";

import type { ChatMessage } from "../../types/chat";
import { formatMessageTimestamp } from "../../utils/formatters";
import styles from "./ChatTranscript.module.css";

interface ChatTranscriptProps {
  messages: ChatMessage[];
}

export function ChatTranscript({ messages }: ChatTranscriptProps) {
  const endAnchorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <div className={styles.transcript}>
      {messages.map((message) => {
        const isAssistant = message.role === "assistant";
        const isPending = message.state === "pending";
        const isError = message.state === "error";

        return (
          <article
            className={`${styles.messageRow} ${isAssistant ? styles.assistantRow : styles.userRow}`}
            key={message.id}
          >
            <div
              className={`${styles.messageCard} ${
                isAssistant ? styles.assistantCard : styles.userCard
              } ${isError ? styles.errorCard : ""}`}
            >
              <div className={styles.messageHeader}>
                <span className={styles.roleLabel}>{isAssistant ? "Assistant" : "You"}</span>
                <span className={styles.timestamp}>{formatMessageTimestamp(message.createdAt)}</span>
              </div>

              {isPending ? (
                <div aria-label="Assistant is thinking" className={styles.typingIndicator}>
                  <span />
                  <span />
                  <span />
                </div>
              ) : (
                <p className={styles.messageContent}>{message.content}</p>
              )}

              {message.sources && message.sources.length > 0 ? (
                <div className={styles.sourceList}>
                  {message.sources.map((source) =>
                    source.url ? (
                      <a
                        className={styles.sourceChip}
                        href={source.url}
                        key={source.id}
                        rel="noreferrer"
                        target="_blank"
                      >
                        <strong>{source.title}</strong>
                        <span>{source.excerpt}</span>
                      </a>
                    ) : (
                      <div className={styles.sourceChip} key={source.id}>
                        <strong>{source.title}</strong>
                        <span>{source.excerpt}</span>
                      </div>
                    ),
                  )}
                </div>
              ) : null}
            </div>
          </article>
        );
      })}

      <div ref={endAnchorRef} />
    </div>
  );
}
