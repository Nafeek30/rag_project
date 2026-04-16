import type { ConversationSummary } from "../../types/chat";
import { formatConversationTimestamp } from "../../utils/formatters";
import styles from "./ConversationSidebar.module.css";

interface ConversationSidebarProps {
  activeConversationId: string;
  conversations: ConversationSummary[];
  isOpen: boolean;
  onClose: () => void;
  onSelectConversation: (conversationId: string) => void;
  onStartNewConversation: () => void;
}

export function ConversationSidebar({
  activeConversationId,
  conversations,
  isOpen,
  onClose,
  onSelectConversation,
  onStartNewConversation,
}: ConversationSidebarProps) {
  return (
    <div className={`${styles.container} ${isOpen ? styles.open : ""}`}>
      <button
        aria-hidden={!isOpen}
        className={styles.backdrop}
        onClick={onClose}
        tabIndex={isOpen ? 0 : -1}
        type="button"
      />

      <aside className={styles.sidebar}>
        <div className={styles.brandBlock}>
          <div className={styles.brandBadge}>R</div>
          <div>
            <p className={styles.eyebrow}>Research assistant</p>
            <h1 className={styles.brandTitle}>RAG Workspace</h1>
          </div>
        </div>

        <button className={styles.primaryAction} onClick={onStartNewConversation} type="button">
          New conversation
        </button>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>Recent chats</span>
            <span className={styles.sectionMeta}>{conversations.length}</span>
          </div>

          <div className={styles.list}>
            {conversations.map((conversation) => {
              const isActive = conversation.id === activeConversationId;

              return (
                <button
                  className={`${styles.conversationCard} ${isActive ? styles.activeCard : ""}`}
                  key={conversation.id}
                  onClick={() => onSelectConversation(conversation.id)}
                  type="button"
                >
                  <div className={styles.cardTopRow}>
                    <strong className={styles.cardTitle}>{conversation.title}</strong>
                    <span className={styles.cardTime}>
                      {formatConversationTimestamp(conversation.updatedAt)}
                    </span>
                  </div>
                  <p className={styles.cardPreview}>{conversation.preview}</p>
                </button>
              );
            })}
          </div>
        </section>
      </aside>
    </div>
  );
}
