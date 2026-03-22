import { useEffect, useRef, useState } from "react";

import styles from "./MessageComposer.module.css";

interface MessageComposerProps {
  isSending: boolean;
  onSubmit: (value: string) => Promise<void>;
}

export function MessageComposer({ isSending, onSubmit }: MessageComposerProps) {
  const [draft, setDraft] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;

    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 240)}px`;
  }, [draft]);

  async function handleSubmit() {
    const trimmedDraft = draft.trim();

    if (!trimmedDraft || isSending) {
      return;
    }

    setDraft("");
    await onSubmit(trimmedDraft);
  }

  return (
    <form
      className={styles.form}
      onSubmit={(event) => {
        event.preventDefault();
        void handleSubmit();
      }}
    >
      <label className={styles.label} htmlFor="chat-composer">
        Message the assistant
      </label>

      <textarea
        className={styles.textarea}
        id="chat-composer"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void handleSubmit();
          }
        }}
        placeholder="Ask about course material, uploaded sources, or retrieval results..."
        ref={textareaRef}
        rows={1}
        value={draft}
      />

      <div className={styles.actions}>
        <button className={styles.submitButton} disabled={!draft.trim() || isSending} type="submit">
          {isSending ? "Thinking..." : "Send"}
        </button>
      </div>
    </form>
  );
}
