const messageTimeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "numeric",
  minute: "2-digit",
});

const conversationDateFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
});

export function formatMessageTimestamp(timestamp: string) {
  return messageTimeFormatter.format(new Date(timestamp));
}

export function formatConversationTimestamp(timestamp: string) {
  const date = new Date(timestamp);
  const millisecondsSince = Date.now() - date.getTime();
  const hoursSince = Math.round(millisecondsSince / (1000 * 60 * 60));

  if (hoursSince < 1) {
    return "Now";
  }

  if (hoursSince < 24) {
    return `${hoursSince}h`;
  }

  return conversationDateFormatter.format(date);
}

