export function MessageThread({ messages }: { messages: any[] }) {
  return <div className="thread">
    <div className="thread-date">CASE CONVERSATION</div>
    {messages.length === 0 ? <div className="system-note"><strong>PROTOCOL STAGING</strong>No messages have been recorded for this case yet.</div> : messages.map((message) => {
      const sentAt = new Intl.DateTimeFormat("en", { hour: "numeric", minute: "2-digit" }).format(new Date(message.created_at));
      const label = message.direction === "customer" ? "Customer" : message.direction === "staff" ? "You" : "Assistant";
      return <article className={`message ${message.direction}`} key={message.message_id}><div className="message-header"><b>{label}</b><span>{sentAt} · {message.delivery_status}</span></div><p>{message.content}</p></article>;
    })}
  </div>;
}
