export default function Section({ title, tag, children }) {
  return (
    <div className="panel">
      <h2>
        {title}
        {tag && <span className="tag">{tag}</span>}
      </h2>
      {children}
    </div>
  );
}
