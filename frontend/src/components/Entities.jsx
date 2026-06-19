import Icon from "./Icon";

export default function Entities({ entities }) {
  if (!entities?.length) return <p className="muted">No medicine entities identified.</p>;
  return (
    <div>
      {entities.map((e, i) => (
        <span className="pill" key={i} title={`NER score ${e.score}`}>
          <Icon name="pill" size={14} />
          {e.matched_name || e.text}
          {e.matched_name && e.match_score != null && (
            <small> · match {e.match_score}%</small>
          )}
        </span>
      ))}
    </div>
  );
}
