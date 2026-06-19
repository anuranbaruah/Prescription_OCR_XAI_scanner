import Icon from "./Icon";

const SEV_ICON = { major: "alertOctagon", moderate: "alertTriangle", minor: "info" };

export default function Interactions({ interactions }) {
  if (!interactions?.length)
    return <p className="muted">No known interactions detected among the identified medicines.</p>;

  return (
    <div>
      {interactions.map((it, i) => (
        <div className={`interaction ${it.severity}`} key={i}>
          <div className="ix-head">
            <span className="ix-pair">
              <Icon name={SEV_ICON[it.severity] || "alertTriangle"} size={17} />
              {it.drug_a} <span className="muted">+</span> {it.drug_b}
            </span>
            <span className={`sev ${it.severity}`}>{it.severity}</span>
          </div>
          <p>{it.description}</p>
        </div>
      ))}
    </div>
  );
}
