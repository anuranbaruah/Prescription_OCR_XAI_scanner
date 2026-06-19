import Icon from "./Icon";

export default function Recommendations({ recommendations }) {
  if (!recommendations?.length)
    return <p className="muted">No generic alternatives found for the identified medicines.</p>;

  return (
    <div>
      {recommendations.map((r, i) => (
        <div className="rec-card" key={i}>
          <div className="rec-head">
            <div>
              <span className="name">{r.prescribed}</span>{" "}
              <span className="comp">· {r.composition}</span>
            </div>
            {r.prescribed_price != null && (
              <span className="price">₹{r.prescribed_price.toFixed(2)}</span>
            )}
          </div>
          {r.alternatives.map((a, j) => (
            <div className="alt" key={j}>
              <div>
                <div className="alt-name">
                  {a.name}
                  {j === 0 && (
                    <span className="best-badge">
                      <Icon name="check" size={12} strokeWidth={2.5} /> Best price
                    </span>
                  )}
                </div>
                <div className="comp">{a.manufacturer}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="price">₹{a.price.toFixed(2)}</div>
                {a.saving_pct > 0 && <div className="save">save {a.saving_pct}%</div>}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
