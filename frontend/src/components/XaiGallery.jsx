export default function XaiGallery({ xai }) {
  if (!xai?.length)
    return <p className="muted">XAI explanations not available for this run.</p>;

  return (
    <div className="grid2">
      {xai.map((x, i) => (
        <div className="xai-card" key={i}>
          {x.image_base64 && (
            <div className="img-wrap">
              <img
                src={x.image_base64}
                alt={`${x.method} explanation for ${x.target_stage}: ${x.title}`}
                loading="lazy"
              />
            </div>
          )}
          <div className="meta">
            <div className="method">
              {x.method} · {x.target_stage}
            </div>
            <h4>{x.title}</h4>
            {x.note && <p>{x.note}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}
