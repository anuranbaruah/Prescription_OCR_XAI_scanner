function Table({ data }) {
  if (!data) return null;
  const hasRows = Array.isArray(data.rows) && data.rows.length > 0;
  return (
    <div style={{ marginBottom: 22 }}>
      <h4 style={{ margin: "0 0 8px" }}>{data.title}</h4>
      {data.note && (
        <p className="muted" style={{ margin: "0 0 8px", fontSize: 13 }}>
          {data.note}
        </p>
      )}
      {hasRows ? (
        <div className="table-wrap" style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                {data.columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, i) => (
                <tr key={i} className={data.best_row === i ? "best" : ""}>
                  {row.map((cell, j) => (
                    <td key={j}>{String(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ fontSize: 13 }}>
          Not measured yet — run <code>python -m app.eval.run_all</code>.
        </p>
      )}
    </div>
  );
}

function StatusBanner({ status, generatedAt, disclaimer }) {
  if (!status) return null;
  const label =
    status === "measured"
      ? "Measured results"
      : status === "partial"
      ? "Partially measured"
      : "Not measured";
  const color =
    status === "measured" ? "#137333" : status === "partial" ? "#b06000" : "#a50e0e";
  return (
    <div style={{ marginBottom: 14 }}>
      <span style={{ color, fontWeight: 600 }}>● {label}</span>
      {generatedAt && (
        <span className="muted" style={{ marginLeft: 8, fontSize: 12 }}>
          generated {generatedAt}
        </span>
      )}
      {disclaimer && (
        <p className="muted" style={{ margin: "6px 0 0", fontSize: 12.5 }}>
          {disclaimer}
        </p>
      )}
    </div>
  );
}

export default function BenchmarkTables({ benchmarks }) {
  if (!benchmarks) return <p className="muted">Loading benchmarks…</p>;
  const ocrTables =
    Array.isArray(benchmarks.ocr_datasets) && benchmarks.ocr_datasets.length > 0
      ? benchmarks.ocr_datasets
      : benchmarks.ocr
      ? [benchmarks.ocr]
      : [];
  return (
    <div>
      <StatusBanner
        status={benchmarks.status}
        generatedAt={benchmarks.generated_at}
        disclaimer={benchmarks.disclaimer}
      />
      {ocrTables.map((t, i) => (
        <Table
          key={i}
          data={{
            ...t,
            title: t.source
              ? `OCR — ${t.source} (n=${t.n_samples})`
              : t.title,
          }}
        />
      ))}
      <Table data={benchmarks.detection} />
      <Table data={benchmarks.ner} />
      <Table data={benchmarks.system} />
    </div>
  );
}
