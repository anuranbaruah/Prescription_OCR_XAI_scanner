import Icon from "./Icon";

// Show the value, or "Not specified" when the model couldn't read it.
const val = (v) => (v && String(v).trim() ? String(v) : "Not specified");

export default function PrescriptionResult({ data }) {
  // Always render the card. When the AI could not extract anything (data is
  // missing), the fields fall back to "Not specified" rather than hiding the
  // whole section — the result view appears for every prescription.
  const d = data || {};
  const meds = d.medications || [];
  const hasNotes = d.notes && String(d.notes).trim();

  return (
    <div className="rx-result">
      <div className="rx-disclaimer" role="note">
        <Icon name="info" size={18} />
        <p>
          <strong>Disclaimer </strong>
          This analysis is for informational purposes only and should not replace
          professional medical advice. Always verify prescription details with your
          pharmacist or healthcare provider before taking any medication.
        </p>
      </div>

      <div className="rx-card">
        <div className="rx-card-head">
          <h3>Your Extracted Prescription Result</h3>
          <button
            className="btn secondary rx-export"
            onClick={() => window.print()}
            aria-label="Export prescription as PDF"
          >
            <Icon name="download" size={15} /> Export PDF
          </button>
        </div>

        <div className="rx-meta">
          <div>
            <span className="rx-label">Patient Name</span>
            <div className="rx-val">{val(d.patient_name)}</div>
          </div>
          <div>
            <span className="rx-label">Doctor Name</span>
            <div className="rx-val">{val(d.doctor_name)}</div>
          </div>
          <div>
            <span className="rx-label">Date</span>
            <div className="rx-val">{val(d.date)}</div>
          </div>
        </div>

        <h4 className="rx-section">Prescribed Medications</h4>
        {meds.length === 0 ? (
          <p className="muted">No medications could be read from this image.</p>
        ) : (
          meds.map((m, i) => (
            <div className="rx-med" key={i}>
              <div className="rx-med-name">{m.name}</div>
              <div className="rx-med-grid">
                <div>
                  <span className="rx-k">Dosage:</span> <span className="rx-v">{val(m.dosage)}</span>
                </div>
                <div>
                  <span className="rx-k">Frequency:</span>{" "}
                  <span className="rx-v rx-freq">{val(m.frequency)}</span>
                </div>
                <div>
                  <span className="rx-k">Duration:</span> <span className="rx-v">{val(m.duration)}</span>
                </div>
                <div>
                  <span className="rx-k">Instructions:</span>{" "}
                  <span className="rx-v">{val(m.instructions)}</span>
                </div>
              </div>
            </div>
          ))
        )}

        {hasNotes && (
          <>
            <h4 className="rx-section">Additional Notes</h4>
            <div className="rx-notes">{d.notes}</div>
          </>
        )}
      </div>
    </div>
  );
}
