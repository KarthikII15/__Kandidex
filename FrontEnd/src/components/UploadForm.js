import React, { useState } from 'react';
import axios from 'axios';
import './UploadForm.css';

function UploadForm() {
  const [transcriptFile, setTranscriptFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [role, setRole] = useState('');
  const [skills, setSkills] = useState('');
  const [loading, setLoading] = useState(false);
  const [emailResult, setEmailResult] = useState('');
  const [editableEmail, setEditableEmail] = useState('');
  const [editMode, setEditMode] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setEmailResult('');

    const formData = new FormData();
    formData.append('transcript', transcriptFile);
    formData.append('test', testFile);
    formData.append('role', role);
    formData.append('skills_to_rate', skills);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData);
      // 🛠 FIX: Use response.data directly instead of response.data.result
      setEmailResult(response.data.result);
      setEditableEmail(response.data.result);

    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred while processing your request. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">

        {/* Header */}
      <header className="app-header">
        <img src="Moyivity_labs.png" alt="Company Logo" className="logo" />
      </header>

      <div className="upload-form">
        <div className="logo-container">
            <img src="productlogo1.jpeg" alt="Company Logo" className="logo" />
        </div>
        {/* <h2>Kandidex</h2> */}
        {/* <img src="productlogo1.jpeg" alt="Company Logo" className="logo" /> */}
        <form onSubmit={handleSubmit}>
          <label>Candidate Role:</label>
          <input type="text" value={role} onChange={(e) => setRole(e.target.value)} required />

          <label>Skills to Rate (comma-separated):</label>
          <input type="text" value={skills} onChange={(e) => setSkills(e.target.value)} />

          <label>Upload Transcript:</label>
          <input type="file" accept=".txt,.pdf,.docx" onChange={(e) => setTranscriptFile(e.target.files[0])} required />

          <label>Upload Test Document:</label>
          <input type="file" accept=".txt,.pdf,.docx" onChange={(e) => setTestFile(e.target.files[0])} required />

          <button type="submit" disabled={loading}>
            {loading ? 'Generating...' : 'Submit'}
          </button>
        </form>

        {loading && <p>⏳ Processing... Please wait.</p>}
      </div>

      {/* Separated Output Block */}
      {emailResult && (
        <div className="email-output">
        <h3>📧 Generated Email:</h3>

        {editMode ? (
      <textarea
        value={editableEmail}
        onChange={(e) => setEditableEmail(e.target.value)}
        rows={20}
        style={{ width: '100%', fontFamily: 'monospace' }}
      />
    ) : (
      <pre>{editableEmail}</pre>
    )}

    <div className="email-actions">
      <button onClick={() => setEditMode(!editMode)}>
        {editMode ? 'Save' : 'Edit'}
      </button>
      <button onClick={() => {
        navigator.clipboard.writeText(editableEmail);
        alert("Copied to clipboard!");
      }}>
        Copy
      </button>
      <button onClick={() => {
        const blob = new Blob([editableEmail], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'generated_email.txt';
        link.click();
      }}>
        Download
      </button>

      <button
  onClick={() => {
    const subject = encodeURIComponent("Interview Evaluation Summary");
    const body = encodeURIComponent(editableEmail);
    const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=&su=${subject}&body=${body}`;
    window.open(gmailUrl, "_blank");}}>
  Send via Gmail
    </button>


    </div>
  </div>
)}
    {/* Footer */}
    <footer className="app-footer">
        <p>© 2025 Motivity_labs. | Dallas Center, 6th & 7th Floor, 83/1, Plot No A1, Knowledge City Rd, Rai Durg, Hyderabad, Telangana, India 500032.</p>
      </footer>

    </div>
  );
}

export default UploadForm;
