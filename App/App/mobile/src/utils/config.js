// ─────────────────────────────────────────────────────────
//  NetFixAI — Server Configuration
//  Change API_BASE_URL to your server machine's local IP.
//  Run the backend and it will print the correct URL for you.
// ─────────────────────────────────────────────────────────

// 👇 CHANGE THIS to your server's local IP (shown when you run `python run.py`)
export const API_BASE_URL = 'http://192.168.1.100:8000';

export const ENDPOINTS = {
  health:         `${API_BASE_URL}/health`,
  sessions:       `${API_BASE_URL}/api/sessions`,
  chat:           `${API_BASE_URL}/api/chat`,
  imageUpload:    `${API_BASE_URL}/api/images/upload`,
  imageFile:      (filename) => `${API_BASE_URL}/api/images/file/${filename}`,
  reportGenerate: `${API_BASE_URL}/api/reports/generate`,
  reportGet:      (id) => `${API_BASE_URL}/api/reports/${id}`,
  ragStatus:      `${API_BASE_URL}/api/rag/status`,
  ragReindex:     `${API_BASE_URL}/api/rag/reindex`,
  ragUploadDoc:   `${API_BASE_URL}/api/rag/upload-doc`,
};
