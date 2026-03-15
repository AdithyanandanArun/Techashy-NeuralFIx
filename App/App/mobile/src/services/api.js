import { ENDPOINTS } from '../utils/config';

// ─── Generic fetch wrapper ────────────────────────────────────────────────────
async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${response.status}`);
  }
  return response.json();
}

// ─── Health ───────────────────────────────────────────────────────────────────
export async function checkHealth() {
  return apiFetch(ENDPOINTS.health);
}

// ─── Sessions ─────────────────────────────────────────────────────────────────
export async function createSession(title = 'New Session') {
  return apiFetch(ENDPOINTS.sessions, {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
}

export async function listSessions() {
  return apiFetch(ENDPOINTS.sessions);
}

export async function getSession(sessionId) {
  return apiFetch(`${ENDPOINTS.sessions}/${sessionId}`);
}

export async function deleteSession(sessionId) {
  return apiFetch(`${ENDPOINTS.sessions}/${sessionId}`, { method: 'DELETE' });
}

export async function updateSessionStatus(sessionId, status) {
  return apiFetch(`${ENDPOINTS.sessions}/${sessionId}/status?status=${status}`, {
    method: 'PATCH',
  });
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
export async function sendChatMessage(sessionId, message) {
  return apiFetch(ENDPOINTS.chat, {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, message }),
  });
}

// ─── Image Upload ─────────────────────────────────────────────────────────────
export async function uploadImage(sessionId, imageUri, filename = 'equipment.jpg') {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('file', {
    uri: imageUri,
    name: filename,
    type: 'image/jpeg',
  });

  const response = await fetch(ENDPOINTS.imageUpload, {
    method: 'POST',
    body: formData,
    // Don't set Content-Type header — fetch sets it automatically with boundary for multipart
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${response.status}`);
  }
  return response.json();
}

// ─── Reports ──────────────────────────────────────────────────────────────────
export async function generateReport(sessionId) {
  return apiFetch(ENDPOINTS.reportGenerate, {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function getReport(sessionId) {
  return apiFetch(ENDPOINTS.reportGet(sessionId));
}

// ─── RAG ──────────────────────────────────────────────────────────────────────
export async function getRagStatus() {
  return apiFetch(ENDPOINTS.ragStatus);
}
