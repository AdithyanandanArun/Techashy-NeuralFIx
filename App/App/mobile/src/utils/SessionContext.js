import React, { createContext, useContext, useState, useCallback } from 'react';
import * as api from '../services/api';

const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [serverOnline, setServerOnline] = useState(null); // null=unknown, true, false

  // ── Server health ────────────────────────────────────────────────────────
  const checkServer = useCallback(async () => {
    try {
      await api.checkHealth();
      setServerOnline(true);
      return true;
    } catch {
      setServerOnline(false);
      return false;
    }
  }, []);

  // ── Sessions ─────────────────────────────────────────────────────────────
  const loadSessions = useCallback(async () => {
    try {
      const data = await api.listSessions();
      setSessions(data.sessions || []);
    } catch (e) {
      console.error('loadSessions error:', e);
    }
  }, []);

  const createSession = useCallback(async (title = 'New Session') => {
    const session = await api.createSession(title);
    setSessions(prev => [session, ...prev]);
    setActiveSessionId(session.id);
    return session.id;
  }, []);

  const deleteSession = useCallback(async (sessionId) => {
    await api.deleteSession(sessionId);
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (activeSessionId === sessionId) setActiveSessionId(null);
  }, [activeSessionId]);

  const refreshSession = useCallback(async (sessionId) => {
    try {
      const updated = await api.getSession(sessionId);
      setSessions(prev => prev.map(s => s.id === sessionId ? updated : s));
      return updated;
    } catch (e) {
      console.error('refreshSession error:', e);
      return null;
    }
  }, []);

  const getSession = useCallback(
    (sessionId) => sessions.find(s => s.id === sessionId) || null,
    [sessions]
  );

  return (
    <SessionContext.Provider value={{
      sessions,
      activeSessionId,
      setActiveSessionId,
      serverOnline,
      checkServer,
      loadSessions,
      createSession,
      deleteSession,
      refreshSession,
      getSession,
    }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  return useContext(SessionContext);
}
