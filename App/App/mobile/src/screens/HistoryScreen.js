import React, { useEffect } from 'react';
import { View, Text, FlatList, TouchableOpacity, StyleSheet, Alert, RefreshControl } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { COLORS, FONTS, SPACING, RADIUS } from '../utils/theme';
import { useSession } from '../utils/SessionContext';
import StatusBadge from '../components/StatusBadge';

export default function HistoryScreen() {
  const { sessions, deleteSession, setActiveSessionId, loadSessions, serverOnline } = useSession();
  const navigation = useNavigation();
  const [refreshing, setRefreshing] = React.useState(false);

  useEffect(() => { loadSessions(); }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await loadSessions();
    setRefreshing(false);
  };

  const openSession = (session) => {
    setActiveSessionId(session.id);
    navigation.navigate('ChatTab', { screen: 'Chat' });
  };

  const openReport = (session) => {
    if (!session.diagnostic_report) {
      Alert.alert('No report yet', 'Open this session and tap "Report" to generate one.');
      return;
    }
    setActiveSessionId(session.id);
    navigation.navigate('ChatTab', { screen: 'DiagnosticReport', params: { sessionId: session.id } });
  };

  const confirmDelete = (id, title) => {
    Alert.alert('Delete Session', `Delete "${title}"?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => deleteSession(id) },
    ]);
  };

  const renderSession = ({ item }) => {
    const last = item.messages?.[item.messages.length - 1];
    return (
      <TouchableOpacity style={styles.card} onPress={() => openSession(item)} activeOpacity={0.7}>
        <View style={styles.cardIcon}>
          <Ionicons
            name={item.status === 'resolved' ? 'checkmark-circle' : item.status === 'escalated' ? 'warning' : 'radio'}
            size={18}
            color={item.status === 'resolved' ? COLORS.online : item.status === 'escalated' ? COLORS.error : COLORS.signal}
          />
        </View>
        <View style={styles.cardBody}>
          <View style={styles.cardRow}>
            <Text style={styles.cardTitle} numberOfLines={1}>{item.title}</Text>
            <Text style={styles.cardTime}>{formatTimeAgo(item.updated_at)}</Text>
          </View>
          {last && (
            <Text style={styles.cardPreview} numberOfLines={2}>
              {last.role === 'user' ? 'You: ' : 'AI: '}{last.content}
            </Text>
          )}
          <View style={styles.cardFooter}>
            <StatusBadge status={item.status} />
            <View style={styles.cardStats}>
              {item.image_paths?.length > 0 && (
                <View style={styles.stat}><Ionicons name="image-outline" size={11} color={COLORS.textMuted} /><Text style={styles.statText}>{item.image_paths.length}</Text></View>
              )}
              <View style={styles.stat}><Ionicons name="chatbubble-outline" size={11} color={COLORS.textMuted} /><Text style={styles.statText}>{item.messages?.length || 0}</Text></View>
              {item.diagnostic_report && (
                <TouchableOpacity style={styles.reportTag} onPress={() => openReport(item)}>
                  <Ionicons name="document-text" size={10} color={COLORS.info} />
                  <Text style={styles.reportTagText}>Report</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        </View>
        <TouchableOpacity style={styles.deleteBtn} onPress={() => confirmDelete(item.id, item.title)} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
          <Ionicons name="trash-outline" size={14} color={COLORS.textMuted} />
        </TouchableOpacity>
      </TouchableOpacity>
    );
  };

  const renderEmpty = () => (
    <View style={styles.empty}>
      <Ionicons name="time-outline" size={48} color={COLORS.textMuted} />
      <Text style={styles.emptyTitle}>No sessions yet</Text>
      <Text style={styles.emptyText}>Troubleshooting sessions will appear here.</Text>
    </View>
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Session History</Text>
          <Text style={styles.headerSub}>{sessions.length} session{sessions.length !== 1 ? 's' : ''}</Text>
        </View>
        {sessions.length > 0 && (
          <View style={styles.chip}>
            <Text style={styles.chipText}>{sessions.filter(s => s.status === 'resolved').length} resolved</Text>
          </View>
        )}
      </View>

      <FlatList
        data={sessions}
        renderItem={renderSession}
        keyExtractor={item => item.id}
        contentContainerStyle={[styles.list, sessions.length === 0 && { flex: 1 }]}
        ListEmptyComponent={renderEmpty}
        ItemSeparatorComponent={() => <View style={{ height: SPACING.sm }} />}
        showsVerticalScrollIndicator={false}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={COLORS.signal} />}
      />
    </SafeAreaView>
  );
}

function formatTimeAgo(ts) {
  if (!ts) return '';
  const diff = Math.floor((Date.now() - new Date(ts)) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg0 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: SPACING.base, paddingVertical: SPACING.md, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  headerTitle: { fontSize: FONTS.sizes.lg, fontWeight: FONTS.weights.bold, color: COLORS.textPrimary },
  headerSub:   { fontSize: FONTS.sizes.xs, color: COLORS.textMuted, marginTop: 2 },
  chip:        { paddingHorizontal: SPACING.sm, paddingVertical: 5, backgroundColor: COLORS.bg2, borderRadius: RADIUS.full, borderWidth: 1, borderColor: COLORS.border },
  chipText:    { fontSize: FONTS.sizes.xs, color: COLORS.textSecondary, fontWeight: FONTS.weights.medium },
  list: { padding: SPACING.base, paddingBottom: SPACING.xxl },
  card: { flexDirection: 'row', backgroundColor: COLORS.bg1, borderRadius: RADIUS.lg, borderWidth: 1, borderColor: COLORS.border, padding: SPACING.md, gap: SPACING.md, alignItems: 'flex-start' },
  cardIcon: { width: 38, height: 38, borderRadius: RADIUS.md, backgroundColor: COLORS.bg2, borderWidth: 1, borderColor: COLORS.border, alignItems: 'center', justifyContent: 'center' },
  cardBody: { flex: 1, gap: SPACING.xs },
  cardRow:  { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', gap: SPACING.sm },
  cardTitle:   { flex: 1, fontSize: FONTS.sizes.base, fontWeight: FONTS.weights.semibold, color: COLORS.textPrimary },
  cardTime:    { fontSize: FONTS.sizes.xs, color: COLORS.textMuted, flexShrink: 0 },
  cardPreview: { fontSize: FONTS.sizes.xs, color: COLORS.textSecondary, lineHeight: 17 },
  cardFooter:  { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 2 },
  cardStats:   { flexDirection: 'row', gap: SPACING.sm, alignItems: 'center' },
  stat:        { flexDirection: 'row', gap: 3, alignItems: 'center' },
  statText:    { fontSize: 10, color: COLORS.textMuted },
  reportTag:   { flexDirection: 'row', alignItems: 'center', gap: 3, backgroundColor: '#3B82F622', paddingHorizontal: 5, paddingVertical: 2, borderRadius: RADIUS.sm },
  reportTagText: { fontSize: 9, color: COLORS.info, fontWeight: FONTS.weights.semibold },
  deleteBtn: { padding: 4, marginTop: 2 },
  empty:      { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: SPACING.xl, gap: SPACING.md },
  emptyTitle: { fontSize: FONTS.sizes.xl, fontWeight: FONTS.weights.bold, color: COLORS.textPrimary },
  emptyText:  { fontSize: FONTS.sizes.base, color: COLORS.textSecondary, textAlign: 'center', lineHeight: 22 },
});
