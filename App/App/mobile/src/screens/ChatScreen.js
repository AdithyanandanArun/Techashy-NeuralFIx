import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, FlatList,
  StyleSheet, KeyboardAvoidingView, Platform, ActivityIndicator,
  Image, Alert, Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { COLORS, FONTS, SPACING, RADIUS } from '../utils/theme';
import { useSession } from '../utils/SessionContext';
import * as api from '../services/api';
import MessageBubble from '../components/MessageBubble';
import StatusBadge from '../components/StatusBadge';
import ServerBanner from '../components/ServerBanner';

const QUICK_PROMPTS = [
  'No internet connection',
  'WiFi connected but slow',
  'Router lights are red',
  "Can't connect to WiFi",
  'Network keeps dropping',
];

export default function ChatScreen({ navigation }) {
  const { createSession, refreshSession, getSession, activeSessionId,
          setActiveSessionId, serverOnline, checkServer } = useSession();

  const [sessionId, setSessionId] = useState(activeSessionId);
  const [messages, setMessages] = useState([]);
  const [sessionMeta, setSessionMeta] = useState(null);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pendingImage, setPendingImage] = useState(null); // local uri for preview
  const flatListRef = useRef(null);
  const inputRef = useRef(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Pulse animation
  useEffect(() => {
    if (isLoading) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 0.3, duration: 500, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1,   duration: 500, useNativeDriver: true }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isLoading]);

  // Init: check server, create session
  useEffect(() => {
    (async () => {
      const online = await checkServer();
      if (online) {
        let sid = activeSessionId;
        if (!sid) {
          sid = await createSession();
          setSessionId(sid);
        } else {
          setSessionId(sid);
          const s = await refreshSession(sid);
          if (s) {
            setMessages(s.messages || []);
            setSessionMeta(s);
          }
        }
      }
    })();
  }, []);

  // Keep messages in sync when sessionId changes
  useEffect(() => {
    if (!sessionId) return;
    const s = getSession(sessionId);
    if (s) { setMessages(s.messages || []); setSessionMeta(s); }
  }, [sessionId, getSession]);

  const startNewSession = useCallback(async () => {
    const sid = await createSession();
    setSessionId(sid);
    setActiveSessionId(sid);
    setMessages([]);
    setSessionMeta(null);
    setPendingImage(null);
  }, [createSession, setActiveSessionId]);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') { Alert.alert('Permission needed', 'Allow photo library access.'); return; }
    const r = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, allowsEditing: true, quality: 0.7 });
    if (!r.canceled) setPendingImage(r.assets[0].uri);
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') { Alert.alert('Permission needed', 'Allow camera access.'); return; }
    const r = await ImagePicker.launchCameraAsync({ allowsEditing: true, quality: 0.7 });
    if (!r.canceled) setPendingImage(r.assets[0].uri);
  };

  const showImageOptions = () => {
    Alert.alert('Add Equipment Photo', 'Choose source', [
      { text: 'Take Photo',           onPress: takePhoto },
      { text: 'Choose from Library',  onPress: pickImage },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  const handleSend = async () => {
    const text = inputText.trim();
    if (!text && !pendingImage) return;
    if (isLoading) return;

    const userText = text || 'I have attached an image of the networking equipment.';
    setInputText('');
    setIsLoading(true);

    // Optimistic user message
    const optimisticMsg = {
      id: `tmp-${Date.now()}`,
      role: 'user',
      content: userText,
      localImageUri: pendingImage,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, optimisticMsg]);
    const imageToUpload = pendingImage;
    setPendingImage(null);
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 80);

    try {
      let imagePath = null;

      // 1. Upload image if present
      if (imageToUpload && sessionId) {
        const filename = `equipment_${Date.now()}.jpg`;
        const uploadResult = await api.uploadImage(sessionId, imageToUpload, filename);
        imagePath = uploadResult.image_path;
        // Attach vision analysis note to message
        if (uploadResult.vision_analysis) {
          optimisticMsg.content += `\n\n[Image uploaded — equipment analysis running]`;
        }
      }

      // 2. Send chat message
      const chatResult = await api.sendChatMessage(sessionId, userText);

      // 3. Refresh session from server (has latest messages)
      const updated = await refreshSession(sessionId);
      if (updated) {
        setMessages(updated.messages || []);
        setSessionMeta(updated);
      }

    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: `⚠️ Error: ${err.message}\n\nMake sure the server is running and you're on the same WiFi network.`,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
    }
  };

  const handleGenerateReport = () => {
    if (!sessionId || messages.length < 2) {
      Alert.alert('Not enough data', 'Complete some troubleshooting before generating a report.');
      return;
    }
    navigation.navigate('DiagnosticReport', { sessionId });
  };

  const renderEmpty = () => (
    <View style={styles.empty}>
      <View style={styles.emptyIcon}><Ionicons name="wifi" size={40} color={COLORS.signal} /></View>
      <Text style={styles.emptyTitle}>Network Issue?</Text>
      <Text style={styles.emptySubtitle}>Describe your problem and I'll guide you step by step.</Text>
      <Text style={styles.quickLabel}>COMMON ISSUES</Text>
      <View style={styles.quickList}>
        {QUICK_PROMPTS.map(p => (
          <TouchableOpacity key={p} style={styles.quickChip} onPress={() => setInputText(p)}>
            <Text style={styles.quickChipText}>{p}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.logo}><Ionicons name="radio" size={16} color={COLORS.signal} /></View>
          <View>
            <Text style={styles.headerTitle}>NetFixAI</Text>
            <Text style={styles.headerSub}>Network Troubleshooter</Text>
          </View>
        </View>
        <View style={styles.headerRight}>
          {messages.length >= 2 && (
            <TouchableOpacity style={styles.reportBtn} onPress={handleGenerateReport}>
              <Ionicons name="document-text-outline" size={15} color={COLORS.signal} />
              <Text style={styles.reportBtnText}>Report</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity style={styles.newBtn} onPress={startNewSession}>
            <Ionicons name="add" size={20} color={COLORS.textSecondary} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Server status */}
      <ServerBanner status={serverOnline} onRetry={checkServer} />

      {/* Session strip */}
      {sessionMeta && messages.length > 0 && (
        <View style={styles.sessionStrip}>
          <StatusBadge status={sessionMeta.status} />
          <Text style={styles.sessionTitle} numberOfLines={1}>{sessionMeta.title}</Text>
          {(sessionMeta.image_paths?.length > 0) && (
            <View style={styles.imgCount}>
              <Ionicons name="image-outline" size={12} color={COLORS.textMuted} />
              <Text style={styles.imgCountText}>{sessionMeta.image_paths.length}</Text>
            </View>
          )}
        </View>
      )}

      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={({ item }) => <MessageBubble message={item} />}
          keyExtractor={(item, i) => item.id || String(i)}
          contentContainerStyle={[styles.msgList, messages.length === 0 && { flex: 1 }]}
          ListEmptyComponent={renderEmpty}
          showsVerticalScrollIndicator={false}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: false })}
        />

        {isLoading && (
          <View style={styles.typing}>
            {[0, 1, 2].map(i => (
              <Animated.View key={i} style={[styles.typingDot, { opacity: pulseAnim }]} />
            ))}
            <Text style={styles.typingText}>Analyzing...</Text>
          </View>
        )}

        {pendingImage && (
          <View style={styles.imgPreview}>
            <Image source={{ uri: pendingImage }} style={styles.previewImg} />
            <TouchableOpacity style={styles.removeImg} onPress={() => setPendingImage(null)}>
              <Ionicons name="close-circle" size={20} color={COLORS.error} />
            </TouchableOpacity>
            <Text style={styles.imgPreviewLabel}>Equipment photo ready to send</Text>
          </View>
        )}

        <View style={styles.inputBar}>
          <TouchableOpacity style={styles.mediaBtn} onPress={showImageOptions}>
            <Ionicons name="camera-outline" size={22} color={COLORS.textSecondary} />
          </TouchableOpacity>
          <TextInput
            ref={inputRef}
            style={styles.input}
            placeholder="Describe the problem..."
            placeholderTextColor={COLORS.textMuted}
            value={inputText}
            onChangeText={setInputText}
            multiline maxLength={1000}
          />
          <TouchableOpacity
            style={[styles.sendBtn, (!inputText.trim() && !pendingImage) && styles.sendBtnOff]}
            onPress={handleSend}
            disabled={!inputText.trim() && !pendingImage}
          >
            <Ionicons name="arrow-up" size={18} color={COLORS.textInverse} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg0 },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: SPACING.base, paddingVertical: SPACING.md,
    borderBottomWidth: 1, borderBottomColor: COLORS.border,
  },
  headerLeft:  { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm },
  logo: {
    width: 34, height: 34, borderRadius: RADIUS.sm,
    backgroundColor: COLORS.signalDim, borderWidth: 1, borderColor: COLORS.signalMid,
    alignItems: 'center', justifyContent: 'center',
  },
  headerTitle: { fontSize: FONTS.sizes.md, fontWeight: FONTS.weights.bold, color: COLORS.textPrimary, letterSpacing: 0.5 },
  headerSub:   { fontSize: FONTS.sizes.xs, color: COLORS.textMuted, textTransform: 'uppercase', letterSpacing: 0.3 },
  reportBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: SPACING.sm, paddingVertical: 6,
    borderRadius: RADIUS.sm, borderWidth: 1, borderColor: COLORS.signalMid, backgroundColor: COLORS.signalDim,
  },
  reportBtnText: { fontSize: FONTS.sizes.xs, color: COLORS.signal, fontWeight: FONTS.weights.semibold },
  newBtn: {
    width: 34, height: 34, borderRadius: RADIUS.sm,
    backgroundColor: COLORS.bg2, borderWidth: 1, borderColor: COLORS.border,
    alignItems: 'center', justifyContent: 'center',
  },
  sessionStrip: {
    flexDirection: 'row', alignItems: 'center', gap: SPACING.sm,
    paddingHorizontal: SPACING.base, paddingVertical: SPACING.xs,
    backgroundColor: COLORS.bg1, borderBottomWidth: 1, borderBottomColor: COLORS.border,
  },
  sessionTitle: { flex: 1, fontSize: FONTS.sizes.xs, color: COLORS.textSecondary },
  imgCount:     { flexDirection: 'row', alignItems: 'center', gap: 3 },
  imgCountText: { fontSize: FONTS.sizes.xs, color: COLORS.textMuted },
  msgList: { paddingHorizontal: SPACING.base, paddingVertical: SPACING.base, gap: SPACING.md },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: SPACING.xl },
  emptyIcon: {
    width: 72, height: 72, borderRadius: RADIUS.xl,
    backgroundColor: COLORS.signalDim, borderWidth: 1, borderColor: COLORS.signalMid,
    alignItems: 'center', justifyContent: 'center', marginBottom: SPACING.base,
  },
  emptyTitle:    { fontSize: FONTS.sizes.xl, fontWeight: FONTS.weights.bold, color: COLORS.textPrimary, marginBottom: SPACING.sm },
  emptySubtitle: { fontSize: FONTS.sizes.base, color: COLORS.textSecondary, textAlign: 'center', lineHeight: 22, marginBottom: SPACING.lg },
  quickLabel: { fontSize: FONTS.sizes.xs, color: COLORS.textMuted, letterSpacing: 1, fontWeight: FONTS.weights.semibold, marginBottom: SPACING.sm },
  quickList:  { width: '100%', gap: SPACING.sm },
  quickChip:  { paddingHorizontal: SPACING.base, paddingVertical: SPACING.sm, backgroundColor: COLORS.bg2, borderRadius: RADIUS.md, borderWidth: 1, borderColor: COLORS.border },
  quickChipText: { fontSize: FONTS.sizes.sm, color: COLORS.textSecondary, fontWeight: FONTS.weights.medium },
  typing: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: SPACING.base, paddingVertical: SPACING.sm, gap: 4 },
  typingDot:  { width: 6, height: 6, borderRadius: 3, backgroundColor: COLORS.signal },
  typingText: { fontSize: FONTS.sizes.xs, color: COLORS.textMuted, marginLeft: 4, textTransform: 'uppercase', letterSpacing: 0.5 },
  imgPreview: {
    flexDirection: 'row', alignItems: 'center',
    marginHorizontal: SPACING.base, marginBottom: SPACING.sm,
    padding: SPACING.sm, backgroundColor: COLORS.bg2, borderRadius: RADIUS.md,
    borderWidth: 1, borderColor: COLORS.border, gap: SPACING.sm,
  },
  previewImg:      { width: 48, height: 48, borderRadius: RADIUS.sm, backgroundColor: COLORS.bg3 },
  removeImg:       { position: 'absolute', top: -6, left: 40 },
  imgPreviewLabel: { flex: 1, fontSize: FONTS.sizes.sm, color: COLORS.textSecondary },
  inputBar: {
    flexDirection: 'row', alignItems: 'flex-end', gap: SPACING.sm,
    paddingHorizontal: SPACING.base, paddingVertical: SPACING.sm, paddingBottom: SPACING.md,
    borderTopWidth: 1, borderTopColor: COLORS.border, backgroundColor: COLORS.bg0,
  },
  mediaBtn: {
    width: 40, height: 40, borderRadius: RADIUS.md,
    backgroundColor: COLORS.bg2, borderWidth: 1, borderColor: COLORS.border,
    alignItems: 'center', justifyContent: 'center',
  },
  input: {
    flex: 1, minHeight: 40, maxHeight: 120,
    backgroundColor: COLORS.bg2, borderWidth: 1, borderColor: COLORS.borderLight,
    borderRadius: RADIUS.md, paddingHorizontal: SPACING.md, paddingVertical: 10,
    fontSize: FONTS.sizes.base, color: COLORS.textPrimary,
  },
  sendBtn:    { width: 40, height: 40, borderRadius: RADIUS.md, backgroundColor: COLORS.signal, alignItems: 'center', justifyContent: 'center' },
  sendBtnOff: { backgroundColor: COLORS.bg3 },
});
