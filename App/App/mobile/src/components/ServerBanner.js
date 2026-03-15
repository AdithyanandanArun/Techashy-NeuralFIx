import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SPACING } from '../utils/theme';

export default function ServerBanner({ status, onRetry }) {
  // status: null=checking, true=online, false=offline
  if (status === true) return null; // hide when connected

  return (
    <View style={[styles.banner, status === false ? styles.offline : styles.checking]}>
      {status === null ? (
        <>
          <ActivityIndicator size="small" color={COLORS.warning} />
          <Text style={styles.text}>Connecting to server...</Text>
        </>
      ) : (
        <>
          <Ionicons name="cloud-offline-outline" size={16} color={COLORS.error} />
          <Text style={styles.text}>Cannot reach server — check WiFi & server IP</Text>
          <TouchableOpacity onPress={onRetry} style={styles.retryBtn}>
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: 'row', alignItems: 'center', gap: SPACING.sm,
    paddingHorizontal: SPACING.base, paddingVertical: SPACING.sm,
  },
  checking: { backgroundColor: '#F59E0B18' },
  offline:  { backgroundColor: '#EF444418' },
  text: { flex: 1, fontSize: FONTS.sizes.xs, color: COLORS.textSecondary },
  retryBtn: { paddingHorizontal: SPACING.sm, paddingVertical: 3, backgroundColor: COLORS.bg3, borderRadius: 4 },
  retryText: { fontSize: FONTS.sizes.xs, color: COLORS.textPrimary, fontWeight: FONTS.weights.semibold },
});
