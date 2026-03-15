import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS, FONTS, SPACING, RADIUS } from '../utils/theme';

const STATUS_CONFIG = {
  active:    { label: 'IN PROGRESS', color: COLORS.info,    bg: '#3B82F622' },
  resolved:  { label: 'RESOLVED',    color: COLORS.online,  bg: '#22C55E22' },
  escalated: { label: 'ESCALATED',   color: COLORS.error,   bg: '#EF444422' },
};

export default function StatusBadge({ status = 'active' }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.active;
  return (
    <View style={[styles.badge, { backgroundColor: cfg.bg, borderColor: cfg.color + '44' }]}>
      <View style={[styles.dot, { backgroundColor: cfg.color }]} />
      <Text style={[styles.label, { color: cfg.color }]}>{cfg.label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: SPACING.sm, paddingVertical: 3,
    borderRadius: RADIUS.full, borderWidth: 1, gap: 4, flexShrink: 0,
  },
  dot:   { width: 5, height: 5, borderRadius: 3 },
  label: { fontSize: 9, fontWeight: FONTS.weights.bold, letterSpacing: 0.8 },
});
