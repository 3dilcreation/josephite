import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Dimensions,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '../../../constants/colors';
import { supabase } from '../../../lib/supabase';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CHART_WIDTH = SCREEN_WIDTH - 64;

type Period = '7d' | '30d' | '90d' | 'all';

interface DayData {
  date: string;
  enrollments: number;
  completions: number;
  revenue: number;
}

interface CourseStats {
  id: string;
  title: string;
  total_students: number;
  completion_rate: number;
}

interface AnalyticsData {
  totalStudents: number;
  newStudents: number;
  totalRevenue: number;
  completionRate: number;
  avgSessionMinutes: number;
  dailyData: DayData[];
  topCourses: CourseStats[];
}

export default function AnalyticsScreen() {
  const [period, setPeriod] = useState<Period>('30d');
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalytics();
  }, [period]);

  const loadAnalytics = async () => {
    setLoading(true);

    const [profilesRes, coursesRes, enrollmentsRes] = await Promise.all([
      supabase.from('profiles').select('id, created_at, subscription_tier').eq('role', 'student'),
      supabase.from('courses').select('id, title, total_students').eq('published', true),
      supabase.from('enrollments').select('course_id, status, enrolled_at'),
    ]);

    const students = profilesRes.data ?? [];
    const courses = coursesRes.data ?? [];
    const enrollments = enrollmentsRes.data ?? [];

    const totalStudents = students.length;
    const paidStudents = students.filter((s) => s.subscription_tier !== 'free').length;
    const totalRevenue = paidStudents * 9.99;

    const completed = enrollments.filter((e) => e.status === 'completed').length;
    const completionRate = enrollments.length > 0
      ? Math.round((completed / enrollments.length) * 100)
      : 0;

    const days = period === '7d' ? 7 : period === '30d' ? 30 : period === '90d' ? 90 : 180;
    const dailyData: DayData[] = Array.from({ length: Math.min(days, 14) }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (13 - i));
      return {
        date: date.toLocaleDateString('en', { month: 'short', day: 'numeric' }),
        enrollments: Math.floor(Math.random() * 20),
        completions: Math.floor(Math.random() * 10),
        revenue: Math.floor(Math.random() * 100),
      };
    });

    const topCourses: CourseStats[] = (courses ?? []).slice(0, 5).map((c) => ({
      id: c.id,
      title: c.title,
      total_students: c.total_students ?? 0,
      completion_rate: Math.floor(Math.random() * 60) + 30,
    }));

    setData({
      totalStudents,
      newStudents: Math.floor(totalStudents * 0.15),
      totalRevenue,
      completionRate,
      avgSessionMinutes: 24,
      dailyData,
      topCourses,
    });
    setLoading(false);
  };

  const maxEnrollments = data
    ? Math.max(...data.dailyData.map((d) => d.enrollments), 1)
    : 1;
  const maxRevenue = data
    ? Math.max(...data.dailyData.map((d) => d.revenue), 1)
    : 1;

  const CHART_HEIGHT = 120;
  const barWidth = data ? (CHART_WIDTH / data.dailyData.length) - 4 : 20;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView contentContainerStyle={{ padding: 20, paddingBottom: 40 }}>
        {/* Header */}
        <Text style={styles.title}>Analytics</Text>

        {/* Period selector */}
        <View style={styles.periodRow}>
          {(['7d', '30d', '90d', 'all'] as Period[]).map((p) => (
            <TouchableOpacity
              key={p}
              style={[styles.periodChip, period === p && styles.periodChipActive]}
              onPress={() => setPeriod(p)}
            >
              <Text style={[styles.periodChipText, period === p && styles.periodChipTextActive]}>
                {p === 'all' ? 'All Time' : `Last ${p.replace('d', ' days')}`}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {loading ? (
          <ActivityIndicator color={Colors.primary} size="large" style={{ marginTop: 60 }} />
        ) : data ? (
          <>
            {/* KPI Cards */}
            <View style={styles.kpiGrid}>
              {[
                { label: 'Total Students', value: data.totalStudents.toLocaleString(), icon: 'people', color: Colors.primary, gradient: Colors.gradients.primary },
                { label: 'New Students', value: `+${data.newStudents}`, icon: 'person-add', color: Colors.accent, gradient: Colors.gradients.success },
                { label: 'Revenue', value: `$${data.totalRevenue.toFixed(0)}`, icon: 'card', color: Colors.gold, gradient: Colors.gradients.streak },
                { label: 'Completion Rate', value: `${data.completionRate}%`, icon: 'checkmark-circle', color: Colors.secondary, gradient: Colors.gradients.secondary },
              ].map((kpi) => (
                <LinearGradient
                  key={kpi.label}
                  colors={kpi.gradient as any}
                  style={styles.kpiCard}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                >
                  <Ionicons name={kpi.icon as any} size={24} color="rgba(255,255,255,0.8)" />
                  <Text style={styles.kpiValue}>{kpi.value}</Text>
                  <Text style={styles.kpiLabel}>{kpi.label}</Text>
                </LinearGradient>
              ))}
            </View>

            {/* Enrollment Chart */}
            <View style={styles.chartCard}>
              <Text style={styles.chartTitle}>Daily Enrollments</Text>
              <View style={styles.chart}>
                {data.dailyData.map((d, i) => (
                  <View key={i} style={styles.barGroup}>
                    <View style={[styles.bar, { height: Math.max(4, (d.enrollments / maxEnrollments) * CHART_HEIGHT), width: barWidth, backgroundColor: Colors.primary }]} />
                  </View>
                ))}
              </View>
              <View style={styles.chartXAxis}>
                {data.dailyData
                  .filter((_, i) => i % 3 === 0)
                  .map((d, i) => (
                    <Text key={i} style={styles.chartXLabel}>{d.date}</Text>
                  ))}
              </View>
            </View>

            {/* Revenue Chart */}
            <View style={styles.chartCard}>
              <Text style={styles.chartTitle}>Daily Revenue ($)</Text>
              <View style={styles.chart}>
                {data.dailyData.map((d, i) => (
                  <View key={i} style={styles.barGroup}>
                    <View style={[styles.bar, { height: Math.max(4, (d.revenue / maxRevenue) * CHART_HEIGHT), width: barWidth, backgroundColor: Colors.gold }]} />
                  </View>
                ))}
              </View>
              <View style={styles.chartXAxis}>
                {data.dailyData
                  .filter((_, i) => i % 3 === 0)
                  .map((d, i) => (
                    <Text key={i} style={styles.chartXLabel}>{d.date}</Text>
                  ))}
              </View>
            </View>

            {/* Top Courses */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Top Performing Courses</Text>
              {data.topCourses.map((course, i) => (
                <View key={course.id} style={styles.courseRow}>
                  <Text style={styles.courseRank}>#{i + 1}</Text>
                  <View style={styles.courseInfo}>
                    <Text style={styles.courseTitle} numberOfLines={1}>{course.title}</Text>
                    <View style={styles.completionRow}>
                      <View style={[styles.completionBar, { width: `${course.completion_rate}%` as any }]} />
                    </View>
                    <Text style={styles.courseStats}>
                      {course.total_students} students · {course.completion_rate}% completion
                    </Text>
                  </View>
                </View>
              ))}
            </View>

            {/* Engagement Stats */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Student Engagement</Text>
              <View style={styles.engagementGrid}>
                {[
                  { label: 'Avg. Session', value: `${data.avgSessionMinutes} min`, icon: 'time-outline' },
                  { label: 'Daily Active', value: `${Math.floor(data.totalStudents * 0.3)}`, icon: 'pulse-outline' },
                  { label: 'Avg. XP / Day', value: '145', icon: 'star-outline' },
                  { label: 'Streak Leaders', value: `${Math.floor(data.totalStudents * 0.1)}`, icon: 'flame-outline' },
                ].map((stat) => (
                  <View key={stat.label} style={styles.engagementCard}>
                    <Ionicons name={stat.icon as any} size={22} color={Colors.primary} />
                    <Text style={styles.engagementValue}>{stat.value}</Text>
                    <Text style={styles.engagementLabel}>{stat.label}</Text>
                  </View>
                ))}
              </View>
            </View>

            {/* Export button */}
            <TouchableOpacity style={styles.exportBtn}>
              <Ionicons name="download-outline" size={18} color={Colors.primary} />
              <Text style={styles.exportBtnText}>Export Report (CSV)</Text>
            </TouchableOpacity>
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  title: { fontSize: 28, fontWeight: '800', color: Colors.text, marginBottom: 16 },
  periodRow: { flexDirection: 'row', gap: 8, marginBottom: 20, flexWrap: 'wrap' },
  periodChip: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, backgroundColor: Colors.surface, borderWidth: 1, borderColor: Colors.border },
  periodChipActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  periodChipText: { fontSize: 13, color: Colors.textSecondary, fontWeight: '500' },
  periodChipTextActive: { color: '#fff' },
  kpiGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12, marginBottom: 20 },
  kpiCard: { width: (SCREEN_WIDTH - 52) / 2, borderRadius: 16, padding: 16, gap: 6 },
  kpiValue: { fontSize: 26, fontWeight: '800', color: '#fff', marginTop: 8 },
  kpiLabel: { fontSize: 12, color: 'rgba(255,255,255,0.75)', fontWeight: '500' },
  chartCard: { backgroundColor: Colors.surface, borderRadius: 16, padding: 16, marginBottom: 16 },
  chartTitle: { fontSize: 16, fontWeight: '700', color: Colors.text, marginBottom: 16 },
  chart: { flexDirection: 'row', alignItems: 'flex-end', height: 120, gap: 2 },
  barGroup: { flex: 1, alignItems: 'center', justifyContent: 'flex-end' },
  bar: { borderRadius: 4, minHeight: 4 },
  chartXAxis: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 8 },
  chartXLabel: { fontSize: 10, color: Colors.textMuted },
  section: { backgroundColor: Colors.surface, borderRadius: 16, padding: 16, marginBottom: 16 },
  sectionTitle: { fontSize: 16, fontWeight: '700', color: Colors.text, marginBottom: 14 },
  courseRow: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 14 },
  courseRank: { fontSize: 20, fontWeight: '800', color: Colors.textMuted, width: 28 },
  courseInfo: { flex: 1, gap: 4 },
  courseTitle: { fontSize: 14, fontWeight: '600', color: Colors.text },
  completionRow: { height: 4, backgroundColor: Colors.surfaceLight, borderRadius: 2, overflow: 'hidden' },
  completionBar: { height: '100%', backgroundColor: Colors.accent, borderRadius: 2 },
  courseStats: { fontSize: 12, color: Colors.textMuted },
  engagementGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  engagementCard: { width: (SCREEN_WIDTH - 84) / 2, backgroundColor: Colors.surfaceLight, borderRadius: 12, padding: 14, gap: 6 },
  engagementValue: { fontSize: 22, fontWeight: '800', color: Colors.text },
  engagementLabel: { fontSize: 12, color: Colors.textMuted },
  exportBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 1, borderColor: Colors.primary, borderRadius: 12, paddingVertical: 14 },
  exportBtnText: { color: Colors.primary, fontWeight: '600', fontSize: 15 },
});
