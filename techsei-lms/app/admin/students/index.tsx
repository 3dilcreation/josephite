// ============================================================
// TechSei LMS — Admin Student Management Screen
// ============================================================
import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  TextInput,
  StyleSheet,
  Modal,
  ScrollView,
  Alert,
  Dimensions,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─── Design tokens ───────────────────────────────────────────
const COLORS = {
  background: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  text: '#FFFFFF',
  textMuted: '#8A8AAA',
  border: '#2A2A4A',
  warning: '#FFB84C',
  error: '#FF6B6B',
  blue: '#4FC3F7',
  orange: '#FF9800',
  gold: '#FFD700',
};

// ─── Types ────────────────────────────────────────────────────
export interface Student {
  id: string;
  name: string;
  email: string;
  level: number;
  subscriptionTier: 'free' | 'pro' | 'enterprise';
  lastActive: string;
  xp: number;
  isActive: boolean;
  joinDate: string;
  coursesEnrolled: number;
  coursesCompleted: number;
}

type FilterType = 'all' | 'active' | 'inactive' | 'pro' | 'free';
type SortType = 'name' | 'joinDate' | 'xp' | 'lastActive';

interface AddStudentForm {
  name: string;
  email: string;
  password: string;
  role: 'student' | 'instructor';
  tier: 'free' | 'pro' | 'enterprise';
}

interface CourseEnrollment {
  courseId: string;
  courseName: string;
  progress: number;
  completed: boolean;
}

// ─── Mock data ────────────────────────────────────────────────
const MOCK_STUDENTS: Student[] = [
  { id: '1', name: 'Marcus Chen', email: 'marcus.chen@email.com', level: 24, subscriptionTier: 'pro', lastActive: new Date(Date.now() - 1000 * 60 * 5).toISOString(), xp: 12400, isActive: true, joinDate: '2024-01-15', coursesEnrolled: 8, coursesCompleted: 5 },
  { id: '2', name: 'Priya Sharma', email: 'priya.sharma@email.com', level: 18, subscriptionTier: 'free', lastActive: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), xp: 7800, isActive: false, joinDate: '2024-02-20', coursesEnrolled: 4, coursesCompleted: 2 },
  { id: '3', name: 'Alex Thompson', email: 'alex.t@email.com', level: 31, subscriptionTier: 'enterprise', lastActive: new Date(Date.now() - 1000 * 60 * 10).toISOString(), xp: 22100, isActive: true, joinDate: '2023-11-03', coursesEnrolled: 15, coursesCompleted: 13 },
  { id: '4', name: 'Jordan Lee', email: 'jordan.lee@email.com', level: 9, subscriptionTier: 'free', lastActive: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(), xp: 3200, isActive: false, joinDate: '2024-05-10', coursesEnrolled: 2, coursesCompleted: 0 },
  { id: '5', name: 'Sarah Kim', email: 'sarah.kim@email.com', level: 22, subscriptionTier: 'pro', lastActive: new Date(Date.now() - 1000 * 60 * 30).toISOString(), xp: 11600, isActive: true, joinDate: '2024-03-08', coursesEnrolled: 7, coursesCompleted: 4 },
  { id: '6', name: 'David Park', email: 'david.park@email.com', level: 15, subscriptionTier: 'pro', lastActive: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(), xp: 6900, isActive: false, joinDate: '2024-04-12', coursesEnrolled: 5, coursesCompleted: 3 },
  { id: '7', name: 'Elena Rodriguez', email: 'elena.r@email.com', level: 6, subscriptionTier: 'free', lastActive: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3).toISOString(), xp: 1800, isActive: false, joinDate: '2024-06-01', coursesEnrolled: 1, coursesCompleted: 0 },
  { id: '8', name: 'Mohammed Al-Hassan', email: 'mohammed.h@email.com', level: 28, subscriptionTier: 'pro', lastActive: new Date(Date.now() - 1000 * 60 * 15).toISOString(), xp: 18500, isActive: true, joinDate: '2023-12-20', coursesEnrolled: 11, coursesCompleted: 9 },
  { id: '9', name: 'Yuki Tanaka', email: 'yuki.tanaka@email.com', level: 19, subscriptionTier: 'enterprise', lastActive: new Date(Date.now() - 1000 * 60 * 60).toISOString(), xp: 9300, isActive: true, joinDate: '2024-01-30', coursesEnrolled: 9, coursesCompleted: 6 },
  { id: '10', name: 'Lena Müller', email: 'lena.mueller@email.com', level: 12, subscriptionTier: 'free', lastActive: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), xp: 4400, isActive: false, joinDate: '2024-04-25', coursesEnrolled: 3, coursesCompleted: 1 },
  { id: '11', name: 'James Okonkwo', email: 'james.o@email.com', level: 35, subscriptionTier: 'enterprise', lastActive: new Date(Date.now() - 1000 * 60 * 3).toISOString(), xp: 28900, isActive: true, joinDate: '2023-09-14', coursesEnrolled: 18, coursesCompleted: 16 },
  { id: '12', name: 'Fatima Al-Zahra', email: 'fatima.z@email.com', level: 20, subscriptionTier: 'pro', lastActive: new Date(Date.now() - 1000 * 60 * 45).toISOString(), xp: 10200, isActive: true, joinDate: '2024-02-14', coursesEnrolled: 6, coursesCompleted: 4 },
];

const MOCK_ENROLLMENTS: Record<string, CourseEnrollment[]> = {
  '1': [
    { courseId: 'c1', courseName: 'React Native Mastery', progress: 100, completed: true },
    { courseId: 'c2', courseName: 'TypeScript Deep Dive', progress: 75, completed: false },
    { courseId: 'c3', courseName: 'Python for AI', progress: 45, completed: false },
  ],
  '3': [
    { courseId: 'c1', courseName: 'React Native Mastery', progress: 100, completed: true },
    { courseId: 'c4', courseName: 'Cloud Architecture', progress: 100, completed: true },
  ],
  '8': [
    { courseId: 'c5', courseName: 'Full-Stack Web Dev', progress: 100, completed: true },
    { courseId: 'c2', courseName: 'TypeScript Deep Dive', progress: 88, completed: false },
  ],
};

// ─── Helpers ──────────────────────────────────────────────────
function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days}d ago`;
  if (days < 30) return `${Math.floor(days / 7)}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

function formatXP(xp: number): string {
  if (xp >= 1000) return `${(xp / 1000).toFixed(1)}k`;
  return String(xp);
}

// ─── Avatar ───────────────────────────────────────────────────
function AvatarCircle({ name, size = 44 }: { name: string; size?: number }) {
  const initials = name.split(' ').map((n) => n[0]).slice(0, 2).join('').toUpperCase();
  const hue = name.charCodeAt(0) * 15 % 360;
  return (
    <View style={[styles.avatarCircle, { width: size, height: size, borderRadius: size / 2, backgroundColor: `hsl(${hue}, 55%, 32%)` }]}>
      <Text style={[styles.avatarText, { fontSize: size * 0.36 }]}>{initials}</Text>
    </View>
  );
}

// ─── Level / Tier badges ──────────────────────────────────────
function LevelBadge({ level }: { level: number }) {
  return (
    <View style={styles.levelBadge}>
      <Text style={styles.levelBadgeText}>Lv.{level}</Text>
    </View>
  );
}

function TierBadge({ tier }: { tier: Student['subscriptionTier'] }) {
  const cfg = {
    pro: { colors: [COLORS.gold, '#FFA500'] as [string, string], label: 'PRO', textColor: '#000' },
    enterprise: { colors: [COLORS.primary, '#A855F7'] as [string, string], label: 'ENT', textColor: '#fff' },
    free: { colors: [COLORS.surfaceLight, COLORS.border] as [string, string], label: 'FREE', textColor: COLORS.textMuted },
  };
  const c = cfg[tier];
  return (
    <LinearGradient colors={c.colors} style={styles.tierBadge} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
      <Text style={[styles.tierBadgeText, { color: c.textColor }]}>{c.label}</Text>
    </LinearGradient>
  );
}

// ─── Student Row ──────────────────────────────────────────────
function StudentListRow({
  student,
  onPress,
  onLongPress,
  selected,
  selectionMode,
}: {
  student: Student;
  onPress: () => void;
  onLongPress: () => void;
  selected: boolean;
  selectionMode: boolean;
}) {
  return (
    <TouchableOpacity
      style={[styles.studentRow, selected && styles.studentRowSelected]}
      onPress={onPress}
      onLongPress={onLongPress}
      activeOpacity={0.8}
    >
      {selectionMode && (
        <View style={[styles.checkbox, selected && styles.checkboxSelected]}>
          {selected && <Ionicons name="checkmark" size={14} color="#fff" />}
        </View>
      )}
      <View style={styles.avatarWrapper}>
        <AvatarCircle name={student.name} />
        <View style={[styles.onlineDot, { backgroundColor: student.isActive ? COLORS.accent : COLORS.textMuted }]} />
      </View>
      <View style={styles.studentInfo}>
        <View style={styles.nameRow}>
          <Text style={styles.studentName} numberOfLines={1}>{student.name}</Text>
          <LevelBadge level={student.level} />
        </View>
        <Text style={styles.studentEmail} numberOfLines={1}>{student.email}</Text>
        <View style={styles.metaRow}>
          <Ionicons name="time-outline" size={11} color={COLORS.textMuted} />
          <Text style={styles.metaText}>{formatDate(student.lastActive)}</Text>
          <View style={styles.metaDot} />
          <Ionicons name="flash-outline" size={11} color={COLORS.warning} />
          <Text style={[styles.metaText, { color: COLORS.warning }]}>{formatXP(student.xp)} XP</Text>
        </View>
      </View>
      <View style={styles.studentRight}>
        <TierBadge tier={student.subscriptionTier} />
        <Ionicons name="chevron-forward" size={15} color={COLORS.textMuted} style={{ marginTop: 8 }} />
      </View>
    </TouchableOpacity>
  );
}

// ─── Add Student Modal ────────────────────────────────────────
function AddStudentModal({
  visible,
  onClose,
  onAdd,
}: {
  visible: boolean;
  onClose: () => void;
  onAdd: (form: AddStudentForm) => void;
}) {
  const [form, setForm] = useState<AddStudentForm>({ name: '', email: '', password: '', role: 'student', tier: 'free' });
  const [errors, setErrors] = useState<Partial<Record<keyof AddStudentForm, string>>>({});

  const validate = () => {
    const e: typeof errors = {};
    if (!form.name.trim()) e.name = 'Name is required';
    if (!form.email.trim() || !/\S+@\S+\.\S+/.test(form.email)) e.email = 'Valid email required';
    if (form.password.length < 6) e.password = 'Minimum 6 characters';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = () => {
    if (validate()) {
      onAdd(form);
      setForm({ name: '', email: '', password: '', role: 'student', tier: 'free' });
      setErrors({});
    }
  };

  const Field = ({ label, field, placeholder, secure = false, keyboard = 'default' }: {
    label: string; field: keyof AddStudentForm; placeholder: string; secure?: boolean; keyboard?: any;
  }) => (
    <View style={styles.formGroup}>
      <Text style={styles.formLabel}>{label}</Text>
      <TextInput
        style={[styles.formInput, errors[field] && styles.formInputError]}
        value={form[field] as string}
        onChangeText={(t) => setForm({ ...form, [field]: t })}
        placeholder={placeholder}
        placeholderTextColor={COLORS.textMuted}
        secureTextEntry={secure}
        keyboardType={keyboard}
        autoCapitalize={field === 'email' ? 'none' : 'words'}
      />
      {errors[field] && <Text style={styles.formError}>{errors[field]}</Text>}
    </View>
  );

  const SegmentControl = <T extends string>({
    label, options, value, onChange,
  }: { label: string; options: T[]; value: T; onChange: (v: T) => void }) => (
    <View style={styles.formGroup}>
      <Text style={styles.formLabel}>{label}</Text>
      <View style={styles.segmentRow}>
        {options.map((o) => (
          <TouchableOpacity
            key={o}
            style={[styles.segment, value === o && styles.segmentActive]}
            onPress={() => onChange(o)}
          >
            <Text style={[styles.segmentText, value === o && styles.segmentTextActive]}>
              {o.charAt(0).toUpperCase() + o.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <KeyboardAvoidingView
        style={styles.modalOverlay}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <View style={styles.modalSheet}>
          <View style={styles.modalHandle} />
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Add New Student</Text>
            <TouchableOpacity onPress={onClose} style={styles.modalCloseBtn}>
              <Ionicons name="close" size={22} color={COLORS.textMuted} />
            </TouchableOpacity>
          </View>
          <ScrollView style={styles.modalBody} showsVerticalScrollIndicator={false}>
            <Field label="FULL NAME" field="name" placeholder="Jane Doe" />
            <Field label="EMAIL ADDRESS" field="email" placeholder="jane@example.com" keyboard="email-address" />
            <Field label="TEMPORARY PASSWORD" field="password" placeholder="Min 6 characters" secure />
            <SegmentControl
              label="ROLE"
              options={['student', 'instructor'] as const}
              value={form.role}
              onChange={(v) => setForm({ ...form, role: v })}
            />
            <SegmentControl
              label="SUBSCRIPTION TIER"
              options={['free', 'pro', 'enterprise'] as const}
              value={form.tier}
              onChange={(v) => setForm({ ...form, tier: v })}
            />
            <TouchableOpacity onPress={handleSubmit} style={styles.submitBtn} activeOpacity={0.85}>
              <LinearGradient
                colors={[COLORS.primary, '#A855F7']}
                style={styles.submitBtnGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              >
                <Ionicons name="person-add" size={18} color="#fff" />
                <Text style={styles.submitBtnText}>Create Student Account</Text>
              </LinearGradient>
            </TouchableOpacity>
            <View style={{ height: 40 }} />
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

// ─── Student Detail Modal ─────────────────────────────────────
function StudentDetailModal({
  student,
  visible,
  onClose,
}: {
  student: Student | null;
  visible: boolean;
  onClose: () => void;
}) {
  if (!student) return null;
  const enrollments = MOCK_ENROLLMENTS[student.id] ?? [];
  const hue = student.name.charCodeAt(0) * 15 % 360;

  const handleAction = (action: string) => {
    Alert.alert(action, `Perform "${action}" on ${student.name}?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Confirm',
        style: action.toLowerCase().includes('delete') || action.toLowerCase().includes('suspend') ? 'destructive' : 'default',
        onPress: () => { Alert.alert('Done', `${action} completed successfully.`); onClose(); },
      },
    ]);
  };

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={styles.detailModal}>
        <LinearGradient colors={['#141428', '#0A0A1A']} style={styles.detailHeaderBg}>
          <TouchableOpacity onPress={onClose} style={styles.detailCloseBtn}>
            <Ionicons name="chevron-down" size={24} color={COLORS.textMuted} />
          </TouchableOpacity>
          <View style={[styles.detailAvatar, { backgroundColor: `hsl(${hue}, 55%, 32%)` }]}>
            <Text style={styles.detailAvatarText}>
              {student.name.split(' ').map((n) => n[0]).slice(0, 2).join('').toUpperCase()}
            </Text>
            <View style={[styles.detailOnlineDot, { backgroundColor: student.isActive ? COLORS.accent : COLORS.textMuted }]} />
          </View>
          <Text style={styles.detailName}>{student.name}</Text>
          <Text style={styles.detailEmail}>{student.email}</Text>
          <View style={styles.detailBadgeRow}>
            <View style={styles.detailBadge}>
              <Text style={[styles.detailBadgeText, { color: COLORS.primary }]}>Level {student.level}</Text>
            </View>
            <View style={[styles.detailBadge, { backgroundColor: student.subscriptionTier === 'pro' ? '#FFD70020' : student.subscriptionTier === 'enterprise' ? '#6C63FF20' : COLORS.surfaceLight }]}>
              <Text style={[styles.detailBadgeText, { color: student.subscriptionTier === 'pro' ? COLORS.gold : student.subscriptionTier === 'enterprise' ? COLORS.primary : COLORS.textMuted }]}>
                {student.subscriptionTier.toUpperCase()}
              </Text>
            </View>
            <View style={[styles.detailBadge, { backgroundColor: `${COLORS.warning}20` }]}>
              <Ionicons name="flash" size={11} color={COLORS.warning} />
              <Text style={[styles.detailBadgeText, { color: COLORS.warning }]}>{student.xp.toLocaleString()} XP</Text>
            </View>
          </View>
        </LinearGradient>

        <ScrollView style={{ flex: 1 }} showsVerticalScrollIndicator={false}>
          {/* Stats */}
          <View style={styles.detailSection}>
            <Text style={styles.detailSectionTitle}>Progress Stats</Text>
            <View style={styles.detailStatsRow}>
              {[
                { label: 'Enrolled', value: student.coursesEnrolled, icon: 'book-outline' as const, color: COLORS.blue },
                { label: 'Completed', value: student.coursesCompleted, icon: 'checkmark-circle-outline' as const, color: COLORS.accent },
                { label: 'Level', value: student.level, icon: 'star-outline' as const, color: COLORS.warning },
              ].map((s) => (
                <View key={s.label} style={styles.detailStatCard}>
                  <View style={[styles.detailStatIcon, { backgroundColor: `${s.color}20` }]}>
                    <Ionicons name={s.icon} size={18} color={s.color} />
                  </View>
                  <Text style={[styles.detailStatValue, { color: s.color }]}>{s.value}</Text>
                  <Text style={styles.detailStatLabel}>{s.label}</Text>
                </View>
              ))}
            </View>
          </View>

          {/* Account info */}
          <View style={styles.detailSection}>
            <Text style={styles.detailSectionTitle}>Account Information</Text>
            <View style={styles.infoCard}>
              {[
                { label: 'Member since', value: new Date(student.joinDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }) },
                { label: 'Last active', value: formatDate(student.lastActive) },
                { label: 'Status', value: student.isActive ? 'Active' : 'Inactive' },
              ].map((item, i) => (
                <View key={item.label} style={[styles.infoRow, i < 2 && { borderBottomWidth: 1, borderBottomColor: COLORS.border }]}>
                  <Text style={styles.infoLabel}>{item.label}</Text>
                  <Text style={[styles.infoValue, item.label === 'Status' && { color: student.isActive ? COLORS.accent : COLORS.textMuted }]}>
                    {item.value}
                  </Text>
                </View>
              ))}
            </View>
          </View>

          {/* Enrollment history */}
          {enrollments.length > 0 && (
            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>Enrollment History</Text>
              {enrollments.map((e) => (
                <View key={e.courseId} style={styles.enrollCard}>
                  <View style={styles.enrollTop}>
                    <Text style={styles.enrollName} numberOfLines={1}>{e.courseName}</Text>
                    {e.completed && (
                      <View style={styles.completedBadge}>
                        <Ionicons name="checkmark" size={10} color={COLORS.accent} />
                        <Text style={styles.completedText}>Done</Text>
                      </View>
                    )}
                  </View>
                  <View style={styles.progressTrack}>
                    <View style={[styles.progressFill, { width: `${e.progress}%`, backgroundColor: e.completed ? COLORS.accent : COLORS.primary }]} />
                  </View>
                  <Text style={styles.progressLabel}>{e.progress}% complete</Text>
                </View>
              ))}
            </View>
          )}

          {/* Admin actions */}
          <View style={styles.detailSection}>
            <Text style={styles.detailSectionTitle}>Admin Actions</Text>
            {[
              { label: 'Edit Profile', icon: 'pencil-outline' as const, color: COLORS.blue },
              { label: 'Reset Password', icon: 'key-outline' as const, color: COLORS.warning },
              { label: 'Grant Pro Access', icon: 'star-outline' as const, color: COLORS.gold },
              { label: 'Suspend Account', icon: 'ban-outline' as const, color: COLORS.orange },
              { label: 'Delete Account', icon: 'trash-outline' as const, color: COLORS.error },
            ].map((a) => (
              <TouchableOpacity
                key={a.label}
                style={[styles.actionCard, { borderColor: `${a.color}40` }]}
                onPress={() => handleAction(a.label)}
                activeOpacity={0.8}
              >
                <View style={[styles.actionCardIcon, { backgroundColor: `${a.color}20` }]}>
                  <Ionicons name={a.icon} size={18} color={a.color} />
                </View>
                <Text style={[styles.actionCardLabel, { color: a.color }]}>{a.label}</Text>
                <Ionicons name="chevron-forward" size={15} color={`${a.color}80`} />
              </TouchableOpacity>
            ))}
          </View>
          <View style={{ height: 40 }} />
        </ScrollView>
      </View>
    </Modal>
  );
}

// ─── Sort Sheet ───────────────────────────────────────────────
function SortSheet({
  visible,
  sort,
  onSelect,
  onClose,
}: {
  visible: boolean;
  sort: SortType;
  onSelect: (s: SortType) => void;
  onClose: () => void;
}) {
  const options: { key: SortType; label: string; icon: React.ComponentProps<typeof Ionicons>['name'] }[] = [
    { key: 'name', label: 'By Name (A–Z)', icon: 'text-outline' },
    { key: 'joinDate', label: 'By Join Date (Newest)', icon: 'calendar-outline' },
    { key: 'xp', label: 'By XP (Highest)', icon: 'flash-outline' },
    { key: 'lastActive', label: 'By Last Active', icon: 'time-outline' },
  ];
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <TouchableOpacity style={styles.sheetOverlay} onPress={onClose} activeOpacity={1}>
        <View style={styles.sortSheet}>
          <View style={styles.modalHandle} />
          <Text style={styles.sheetTitle}>Sort Students</Text>
          {options.map((o) => (
            <TouchableOpacity
              key={o.key}
              style={styles.sortOption}
              onPress={() => { onSelect(o.key); onClose(); }}
            >
              <Ionicons name={o.icon} size={18} color={sort === o.key ? COLORS.primary : COLORS.textMuted} />
              <Text style={[styles.sortOptionText, sort === o.key && { color: COLORS.primary }]}>{o.label}</Text>
              {sort === o.key && <Ionicons name="checkmark" size={18} color={COLORS.primary} />}
            </TouchableOpacity>
          ))}
          <View style={{ height: 24 }} />
        </View>
      </TouchableOpacity>
    </Modal>
  );
}

// ─── Main Screen ──────────────────────────────────────────────
export default function StudentsScreen() {
  const [students, setStudents] = useState<Student[]>(MOCK_STUDENTS);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<FilterType>('all');
  const [sort, setSort] = useState<SortType>('name');
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [detailVisible, setDetailVisible] = useState(false);
  const [addVisible, setAddVisible] = useState(false);
  const [showSortSheet, setShowSortSheet] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectionMode, setSelectionMode] = useState(false);

  const filtered = useMemo(() => {
    let list = [...students];
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((s) => s.name.toLowerCase().includes(q) || s.email.toLowerCase().includes(q));
    }
    if (filter === 'active') list = list.filter((s) => s.isActive);
    else if (filter === 'inactive') list = list.filter((s) => !s.isActive);
    else if (filter === 'pro') list = list.filter((s) => s.subscriptionTier !== 'free');
    else if (filter === 'free') list = list.filter((s) => s.subscriptionTier === 'free');
    list.sort((a, b) => {
      if (sort === 'name') return a.name.localeCompare(b.name);
      if (sort === 'xp') return b.xp - a.xp;
      if (sort === 'joinDate') return new Date(b.joinDate).getTime() - new Date(a.joinDate).getTime();
      if (sort === 'lastActive') return new Date(b.lastActive).getTime() - new Date(a.lastActive).getTime();
      return 0;
    });
    return list;
  }, [students, search, filter, sort]);

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const exitSelection = () => {
    setSelectionMode(false);
    setSelectedIds(new Set());
  };

  const handleBulkAction = (action: string) => {
    Alert.alert(`${action} (${selectedIds.size} students)`, 'Apply to all selected?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Proceed', onPress: () => { Alert.alert('Done', `${action} applied.`); exitSelection(); } },
    ]);
  };

  const handleAdd = useCallback((form: AddStudentForm) => {
    const newStudent: Student = {
      id: String(Date.now()),
      name: form.name,
      email: form.email,
      level: 1,
      subscriptionTier: form.tier,
      lastActive: new Date().toISOString(),
      xp: 0,
      isActive: true,
      joinDate: new Date().toISOString(),
      coursesEnrolled: 0,
      coursesCompleted: 0,
    };
    setStudents((prev) => [newStudent, ...prev]);
    setAddVisible(false);
    Alert.alert('Student Added', `${form.name} has been created successfully.`);
  }, []);

  const renderItem = useCallback(({ item }: { item: Student }) => (
    <StudentListRow
      student={item}
      selected={selectedIds.has(item.id)}
      selectionMode={selectionMode}
      onPress={() => {
        if (selectionMode) {
          toggleSelect(item.id);
        } else {
          setSelectedStudent(item);
          setDetailVisible(true);
        }
      }}
      onLongPress={() => {
        if (!selectionMode) setSelectionMode(true);
        toggleSelect(item.id);
      }}
    />
  ), [selectedIds, selectionMode, toggleSelect]);

  const filterOptions: { key: FilterType; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'active', label: 'Active' },
    { key: 'inactive', label: 'Inactive' },
    { key: 'pro', label: 'Pro' },
    { key: 'free', label: 'Free' },
  ];

  return (
    <SafeAreaView style={styles.safeArea} edges={['bottom']}>
      {/* Search row */}
      <View style={styles.searchBar}>
        <View style={styles.searchInputWrap}>
          <Ionicons name="search" size={16} color={COLORS.textMuted} />
          <TextInput
            style={styles.searchText}
            value={search}
            onChangeText={setSearch}
            placeholder="Search by name or email…"
            placeholderTextColor={COLORS.textMuted}
            returnKeyType="search"
          />
          {search.length > 0 && (
            <TouchableOpacity onPress={() => setSearch('')}>
              <Ionicons name="close-circle" size={16} color={COLORS.textMuted} />
            </TouchableOpacity>
          )}
        </View>
        <TouchableOpacity style={styles.iconBtn} onPress={() => setShowSortSheet(true)}>
          <Ionicons name="swap-vertical" size={18} color={COLORS.primary} />
        </TouchableOpacity>
        <TouchableOpacity style={styles.iconBtn} onPress={() => Alert.alert('Export', 'Exporting students.csv…')}>
          <Ionicons name="download-outline" size={18} color={COLORS.accent} />
        </TouchableOpacity>
      </View>

      {/* Filter chips */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.filterScroll}
        contentContainerStyle={styles.filterContent}
      >
        {filterOptions.map((f) => (
          <TouchableOpacity
            key={f.key}
            style={[styles.filterChip, filter === f.key && styles.filterChipActive]}
            onPress={() => setFilter(f.key)}
          >
            <Text style={[styles.filterChipText, filter === f.key && styles.filterChipTextActive]}>{f.label}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Status bar — count or bulk actions */}
      {selectionMode ? (
        <View style={styles.bulkBar}>
          <TouchableOpacity onPress={exitSelection}>
            <Ionicons name="close" size={20} color={COLORS.text} />
          </TouchableOpacity>
          <Text style={styles.bulkCount}>{selectedIds.size} selected</Text>
          <View style={styles.bulkActions}>
            {[
              { icon: 'notifications-outline' as const, action: 'Send Notification', color: COLORS.primary },
              { icon: 'star-outline' as const, action: 'Grant Pro Access', color: COLORS.gold },
              { icon: 'download-outline' as const, action: 'Export Selected', color: COLORS.accent },
            ].map((b) => (
              <TouchableOpacity key={b.action} style={styles.bulkActionBtn} onPress={() => handleBulkAction(b.action)}>
                <Ionicons name={b.icon} size={16} color={b.color} />
              </TouchableOpacity>
            ))}
          </View>
        </View>
      ) : (
        <View style={styles.countBar}>
          <Text style={styles.countText}>{filtered.length} student{filtered.length !== 1 ? 's' : ''}</Text>
          <Text style={styles.sortLabel}>Long-press to select</Text>
        </View>
      )}

      {/* List */}
      <FlatList
        data={filtered}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Ionicons name="people-outline" size={52} color={COLORS.border} />
            <Text style={styles.emptyTitle}>No students found</Text>
            <Text style={styles.emptySubtitle}>Try adjusting your search or filters</Text>
          </View>
        }
      />

      {/* FAB */}
      <TouchableOpacity style={styles.fab} onPress={() => setAddVisible(true)} activeOpacity={0.85}>
        <LinearGradient colors={[COLORS.primary, '#A855F7']} style={styles.fabGradient} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}>
          <Ionicons name="person-add" size={22} color="#fff" />
        </LinearGradient>
      </TouchableOpacity>

      {/* Modals */}
      <SortSheet visible={showSortSheet} sort={sort} onSelect={setSort} onClose={() => setShowSortSheet(false)} />
      <AddStudentModal visible={addVisible} onClose={() => setAddVisible(false)} onAdd={handleAdd} />
      <StudentDetailModal student={selectedStudent} visible={detailVisible} onClose={() => setDetailVisible(false)} />
    </SafeAreaView>
  );
}

// ─── Styles ──────────────────────────────────────────────────
const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: COLORS.background },

  // Search
  searchBar: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 8,
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  searchInputWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 9,
    gap: 8,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  searchText: { flex: 1, color: COLORS.text, fontSize: 14 },
  iconBtn: {
    width: 42,
    height: 42,
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  // Filters
  filterScroll: {
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    maxHeight: 52,
  },
  filterContent: { paddingHorizontal: 16, paddingVertical: 9, gap: 8 },
  filterChip: {
    paddingHorizontal: 16,
    paddingVertical: 7,
    borderRadius: 20,
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  filterChipActive: { backgroundColor: `${COLORS.primary}25`, borderColor: COLORS.primary },
  filterChipText: { color: COLORS.textMuted, fontSize: 13, fontWeight: '600' },
  filterChipTextActive: { color: COLORS.primary },

  // Count / bulk bars
  countBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  countText: { color: COLORS.textMuted, fontSize: 13, fontWeight: '600' },
  sortLabel: { color: COLORS.textMuted, fontSize: 11 },
  bulkBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    gap: 12,
    backgroundColor: `${COLORS.primary}15`,
    borderBottomWidth: 1,
    borderBottomColor: `${COLORS.primary}40`,
  },
  bulkCount: { flex: 1, color: COLORS.text, fontSize: 14, fontWeight: '700' },
  bulkActions: { flexDirection: 'row', gap: 8 },
  bulkActionBtn: {
    width: 36,
    height: 36,
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  // Student row
  listContent: { paddingHorizontal: 16, paddingTop: 8, paddingBottom: 100 },
  studentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
    gap: 12,
    marginBottom: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 6,
    elevation: 3,
  },
  studentRowSelected: { borderColor: COLORS.primary, backgroundColor: `${COLORS.primary}12` },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: COLORS.textMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxSelected: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  avatarWrapper: { position: 'relative' },
  avatarCircle: { alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: '#fff', fontWeight: '700' },
  onlineDot: {
    position: 'absolute',
    bottom: 1,
    right: 1,
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: COLORS.surface,
  },
  studentInfo: { flex: 1, gap: 3 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  studentName: { color: COLORS.text, fontSize: 14, fontWeight: '700', flex: 1 },
  studentEmail: { color: COLORS.textMuted, fontSize: 12 },
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2 },
  metaText: { color: COLORS.textMuted, fontSize: 11 },
  metaDot: { width: 3, height: 3, borderRadius: 1.5, backgroundColor: COLORS.textMuted, marginHorizontal: 2 },
  studentRight: { alignItems: 'flex-end', justifyContent: 'center' },

  // Level / tier badges
  levelBadge: {
    backgroundColor: `${COLORS.primary}25`,
    borderWidth: 1,
    borderColor: `${COLORS.primary}60`,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 6,
  },
  levelBadgeText: { color: COLORS.primary, fontSize: 10, fontWeight: '700' },
  tierBadge: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 7 },
  tierBadgeText: { fontSize: 10, fontWeight: '800', letterSpacing: 0.5 },

  // Empty
  emptyState: { alignItems: 'center', paddingTop: 80, gap: 12 },
  emptyTitle: { color: COLORS.text, fontSize: 18, fontWeight: '700' },
  emptySubtitle: { color: COLORS.textMuted, fontSize: 14 },

  // FAB
  fab: {
    position: 'absolute',
    bottom: 24,
    right: 20,
    borderRadius: 28,
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.45,
    shadowRadius: 12,
    elevation: 10,
  },
  fabGradient: { width: 56, height: 56, alignItems: 'center', justifyContent: 'center' },

  // Modal overlay
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.72)', justifyContent: 'flex-end' },
  modalSheet: {
    backgroundColor: COLORS.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingTop: 12,
    maxHeight: '92%',
    borderTopWidth: 1,
    borderColor: COLORS.border,
  },
  modalHandle: { width: 40, height: 4, backgroundColor: COLORS.border, borderRadius: 2, alignSelf: 'center', marginBottom: 16 },
  modalHeader: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 20, marginBottom: 8 },
  modalTitle: { flex: 1, color: COLORS.text, fontSize: 18, fontWeight: '800' },
  modalCloseBtn: {
    width: 36,
    height: 36,
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalBody: { paddingHorizontal: 20, paddingTop: 12 },

  // Form
  formGroup: { marginBottom: 18 },
  formLabel: { color: COLORS.textMuted, fontSize: 11, fontWeight: '700', letterSpacing: 0.6, textTransform: 'uppercase', marginBottom: 8 },
  formInput: {
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 13,
    color: COLORS.text,
    fontSize: 15,
  },
  formInputError: { borderColor: COLORS.error },
  formError: { color: COLORS.error, fontSize: 12, marginTop: 4 },
  segmentRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    overflow: 'hidden',
  },
  segment: { flex: 1, paddingVertical: 12, alignItems: 'center' },
  segmentActive: { backgroundColor: `${COLORS.primary}35` },
  segmentText: { color: COLORS.textMuted, fontSize: 13, fontWeight: '600' },
  segmentTextActive: { color: COLORS.primary, fontWeight: '800' },
  submitBtn: { borderRadius: 14, overflow: 'hidden', marginTop: 8 },
  submitBtnGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, gap: 10 },
  submitBtnText: { color: '#fff', fontSize: 16, fontWeight: '800' },

  // Sort sheet
  sheetOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.65)', justifyContent: 'flex-end' },
  sortSheet: {
    backgroundColor: COLORS.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingTop: 12,
    paddingHorizontal: 20,
    borderTopWidth: 1,
    borderColor: COLORS.border,
  },
  sheetTitle: { color: COLORS.text, fontSize: 16, fontWeight: '800', marginBottom: 16, textAlign: 'center' },
  sortOption: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  sortOptionText: { flex: 1, color: COLORS.text, fontSize: 15, fontWeight: '600' },

  // Detail modal
  detailModal: { flex: 1, backgroundColor: COLORS.background },
  detailHeaderBg: { paddingTop: 12, paddingBottom: 24, alignItems: 'center', borderBottomWidth: 1, borderBottomColor: COLORS.border },
  detailCloseBtn: { alignSelf: 'flex-start', marginLeft: 16, marginBottom: 16, padding: 4 },
  detailAvatar: { width: 72, height: 72, borderRadius: 36, alignItems: 'center', justifyContent: 'center', marginBottom: 12, position: 'relative' },
  detailAvatarText: { color: '#fff', fontSize: 26, fontWeight: '800' },
  detailOnlineDot: { position: 'absolute', bottom: 3, right: 3, width: 14, height: 14, borderRadius: 7, borderWidth: 3, borderColor: '#141428' },
  detailName: { color: COLORS.text, fontSize: 22, fontWeight: '800', marginBottom: 4 },
  detailEmail: { color: COLORS.textMuted, fontSize: 14, marginBottom: 14 },
  detailBadgeRow: { flexDirection: 'row', gap: 8 },
  detailBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: `${COLORS.primary}20`,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
  },
  detailBadgeText: { fontSize: 12, fontWeight: '700' },
  detailSection: { paddingHorizontal: 16, paddingTop: 20 },
  detailSectionTitle: { color: COLORS.text, fontSize: 15, fontWeight: '800', marginBottom: 12 },
  detailStatsRow: { flexDirection: 'row', gap: 10 },
  detailStatCard: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
    alignItems: 'center',
    gap: 6,
  },
  detailStatIcon: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  detailStatValue: { fontSize: 22, fontWeight: '800' },
  detailStatLabel: { color: COLORS.textMuted, fontSize: 12, fontWeight: '600' },
  infoCard: { backgroundColor: COLORS.surface, borderRadius: 14, borderWidth: 1, borderColor: COLORS.border, overflow: 'hidden' },
  infoRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 14 },
  infoLabel: { color: COLORS.textMuted, fontSize: 13 },
  infoValue: { color: COLORS.text, fontSize: 13, fontWeight: '600', maxWidth: '55%', textAlign: 'right' },
  enrollCard: { backgroundColor: COLORS.surface, borderRadius: 12, borderWidth: 1, borderColor: COLORS.border, padding: 14, marginBottom: 10 },
  enrollTop: { flexDirection: 'row', alignItems: 'center', marginBottom: 10, gap: 8 },
  enrollName: { flex: 1, color: COLORS.text, fontSize: 14, fontWeight: '700' },
  completedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    backgroundColor: `${COLORS.accent}20`,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  completedText: { color: COLORS.accent, fontSize: 11, fontWeight: '700' },
  progressTrack: { height: 6, backgroundColor: COLORS.border, borderRadius: 3, overflow: 'hidden', marginBottom: 6 },
  progressFill: { height: '100%', borderRadius: 3 },
  progressLabel: { color: COLORS.textMuted, fontSize: 11 },
  actionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    borderWidth: 1,
    padding: 14,
    marginBottom: 10,
  },
  actionCardIcon: { width: 40, height: 40, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  actionCardLabel: { flex: 1, fontSize: 15, fontWeight: '700' },
});
