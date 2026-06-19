import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  Modal,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { LinearGradient } from 'expo-linear-gradient';
import { Colors } from '../../../constants/colors';
import { GradientButton } from '../../../components/common/GradientButton';
import { EmptyState } from '../../../components/common/EmptyState';
import { supabase } from '../../../lib/supabase';
import type { Course } from '../../../types';

const CATEGORIES = ['Web Development', 'Data Science', 'Mobile Development', 'AI / ML', 'Cybersecurity', 'Cloud Computing', 'Design', 'Business'];
const LEVELS = ['Beginner', 'Intermediate', 'Advanced'];
const CONTENT_TYPES = ['video', 'text', 'quiz'] as const;

interface LessonDraft {
  id: string;
  title: string;
  content_type: 'video' | 'text' | 'quiz';
  content_url: string;
  duration_minutes: number;
}

interface ModuleDraft {
  id: string;
  title: string;
  lessons: LessonDraft[];
}

interface CourseDraft {
  title: string;
  description: string;
  category: string;
  thumbnail_url: string;
  is_premium: boolean;
  instructor_name: string;
  difficulty: string;
  modules: ModuleDraft[];
  published: boolean;
}

const emptyDraft = (): CourseDraft => ({
  title: '',
  description: '',
  category: CATEGORIES[0],
  thumbnail_url: '',
  is_premium: false,
  instructor_name: '',
  difficulty: 'Beginner',
  modules: [],
  published: false,
});

export default function AdminCoursesScreen() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filterPublished, setFilterPublished] = useState<'all' | 'published' | 'draft'>('all');
  const [builderVisible, setBuilderVisible] = useState(false);
  const [draft, setDraft] = useState<CourseDraft>(emptyDraft());
  const [step, setStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [editingCourse, setEditingCourse] = useState<Course | null>(null);

  const loadCourses = useCallback(async () => {
    const { data } = await supabase.from('courses').select('*').order('created_at', { ascending: false });
    if (data) setCourses(data as Course[]);
    setLoading(false);
    setRefreshing(false);
  }, []);

  useEffect(() => { loadCourses(); }, [loadCourses]);

  const filtered = courses.filter((c) => {
    if (filterPublished === 'published') return c.published;
    if (filterPublished === 'draft') return !c.published;
    return true;
  });

  const pickThumbnail = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.8 });
    if (!result.canceled && result.assets[0]) {
      setDraft((d) => ({ ...d, thumbnail_url: result.assets[0].uri }));
    }
  };

  const addModule = () => {
    setDraft((d) => ({
      ...d,
      modules: [...d.modules, { id: Date.now().toString(), title: `Module ${d.modules.length + 1}`, lessons: [] }],
    }));
  };

  const updateModule = (moduleId: string, title: string) => {
    setDraft((d) => ({ ...d, modules: d.modules.map((m) => m.id === moduleId ? { ...m, title } : m) }));
  };

  const addLesson = (moduleId: string) => {
    const newLesson: LessonDraft = { id: Date.now().toString(), title: 'New Lesson', content_type: 'video', content_url: '', duration_minutes: 0 };
    setDraft((d) => ({ ...d, modules: d.modules.map((m) => m.id === moduleId ? { ...m, lessons: [...m.lessons, newLesson] } : m) }));
  };

  const updateLesson = (moduleId: string, lessonId: string, updates: Partial<LessonDraft>) => {
    setDraft((d) => ({
      ...d,
      modules: d.modules.map((m) =>
        m.id === moduleId
          ? { ...m, lessons: m.lessons.map((l) => l.id === lessonId ? { ...l, ...updates } : l) }
          : m
      ),
    }));
  };

  const removeLesson = (moduleId: string, lessonId: string) => {
    setDraft((d) => ({
      ...d,
      modules: d.modules.map((m) => m.id === moduleId ? { ...m, lessons: m.lessons.filter((l) => l.id !== lessonId) } : m),
    }));
  };

  const removeModule = (moduleId: string) => {
    setDraft((d) => ({ ...d, modules: d.modules.filter((m) => m.id !== moduleId) }));
  };

  const saveCourse = async (publish: boolean) => {
    if (!draft.title.trim()) { Alert.alert('Error', 'Course title is required'); return; }
    setSaving(true);

    const { data: { user } } = await supabase.auth.getUser();
    const courseData = {
      title: draft.title,
      description: draft.description,
      category: draft.category,
      thumbnail_url: draft.thumbnail_url || null,
      is_premium: draft.is_premium,
      instructor_name: draft.instructor_name,
      published: publish,
      created_by: user?.id,
    };

    let courseId: string | null = editingCourse?.id ?? null;

    if (editingCourse) {
      await supabase.from('courses').update(courseData).eq('id', editingCourse.id);
    } else {
      const { data } = await supabase.from('courses').insert(courseData).select().single();
      courseId = data?.id ?? null;
    }

    if (courseId) {
      for (let mi = 0; mi < draft.modules.length; mi++) {
        const mod = draft.modules[mi];
        const { data: modData } = await supabase.from('modules').insert({ course_id: courseId, title: mod.title, order_index: mi }).select().single();
        if (modData) {
          for (let li = 0; li < mod.lessons.length; li++) {
            const les = mod.lessons[li];
            await supabase.from('lessons').insert({ module_id: modData.id, title: les.title, content_type: les.content_type, content_url: les.content_url, duration_minutes: les.duration_minutes, order_index: li });
          }
        }
      }
    }

    setSaving(false);
    setBuilderVisible(false);
    setDraft(emptyDraft());
    setStep(0);
    setEditingCourse(null);
    loadCourses();
  };

  const handleEdit = (course: Course) => {
    setEditingCourse(course);
    setDraft({
      title: course.title,
      description: course.description ?? '',
      category: course.category,
      thumbnail_url: course.thumbnail_url ?? '',
      is_premium: course.is_premium ?? false,
      instructor_name: course.instructor_name ?? '',
      difficulty: 'Beginner',
      modules: [],
      published: course.published ?? false,
    });
    setStep(0);
    setBuilderVisible(true);
  };

  const handleDelete = (course: Course) => {
    Alert.alert('Delete Course', `Delete "${course.title}"? This cannot be undone.`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive',
        onPress: async () => {
          await supabase.from('courses').delete().eq('id', course.id);
          loadCourses();
        },
      },
    ]);
  };

  const renderCourse = ({ item }: { item: Course }) => (
    <View style={styles.courseCard}>
      {item.thumbnail_url ? (
        <Image source={{ uri: item.thumbnail_url }} style={styles.courseThumb} />
      ) : (
        <LinearGradient colors={Colors.gradients.primary as any} style={styles.courseThumb}>
          <Ionicons name="book" size={32} color="rgba(255,255,255,0.6)" />
        </LinearGradient>
      )}
      <View style={styles.courseInfo}>
        <View style={styles.courseHeader}>
          <Text style={styles.courseTitle} numberOfLines={2}>{item.title}</Text>
          <View style={[styles.statusBadge, { backgroundColor: item.published ? Colors.accent + '22' : Colors.warning + '22' }]}>
            <Text style={[styles.statusText, { color: item.published ? Colors.accent : Colors.warning }]}>
              {item.published ? 'Live' : 'Draft'}
            </Text>
          </View>
        </View>
        <Text style={styles.courseCategory}>{item.category}</Text>
        <View style={styles.courseStats}>
          <Text style={styles.courseStat}>👥 {item.total_students ?? 0}</Text>
          <Text style={styles.courseStat}>⭐ {item.rating ?? '0.0'}</Text>
          {item.is_premium && <Text style={[styles.courseStat, { color: Colors.gold }]}>PRO</Text>}
        </View>
      </View>
      <View style={styles.courseActions}>
        <TouchableOpacity onPress={() => handleEdit(item)} style={styles.actionIcon}>
          <Ionicons name="create-outline" size={20} color={Colors.primary} />
        </TouchableOpacity>
        <TouchableOpacity onPress={() => handleDelete(item)} style={styles.actionIcon}>
          <Ionicons name="trash-outline" size={20} color={Colors.error} />
        </TouchableOpacity>
      </View>
    </View>
  );

  const STEPS = ['Basic Info', 'Curriculum', 'Settings'];

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Courses</Text>
        <TouchableOpacity style={styles.createBtn} onPress={() => { setDraft(emptyDraft()); setStep(0); setEditingCourse(null); setBuilderVisible(true); }}>
          <Ionicons name="add" size={22} color="#fff" />
          <Text style={styles.createBtnText}>Create</Text>
        </TouchableOpacity>
      </View>

      {/* Filter */}
      <View style={styles.filterRow}>
        {(['all', 'published', 'draft'] as const).map((f) => (
          <TouchableOpacity key={f} style={[styles.chip, filterPublished === f && styles.chipActive]} onPress={() => setFilterPublished(f)}>
            <Text style={[styles.chipText, filterPublished === f && styles.chipTextActive]}>
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
        <Text style={styles.countText}>{filtered.length} courses</Text>
      </View>

      {loading ? (
        <ActivityIndicator color={Colors.primary} style={{ marginTop: 40 }} />
      ) : filtered.length === 0 ? (
        <EmptyState icon="book-outline" title="No courses yet" subtitle="Tap Create to build your first course" actionLabel="Create Course" onAction={() => setBuilderVisible(true)} />
      ) : (
        <FlatList
          data={filtered}
          keyExtractor={(item) => item.id}
          renderItem={renderCourse}
          contentContainerStyle={{ padding: 16, paddingTop: 8 }}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => { setRefreshing(true); loadCourses(); }} tintColor={Colors.primary} />}
        />
      )}

      {/* Course Builder Modal */}
      <Modal visible={builderVisible} animationType="slide">
        <SafeAreaView style={styles.builderContainer} edges={['top', 'bottom']}>
          {/* Builder Header */}
          <View style={styles.builderHeader}>
            <TouchableOpacity onPress={() => setBuilderVisible(false)}>
              <Ionicons name="close" size={24} color={Colors.text} />
            </TouchableOpacity>
            <Text style={styles.builderTitle}>{editingCourse ? 'Edit Course' : 'New Course'}</Text>
            <Text style={styles.builderStep}>{step + 1}/{STEPS.length}</Text>
          </View>

          {/* Step indicator */}
          <View style={styles.stepIndicator}>
            {STEPS.map((s, i) => (
              <TouchableOpacity key={s} onPress={() => setStep(i)} style={styles.stepItem}>
                <View style={[styles.stepDot, i <= step && styles.stepDotActive]}>
                  <Text style={[styles.stepDotText, i <= step && styles.stepDotTextActive]}>{i + 1}</Text>
                </View>
                <Text style={[styles.stepLabel, i === step && styles.stepLabelActive]}>{s}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <ScrollView contentContainerStyle={{ padding: 20, paddingBottom: 100 }}>
            {/* Step 1: Basic Info */}
            {step === 0 && (
              <View style={styles.formSection}>
                <View style={styles.field}>
                  <Text style={styles.fieldLabel}>Course Title *</Text>
                  <TextInput style={styles.fieldInput} value={draft.title} onChangeText={(t) => setDraft((d) => ({ ...d, title: t }))} placeholder="e.g. Complete Web Development Bootcamp" placeholderTextColor={Colors.textMuted} />
                </View>
                <View style={styles.field}>
                  <Text style={styles.fieldLabel}>Description</Text>
                  <TextInput style={[styles.fieldInput, { height: 100, textAlignVertical: 'top' }]} value={draft.description} onChangeText={(t) => setDraft((d) => ({ ...d, description: t }))} multiline placeholder="What will students learn?" placeholderTextColor={Colors.textMuted} />
                </View>
                <View style={styles.field}>
                  <Text style={styles.fieldLabel}>Instructor Name</Text>
                  <TextInput style={styles.fieldInput} value={draft.instructor_name} onChangeText={(t) => setDraft((d) => ({ ...d, instructor_name: t }))} placeholder="Your name" placeholderTextColor={Colors.textMuted} />
                </View>
                <View style={styles.field}>
                  <Text style={styles.fieldLabel}>Category</Text>
                  <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                    <View style={{ flexDirection: 'row', gap: 8 }}>
                      {CATEGORIES.map((cat) => (
                        <TouchableOpacity key={cat} style={[styles.chip, draft.category === cat && styles.chipActive]} onPress={() => setDraft((d) => ({ ...d, category: cat }))}>
                          <Text style={[styles.chipText, draft.category === cat && styles.chipTextActive]}>{cat}</Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  </ScrollView>
                </View>
                <View style={styles.field}>
                  <Text style={styles.fieldLabel}>Thumbnail</Text>
                  <TouchableOpacity style={styles.thumbPicker} onPress={pickThumbnail}>
                    {draft.thumbnail_url ? (
                      <Image source={{ uri: draft.thumbnail_url }} style={{ width: '100%', height: '100%', borderRadius: 12 }} />
                    ) : (
                      <>
                        <Ionicons name="image-outline" size={32} color={Colors.textMuted} />
                        <Text style={styles.thumbPickerText}>Tap to select image</Text>
                      </>
                    )}
                  </TouchableOpacity>
                </View>
                <View style={styles.toggleRow}>
                  <View>
                    <Text style={styles.fieldLabel}>Premium Course</Text>
                    <Text style={styles.fieldHint}>Requires Pro subscription</Text>
                  </View>
                  <Switch value={draft.is_premium} onValueChange={(v) => setDraft((d) => ({ ...d, is_premium: v }))} trackColor={{ true: Colors.primary }} thumbColor="#fff" />
                </View>
              </View>
            )}

            {/* Step 2: Curriculum */}
            {step === 1 && (
              <View>
                {draft.modules.map((mod, mi) => (
                  <View key={mod.id} style={styles.moduleBlock}>
                    <View style={styles.moduleHeader}>
                      <TextInput style={styles.moduleTitle} value={mod.title} onChangeText={(t) => updateModule(mod.id, t)} />
                      <TouchableOpacity onPress={() => removeModule(mod.id)}>
                        <Ionicons name="trash-outline" size={18} color={Colors.error} />
                      </TouchableOpacity>
                    </View>
                    {mod.lessons.map((les) => (
                      <View key={les.id} style={styles.lessonRow}>
                        <Ionicons name={les.content_type === 'video' ? 'play-circle-outline' : les.content_type === 'quiz' ? 'help-circle-outline' : 'document-text-outline'} size={18} color={Colors.primary} />
                        <TextInput
                          style={styles.lessonInput}
                          value={les.title}
                          onChangeText={(t) => updateLesson(mod.id, les.id, { title: t })}
                          placeholder="Lesson title"
                          placeholderTextColor={Colors.textMuted}
                        />
                        <View style={styles.lessonTypeRow}>
                          {CONTENT_TYPES.map((ct) => (
                            <TouchableOpacity key={ct} style={[styles.typeChip, les.content_type === ct && styles.typeChipActive]} onPress={() => updateLesson(mod.id, les.id, { content_type: ct })}>
                              <Text style={[styles.typeChipText, les.content_type === ct && styles.typeChipTextActive]}>{ct}</Text>
                            </TouchableOpacity>
                          ))}
                        </View>
                        <TouchableOpacity onPress={() => removeLesson(mod.id, les.id)}>
                          <Ionicons name="remove-circle-outline" size={18} color={Colors.error} />
                        </TouchableOpacity>
                      </View>
                    ))}
                    <TouchableOpacity style={styles.addLessonBtn} onPress={() => addLesson(mod.id)}>
                      <Ionicons name="add" size={16} color={Colors.primary} />
                      <Text style={styles.addLessonText}>Add Lesson</Text>
                    </TouchableOpacity>
                  </View>
                ))}
                <TouchableOpacity style={styles.addModuleBtn} onPress={addModule}>
                  <Ionicons name="add-circle-outline" size={20} color={Colors.primary} />
                  <Text style={styles.addModuleText}>Add Module</Text>
                </TouchableOpacity>
              </View>
            )}

            {/* Step 3: Settings & Publish */}
            {step === 2 && (
              <View style={styles.formSection}>
                <View style={styles.field}>
                  <Text style={styles.fieldLabel}>Difficulty Level</Text>
                  <View style={{ flexDirection: 'row', gap: 8 }}>
                    {LEVELS.map((lv) => (
                      <TouchableOpacity key={lv} style={[styles.chip, draft.difficulty === lv && styles.chipActive]} onPress={() => setDraft((d) => ({ ...d, difficulty: lv }))}>
                        <Text style={[styles.chipText, draft.difficulty === lv && styles.chipTextActive]}>{lv}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                </View>
                <View style={styles.previewCard}>
                  <Text style={styles.previewLabel}>Course Summary</Text>
                  <Text style={styles.previewValue}>{draft.title || 'Untitled Course'}</Text>
                  <Text style={styles.previewMeta}>{draft.category} · {draft.difficulty} · {draft.modules.length} modules · {draft.modules.reduce((t, m) => t + m.lessons.length, 0)} lessons</Text>
                  <Text style={styles.previewMeta}>{draft.is_premium ? 'PRO' : 'Free'} · by {draft.instructor_name || 'Unknown'}</Text>
                </View>
                <View style={styles.publishActions}>
                  <GradientButton label="Save as Draft" onPress={() => saveCourse(false)} loading={saving} gradient={Colors.gradients.dark as any} style={{ flex: 1 }} />
                  <GradientButton label="Publish" onPress={() => saveCourse(true)} loading={saving} style={{ flex: 1 }} />
                </View>
              </View>
            )}
          </ScrollView>

          {/* Builder Navigation */}
          <View style={styles.builderNav}>
            {step > 0 && (
              <TouchableOpacity style={styles.navBack} onPress={() => setStep((s) => s - 1)}>
                <Ionicons name="chevron-back" size={20} color={Colors.text} />
                <Text style={styles.navBackText}>Back</Text>
              </TouchableOpacity>
            )}
            <View style={{ flex: 1 }} />
            {step < STEPS.length - 1 && (
              <GradientButton label="Next" onPress={() => setStep((s) => s + 1)} size="md" />
            )}
          </View>
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingTop: 8, paddingBottom: 12 },
  title: { fontSize: 28, fontWeight: '800', color: Colors.text },
  createBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: Colors.primary, paddingHorizontal: 16, paddingVertical: 9, borderRadius: 20 },
  createBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  filterRow: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, gap: 8, marginBottom: 8 },
  chip: { paddingHorizontal: 14, paddingVertical: 7, borderRadius: 20, backgroundColor: Colors.surface, borderWidth: 1, borderColor: Colors.border },
  chipActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  chipText: { fontSize: 13, color: Colors.textSecondary, fontWeight: '500' },
  chipTextActive: { color: '#fff' },
  countText: { marginLeft: 'auto', fontSize: 13, color: Colors.textMuted },
  courseCard: { backgroundColor: Colors.surface, borderRadius: 16, marginBottom: 12, overflow: 'hidden', flexDirection: 'row' },
  courseThumb: { width: 90, height: 90, alignItems: 'center', justifyContent: 'center' },
  courseInfo: { flex: 1, padding: 12, gap: 4 },
  courseHeader: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  courseTitle: { flex: 1, fontSize: 14, fontWeight: '700', color: Colors.text },
  statusBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
  statusText: { fontSize: 11, fontWeight: '700' },
  courseCategory: { fontSize: 12, color: Colors.primary },
  courseStats: { flexDirection: 'row', gap: 12 },
  courseStat: { fontSize: 12, color: Colors.textSecondary },
  courseActions: { justifyContent: 'center', gap: 8, paddingRight: 12 },
  actionIcon: { padding: 6 },
  // Builder
  builderContainer: { flex: 1, backgroundColor: Colors.background },
  builderHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingVertical: 16, borderBottomWidth: 1, borderColor: Colors.border },
  builderTitle: { fontSize: 18, fontWeight: '700', color: Colors.text },
  builderStep: { fontSize: 14, color: Colors.textMuted },
  stepIndicator: { flexDirection: 'row', paddingHorizontal: 20, paddingVertical: 12, gap: 0 },
  stepItem: { flex: 1, alignItems: 'center', gap: 4 },
  stepDot: { width: 28, height: 28, borderRadius: 14, backgroundColor: Colors.surface, borderWidth: 2, borderColor: Colors.border, alignItems: 'center', justifyContent: 'center' },
  stepDotActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  stepDotText: { fontSize: 13, fontWeight: '700', color: Colors.textMuted },
  stepDotTextActive: { color: '#fff' },
  stepLabel: { fontSize: 11, color: Colors.textMuted },
  stepLabelActive: { color: Colors.primary, fontWeight: '600' },
  formSection: { gap: 16 },
  field: { gap: 8 },
  fieldLabel: { fontSize: 14, color: Colors.textSecondary, fontWeight: '600' },
  fieldHint: { fontSize: 12, color: Colors.textMuted, marginTop: -4 },
  fieldInput: { backgroundColor: Colors.surface, borderRadius: 12, padding: 14, color: Colors.text, fontSize: 15, borderWidth: 1, borderColor: Colors.border },
  toggleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: Colors.surface, borderRadius: 12, padding: 16 },
  thumbPicker: { height: 140, backgroundColor: Colors.surface, borderRadius: 12, borderWidth: 1, borderColor: Colors.border, borderStyle: 'dashed', alignItems: 'center', justifyContent: 'center', gap: 8 },
  thumbPickerText: { color: Colors.textMuted, fontSize: 13 },
  moduleBlock: { backgroundColor: Colors.surface, borderRadius: 14, padding: 14, marginBottom: 12 },
  moduleHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  moduleTitle: { flex: 1, color: Colors.text, fontSize: 15, fontWeight: '700', borderBottomWidth: 1, borderColor: Colors.border, paddingBottom: 4 },
  lessonRow: { backgroundColor: Colors.surfaceLight, borderRadius: 10, padding: 10, marginBottom: 8, gap: 8 },
  lessonInput: { color: Colors.text, fontSize: 14 },
  lessonTypeRow: { flexDirection: 'row', gap: 6 },
  typeChip: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8, backgroundColor: Colors.border },
  typeChipActive: { backgroundColor: Colors.primary },
  typeChipText: { fontSize: 11, color: Colors.textSecondary },
  typeChipTextActive: { color: '#fff' },
  addLessonBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 8 },
  addLessonText: { color: Colors.primary, fontSize: 13, fontWeight: '600' },
  addModuleBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 1, borderColor: Colors.primary, borderRadius: 12, paddingVertical: 14, borderStyle: 'dashed' },
  addModuleText: { color: Colors.primary, fontWeight: '600' },
  previewCard: { backgroundColor: Colors.surface, borderRadius: 14, padding: 16, gap: 6 },
  previewLabel: { fontSize: 12, color: Colors.textMuted, fontWeight: '600', textTransform: 'uppercase' },
  previewValue: { fontSize: 18, fontWeight: '800', color: Colors.text },
  previewMeta: { fontSize: 13, color: Colors.textSecondary },
  publishActions: { flexDirection: 'row', gap: 12 },
  builderNav: { flexDirection: 'row', alignItems: 'center', padding: 20, paddingBottom: 32, borderTopWidth: 1, borderColor: Colors.border, backgroundColor: Colors.background },
  navBack: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  navBackText: { color: Colors.text, fontWeight: '600' },
});
