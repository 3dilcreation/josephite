// ============================================================
// TechSei LMS — Lesson Player Screen
// ============================================================
import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Animated,
  TextInput,
  Dimensions,
  ActivityIndicator,
  Alert,
  Platform,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Video, ResizeMode, AVPlaybackStatus } from 'expo-av';
import { WebView } from 'react-native-webview';
import { useCourseStore } from '../../../../stores/courseStore';
import { useAuthStore } from '../../../../stores/authStore';
import { useGamificationStore, XP_REWARDS } from '../../../../stores/gamificationStore';
import { ProgressBar } from '../../../../components/common/ProgressBar';
import type { Lesson, ContentType } from '../../../../types';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const COLORS = {
  background: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textMuted: '#8A8AAA',
  border: '#2A2A4A',
  error: '#FF6B6B',
};

const SPEED_OPTIONS = [0.75, 1, 1.25, 1.5, 2] as const;
type PlaybackSpeed = typeof SPEED_OPTIONS[number];

// ── Mock quiz data ────────────────────────────────────────────────────────────
const MOCK_QUIZ = {
  id: 'q1',
  lesson_id: 'l3',
  title: 'Knowledge Check',
  passing_score: 70,
  questions: [
    {
      id: 'q1_1',
      question: 'What is the primary purpose of this lesson?',
      options: ['Entertainment', 'Learning new skills', 'Testing memory', 'Social interaction'],
      correct_index: 1,
      explanation: 'This course is designed to help you learn and build new skills effectively.',
    },
    {
      id: 'q1_2',
      question: 'Which concept was introduced in the first section?',
      options: ['Advanced algorithms', 'Core fundamentals', 'Database design', 'Network protocols'],
      correct_index: 1,
      explanation: 'The first section covers core fundamentals to build a strong foundation.',
    },
    {
      id: 'q1_3',
      question: 'How should you approach learning new material?',
      options: ['Rush through it', 'Take notes and practice', 'Watch once and move on', 'Skip difficult parts'],
      correct_index: 1,
      explanation: 'Taking notes and practicing is the most effective learning strategy.',
    },
  ],
};

// ── Mock lesson data ──────────────────────────────────────────────────────────
const MOCK_LESSON_TEXT = `
## Introduction to Core Concepts

Welcome to this lesson! Here we'll cover the foundational concepts that form the building blocks of this course.

### What You'll Learn

In this lesson, we explore:
- The fundamental principles behind the technology
- How these concepts are applied in real-world scenarios
- Best practices that industry professionals follow

### Code Example

\`\`\`javascript
// Example: A simple function
function greet(name) {
  return \`Hello, \${name}! Welcome to TechSei.\`;
}

console.log(greet('Student'));
// Output: Hello, Student! Welcome to TechSei.
\`\`\`

### Key Takeaways

Understanding these basics will help you progress through more advanced topics with confidence. Practice what you've learned by completing the exercises at the end of each module.

> **Pro Tip:** Review your notes after each lesson to reinforce what you've learned. Spaced repetition is a powerful learning technique!
`;

// ── XP Animation ──────────────────────────────────────────────────────────────
function XPPopup({ xp, onDone }: { xp: number; onDone: () => void }) {
  const float = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(0)).current;
  const scale = useRef(new Animated.Value(0.5)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.spring(scale, { toValue: 1, useNativeDriver: true, tension: 100, friction: 6 }),
        Animated.timing(opacity, { toValue: 1, duration: 300, useNativeDriver: true }),
      ]),
      Animated.delay(1000),
      Animated.parallel([
        Animated.timing(float, { toValue: -60, duration: 600, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 0, duration: 600, useNativeDriver: true }),
      ]),
    ]).start(onDone);
  }, [float, opacity, scale, onDone]);

  return (
    <Animated.View
      style={[
        styles.xpPopup,
        { opacity, transform: [{ translateY: float }, { scale }] },
      ]}
    >
      <Ionicons name="flash" size={18} color="#000" />
      <Text style={styles.xpPopupText}>+{xp} XP</Text>
    </Animated.View>
  );
}

// ── Rich Text Renderer ────────────────────────────────────────────────────────
function RichTextContent({ content }: { content: string }) {
  const lines = content.split('\n');
  return (
    <ScrollView style={styles.textContent} showsVerticalScrollIndicator={false}>
      {lines.map((line, i) => {
        if (line.startsWith('## ')) {
          return <Text key={i} style={styles.h2}>{line.replace('## ', '')}</Text>;
        }
        if (line.startsWith('### ')) {
          return <Text key={i} style={styles.h3}>{line.replace('### ', '')}</Text>;
        }
        if (line.startsWith('- ')) {
          return (
            <View key={i} style={styles.bulletItem}>
              <View style={styles.bullet} />
              <Text style={styles.bulletItemText}>{line.replace('- ', '')}</Text>
            </View>
          );
        }
        if (line.startsWith('```')) {
          return null;
        }
        if (line.startsWith('> ')) {
          return (
            <View key={i} style={styles.quoteBlock}>
              <Text style={styles.quoteText}>{line.replace('> ', '').replace(/\*\*(.*?)\*\*/g, '$1')}</Text>
            </View>
          );
        }
        if (line.trim() === '') {
          return <View key={i} style={{ height: 8 }} />;
        }
        // Code block content (simple detection)
        if (line.includes('function') || line.includes('console') || line.includes('//') || line.includes('return')) {
          return <Text key={i} style={styles.codeText}>{line}</Text>;
        }
        return <Text key={i} style={styles.bodyText}>{line.replace(/\*\*(.*?)\*\*/g, '$1')}</Text>;
      })}
      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

// ── Video Player ──────────────────────────────────────────────────────────────
function VideoPlayer({
  uri,
  onComplete,
}: {
  uri: string | null;
  onComplete: () => void;
}) {
  const videoRef = useRef<Video>(null);
  const [status, setStatus] = useState<AVPlaybackStatus | null>(null);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);
  const [showSpeed, setShowSpeed] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);

  const isPlaying = status?.isLoaded ? status.isPlaying : false;
  const duration = status?.isLoaded ? status.durationMillis ?? 0 : 0;
  const position = status?.isLoaded ? status.positionMillis ?? 0 : 0;
  const progress = duration > 0 ? (position / duration) * 100 : 0;

  const togglePlay = async () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      await videoRef.current.pauseAsync();
    } else {
      await videoRef.current.playAsync();
    }
  };

  const handlePlaybackUpdate = (s: AVPlaybackStatus) => {
    setStatus(s);
    if (s.isLoaded && s.didJustFinish) {
      onComplete();
    }
  };

  const seek = async (direction: 'back' | 'forward') => {
    if (!videoRef.current || !status?.isLoaded) return;
    const newPos = Math.max(0, position + (direction === 'forward' ? 10000 : -10000));
    await videoRef.current.setPositionAsync(newPos);
  };

  const setPlaybackSpeed = async (s: PlaybackSpeed) => {
    if (!videoRef.current) return;
    await videoRef.current.setRateAsync(s, true);
    setSpeed(s);
    setShowSpeed(false);
  };

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <View style={styles.videoContainer}>
      <Video
        ref={videoRef}
        style={styles.video}
        source={uri ? { uri } : require('../../../../assets/images/icon.png') as any}
        resizeMode={ResizeMode.CONTAIN}
        shouldPlay={false}
        onPlaybackStatusUpdate={handlePlaybackUpdate}
        useNativeControls={false}
      />

      {/* Custom controls overlay */}
      <View style={styles.videoControls}>
        {/* Speed selector */}
        <View style={styles.videoTopBar}>
          <View style={{ flex: 1 }} />
          <TouchableOpacity
            onPress={() => setShowSpeed((v) => !v)}
            style={styles.speedBtn}
          >
            <Text style={styles.speedBtnText}>{speed}x</Text>
          </TouchableOpacity>
          {showSpeed && (
            <View style={styles.speedMenu}>
              {SPEED_OPTIONS.map((s) => (
                <TouchableOpacity
                  key={s}
                  style={[styles.speedOption, speed === s && styles.speedOptionActive]}
                  onPress={() => setPlaybackSpeed(s)}
                >
                  <Text style={[styles.speedOptionText, speed === s && { color: COLORS.primary }]}>
                    {s}x
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>

        {/* Center controls */}
        <View style={styles.videoCenterControls}>
          <TouchableOpacity onPress={() => seek('back')} style={styles.videoControlBtn}>
            <Ionicons name="play-back" size={24} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity onPress={togglePlay} style={styles.videoPlayBtn}>
            <Ionicons name={isPlaying ? 'pause' : 'play'} size={32} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity onPress={() => seek('forward')} style={styles.videoControlBtn}>
            <Ionicons name="play-forward" size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* Bottom seek bar */}
        <View style={styles.videoBottomBar}>
          <Text style={styles.videoTime}>{formatTime(position)}</Text>
          <View style={{ flex: 1, marginHorizontal: 8 }}>
            <ProgressBar
              progress={progress}
              height={4}
              colorStart="#6C63FF"
              colorEnd="#8B5CF6"
              animated={false}
            />
          </View>
          <Text style={styles.videoTime}>{formatTime(duration)}</Text>
        </View>
      </View>

      {!uri && (
        <View style={styles.videoPlaceholder}>
          <Ionicons name="videocam-outline" size={48} color={COLORS.textMuted} />
          <Text style={styles.videoPlaceholderText}>Video content</Text>
        </View>
      )}
    </View>
  );
}

// ── Quiz Component (inline for lesson screen) ─────────────────────────────────
function LessonQuiz({ onComplete }: { onComplete: (score: number) => void }) {
  const { QuizComponent } = require('../../../../components/student/QuizComponent');
  return <QuizComponent quiz={MOCK_QUIZ} onComplete={onComplete} />;
}

// ── Main Lesson Screen ────────────────────────────────────────────────────────
export default function LessonScreen() {
  const { lessonId } = useLocalSearchParams<{ lessonId: string }>();
  const router = useRouter();
  const { user } = useAuthStore();
  const {
    currentCourse,
    currentLesson,
    lessonProgress,
    getLessonById,
    setCurrentLesson,
    markLessonComplete,
  } = useCourseStore();
  const { addXP, checkAndUpdateStreak } = useGamificationStore();

  const [isCompleted, setIsCompleted] = useState(false);
  const [showXP, setShowXP] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const [noteText, setNoteText] = useState('');
  const [marking, setMarking] = useState(false);

  const xpPopAnim = useRef(new Animated.Value(0)).current;

  // Find the lesson
  const lesson: Lesson | undefined = lessonId
    ? getLessonById(lessonId) ?? currentLesson ?? undefined
    : currentLesson ?? undefined;

  // Check if already completed
  const progressEntry = lessonId ? lessonProgress[lessonId] : null;

  useEffect(() => {
    if (progressEntry?.completed) {
      setIsCompleted(true);
    }
  }, [progressEntry]);

  useEffect(() => {
    if (lesson) setCurrentLesson(lesson);
  }, [lesson, setCurrentLesson]);

  // Determine module/lesson context for breadcrumb + navigation
  const course = currentCourse;
  const allLessons: Lesson[] = course?.modules
    ? course.modules.flatMap((m) => m.lessons)
    : [];
  const currentIndex = allLessons.findIndex((l) => l.id === lessonId);
  const nextLesson = currentIndex >= 0 && currentIndex < allLessons.length - 1
    ? allLessons[currentIndex + 1]
    : null;
  const moduleProgress = course?.modules
    ? (() => {
        const mod = course.modules.find((m) => m.lessons.some((l) => l.id === lessonId));
        if (!mod) return null;
        const modLessons = mod.lessons;
        const modIdx = modLessons.findIndex((l) => l.id === lessonId);
        return { current: modIdx + 1, total: modLessons.length, title: mod.title };
      })()
    : null;

  const handleComplete = useCallback(async (score?: number) => {
    if (isCompleted || !user || !lessonId) return;
    try {
      setMarking(true);
      await markLessonComplete(lessonId, user.id, score);
      await addXP(XP_REWARDS.complete_lesson, 'complete_lesson');
      await checkAndUpdateStreak();
      setIsCompleted(true);
      setShowXP(true);
    } catch {
      Alert.alert('Error', 'Could not save progress. Please try again.');
    } finally {
      setMarking(false);
    }
  }, [isCompleted, user, lessonId, markLessonComplete, addXP, checkAndUpdateStreak]);

  const handleQuizComplete = useCallback((score: number) => {
    const xpType = score === 100 ? 'perfect_quiz' : 'complete_lesson';
    handleComplete(score);
  }, [handleComplete]);

  const goToNext = () => {
    if (nextLesson) {
      router.replace(`/student/courses/lesson/${nextLesson.id}` as any);
    } else {
      router.back();
    }
  };

  if (!lesson) {
    return (
      <SafeAreaView style={[styles.root, { alignItems: 'center', justifyContent: 'center' }]}>
        <ActivityIndicator color={COLORS.primary} size="large" />
        <Text style={[styles.textMuted, { marginTop: 16 }]}>Loading lesson...</Text>
      </SafeAreaView>
    );
  }

  const contentType: ContentType = lesson.content_type;

  return (
    <SafeAreaView style={styles.root} edges={['top']}>
      {/* ── Top Nav Bar ──────────────────────────────────────── */}
      <View style={styles.navBar}>
        <TouchableOpacity onPress={() => router.back()} style={styles.navBtn}>
          <Ionicons name="arrow-back" size={22} color={COLORS.text} />
        </TouchableOpacity>
        <View style={{ flex: 1, marginHorizontal: 10 }}>
          {/* Breadcrumb */}
          {course && (
            <Text style={styles.breadcrumb} numberOfLines={1}>
              {course.title}
              {moduleProgress ? ` › ${moduleProgress.title}` : ''}
            </Text>
          )}
          <Text style={styles.lessonTitle} numberOfLines={1}>{lesson.title}</Text>
        </View>
        <TouchableOpacity
          onPress={() => setNotesOpen((v) => !v)}
          style={[styles.navBtn, notesOpen && { backgroundColor: `${COLORS.primary}33` }]}
        >
          <Ionicons name="create-outline" size={22} color={notesOpen ? COLORS.primary : COLORS.text} />
        </TouchableOpacity>
      </View>

      {/* ── Module progress indicator ─────────────────────────── */}
      {moduleProgress && (
        <View style={styles.moduleProgressBar}>
          <View style={styles.moduleProgressInfo}>
            <Text style={styles.moduleProgressText}>
              Lesson {moduleProgress.current} of {moduleProgress.total}
            </Text>
            <Text style={styles.moduleProgressTitle}>{moduleProgress.title}</Text>
          </View>
          <View style={{ width: 120 }}>
            <ProgressBar
              progress={(moduleProgress.current / moduleProgress.total) * 100}
              height={4}
              colorStart="#6C63FF"
              colorEnd="#8B5CF6"
              animated
            />
          </View>
        </View>
      )}

      {/* ── Content Area ─────────────────────────────────────── */}
      <View style={styles.contentArea}>
        {/* Video */}
        {contentType === 'video' && (
          <VideoPlayer uri={lesson.content_url} onComplete={() => handleComplete()} />
        )}

        {/* Text */}
        {contentType === 'text' && (
          <RichTextContent content={MOCK_LESSON_TEXT} />
        )}

        {/* Quiz — render QuizComponent */}
        {contentType === 'quiz' && (
          <QuizContent onComplete={handleQuizComplete} />
        )}

        {/* Interactive */}
        {contentType === 'interactive' && lesson.content_url && (
          <WebView
            source={{ uri: lesson.content_url }}
            style={styles.webview}
            javaScriptEnabled
            domStorageEnabled
          />
        )}

        {contentType === 'interactive' && !lesson.content_url && (
          <View style={styles.interactivePlaceholder}>
            <Ionicons name="code-slash-outline" size={56} color={COLORS.textMuted} />
            <Text style={styles.placeholderTitle}>Interactive Content</Text>
            <Text style={styles.placeholderSub}>This lesson has an interactive coding exercise.</Text>
          </View>
        )}
      </View>

      {/* ── Notes Sidebar ────────────────────────────────────── */}
      {notesOpen && (
        <View style={styles.notesSidebar}>
          <View style={styles.notesHeader}>
            <Text style={styles.notesTitle}>My Notes</Text>
            <TouchableOpacity onPress={() => setNotesOpen(false)}>
              <Ionicons name="close" size={20} color={COLORS.textMuted} />
            </TouchableOpacity>
          </View>
          <TextInput
            style={styles.notesInput}
            multiline
            placeholder="Type your notes here..."
            placeholderTextColor={COLORS.textMuted}
            value={noteText}
            onChangeText={setNoteText}
            textAlignVertical="top"
          />
          <TouchableOpacity style={styles.saveNotesBtn}>
            <Text style={styles.saveNotesBtnText}>Save Notes</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* ── Bottom Bar ───────────────────────────────────────── */}
      {contentType !== 'quiz' && (
        <View style={styles.bottomBar}>
          {!isCompleted ? (
            <TouchableOpacity
              style={[styles.completeBtn, marking && { opacity: 0.7 }]}
              onPress={() => handleComplete()}
              disabled={marking}
              activeOpacity={0.85}
            >
              <LinearGradient
                colors={['#6C63FF', '#8B5CF6']}
                style={styles.completeBtnGrad}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              >
                {marking ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <>
                    <Ionicons name="checkmark-circle-outline" size={20} color="#fff" />
                    <Text style={styles.completeBtnText}>Mark as Complete</Text>
                  </>
                )}
              </LinearGradient>
            </TouchableOpacity>
          ) : (
            <View style={styles.completedBanner}>
              <Ionicons name="checkmark-circle" size={20} color={COLORS.accent} />
              <Text style={styles.completedText}>Lesson completed!</Text>
            </View>
          )}

          {nextLesson && (
            <TouchableOpacity style={styles.nextBtn} onPress={goToNext}>
              <Text style={styles.nextBtnText} numberOfLines={1}>
                Next: {nextLesson.title}
              </Text>
              <Ionicons name="arrow-forward" size={18} color={COLORS.primary} />
            </TouchableOpacity>
          )}
        </View>
      )}

      {/* ── XP Popup ─────────────────────────────────────────── */}
      {showXP && (
        <View style={styles.xpPopupWrapper} pointerEvents="none">
          <XPPopup xp={XP_REWARDS.complete_lesson} onDone={() => setShowXP(false)} />
        </View>
      )}
    </SafeAreaView>
  );
}

// ── Inline quiz wrapper to defer import ──────────────────────────────────────
import { QuizComponent } from '../../../../components/student/QuizComponent';
function QuizContent({ onComplete }: { onComplete: (score: number) => void }) {
  return <QuizComponent quiz={MOCK_QUIZ} onComplete={onComplete} />;
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  textMuted: {
    color: COLORS.textMuted,
    fontSize: 14,
  },

  // ── Nav Bar ──
  navBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  navBtn: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: COLORS.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  breadcrumb: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginBottom: 2,
  },
  lessonTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '700',
  },

  // ── Module Progress ──
  moduleProgressBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  moduleProgressInfo: {
    flex: 1,
  },
  moduleProgressText: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginBottom: 2,
  },
  moduleProgressTitle: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '600',
  },

  // ── Content ──
  contentArea: {
    flex: 1,
  },

  // ── Video ──
  videoContainer: {
    width: SCREEN_WIDTH,
    height: SCREEN_WIDTH * (9 / 16),
    backgroundColor: '#000',
    position: 'relative',
  },
  video: {
    ...StyleSheet.absoluteFillObject,
  },
  videoControls: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
  },
  videoTopBar: {
    flexDirection: 'row',
    padding: 12,
    alignItems: 'flex-start',
  },
  speedBtn: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 6,
  },
  speedBtnText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
  },
  speedMenu: {
    position: 'absolute',
    top: 48,
    right: 12,
    backgroundColor: COLORS.surface,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    overflow: 'hidden',
    zIndex: 10,
  },
  speedOption: {
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  speedOptionActive: {
    backgroundColor: COLORS.surfaceLight,
  },
  speedOptionText: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  videoCenterControls: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 24,
  },
  videoControlBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.5)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoPlayBtn: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(108,99,255,0.85)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoBottomBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingBottom: 12,
    gap: 8,
  },
  videoTime: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
    minWidth: 36,
  },
  videoPlaceholder: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0A0A1A',
    gap: 10,
  },
  videoPlaceholderText: {
    color: COLORS.textMuted,
    fontSize: 14,
  },

  // ── Text Content ──
  textContent: {
    flex: 1,
    padding: 20,
  },
  h2: {
    color: COLORS.text,
    fontSize: 22,
    fontWeight: '800',
    marginBottom: 12,
    marginTop: 8,
  },
  h3: {
    color: COLORS.text,
    fontSize: 17,
    fontWeight: '700',
    marginBottom: 10,
    marginTop: 16,
  },
  bodyText: {
    color: COLORS.textMuted,
    fontSize: 15,
    lineHeight: 24,
    marginBottom: 4,
  },
  bulletItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 6,
    paddingLeft: 4,
  },
  bullet: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.primary,
    marginTop: 9,
  },
  bulletItemText: {
    color: COLORS.textMuted,
    fontSize: 15,
    lineHeight: 24,
    flex: 1,
  },
  codeText: {
    color: COLORS.accent,
    fontSize: 13,
    fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace',
    backgroundColor: COLORS.surfaceLight,
    paddingHorizontal: 14,
    paddingVertical: 2,
    borderRadius: 4,
    marginBottom: 2,
  },
  quoteBlock: {
    borderLeftWidth: 3,
    borderLeftColor: COLORS.primary,
    paddingLeft: 14,
    marginVertical: 8,
    backgroundColor: `${COLORS.primary}11`,
    borderRadius: 4,
    paddingVertical: 10,
  },
  quoteText: {
    color: COLORS.text,
    fontSize: 14,
    lineHeight: 22,
    fontStyle: 'italic',
  },

  // ── Interactive ──
  webview: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  interactivePlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    padding: 40,
  },
  placeholderTitle: {
    color: COLORS.text,
    fontSize: 18,
    fontWeight: '700',
  },
  placeholderSub: {
    color: COLORS.textMuted,
    fontSize: 14,
    textAlign: 'center',
  },

  // ── Notes Sidebar ──
  notesSidebar: {
    position: 'absolute',
    right: 0,
    top: 0,
    bottom: 0,
    width: Math.min(SCREEN_WIDTH * 0.75, 320),
    backgroundColor: COLORS.surface,
    borderLeftWidth: 1,
    borderLeftColor: COLORS.border,
    padding: 16,
    zIndex: 50,
    elevation: 20,
    shadowColor: '#000',
    shadowOffset: { width: -4, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
  },
  notesHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  notesTitle: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: '700',
  },
  notesInput: {
    flex: 1,
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 12,
    padding: 14,
    color: COLORS.text,
    fontSize: 14,
    lineHeight: 22,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: 12,
  },
  saveNotesBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  saveNotesBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },

  // ── Bottom Bar ──
  bottomBar: {
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    backgroundColor: COLORS.surface,
    paddingHorizontal: 16,
    paddingVertical: 12,
    paddingBottom: 24,
    gap: 10,
  },
  completeBtn: {
    borderRadius: 14,
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  completeBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    height: 50,
    borderRadius: 14,
  },
  completeBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
  completedBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: `${COLORS.accent}18`,
    borderRadius: 12,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: `${COLORS.accent}40`,
  },
  completedText: {
    color: COLORS.accent,
    fontSize: 15,
    fontWeight: '700',
  },
  nextBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  nextBtnText: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
    marginRight: 8,
  },

  // ── XP Popup ──
  xpPopupWrapper: {
    position: 'absolute',
    bottom: 120,
    left: 0,
    right: 0,
    alignItems: 'center',
    pointerEvents: 'none',
  },
  xpPopup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: COLORS.accent,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 30,
    shadowColor: COLORS.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.6,
    shadowRadius: 12,
    elevation: 12,
  },
  xpPopupText: {
    color: '#000',
    fontSize: 18,
    fontWeight: '900',
  },
});
