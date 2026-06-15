// ============================================================
// TechSei LMS — Full Quiz Component
// ============================================================
import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  ScrollView,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { ProgressBar } from '../common/ProgressBar';
import { XP_REWARDS } from '../../stores/gamificationStore';
import type { Quiz, QuizQuestion } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

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
  success: '#43E97B',
};

type AnswerState = 'idle' | 'correct' | 'wrong';

interface QuizComponentProps {
  quiz: Quiz;
  onComplete: (score: number) => void;
}

// ── Score Result Screen ────────────────────────────────────────────────────────
function ScoreScreen({
  score,
  total,
  passing_score,
  xpEarned,
  onRetry,
  onContinue,
}: {
  score: number;
  total: number;
  passing_score: number;
  xpEarned: number;
  onRetry: () => void;
  onContinue: () => void;
}) {
  const percentage = Math.round((score / total) * 100);
  const passed = percentage >= passing_score;

  const scale = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(scale, { toValue: 1, useNativeDriver: true, tension: 80, friction: 7 }),
      Animated.timing(opacity, { toValue: 1, duration: 400, useNativeDriver: true }),
    ]).start();
  }, [scale, opacity]);

  return (
    <ScrollView
      contentContainerStyle={styles.scoreScreen}
      showsVerticalScrollIndicator={false}
    >
      <Animated.View style={[styles.scoreContent, { opacity, transform: [{ scale }] }]}>
        {/* Result Icon */}
        <View style={styles.scoreIconWrapper}>
          <LinearGradient
            colors={passed ? ['#43E97B', '#38F9D7'] : ['#FF6B6B', '#FF4B4B']}
            style={styles.scoreIcon}
          >
            <Text style={styles.scoreEmoji}>{passed ? '🏆' : '😔'}</Text>
          </LinearGradient>
        </View>

        <Text style={styles.scoreHeading}>{passed ? 'Excellent Work!' : 'Keep Practicing!'}</Text>
        <Text style={styles.scoreSubHeading}>
          {passed
            ? `You passed with a score of ${percentage}%!`
            : `You scored ${percentage}%. You need ${passing_score}% to pass.`}
        </Text>

        {/* Score Circle */}
        <View style={[styles.scoreCircle, { borderColor: passed ? COLORS.success : COLORS.error }]}>
          <Text style={[styles.scorePercent, { color: passed ? COLORS.success : COLORS.error }]}>
            {percentage}%
          </Text>
          <Text style={styles.scoreLabel}>
            {score}/{total} correct
          </Text>
        </View>

        {/* Stats Row */}
        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Ionicons name="checkmark-circle" size={22} color={COLORS.success} />
            <Text style={styles.statValue}>{score}</Text>
            <Text style={styles.statLabel}>Correct</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBox}>
            <Ionicons name="close-circle" size={22} color={COLORS.error} />
            <Text style={styles.statValue}>{total - score}</Text>
            <Text style={styles.statLabel}>Incorrect</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBox}>
            <Ionicons name="flash" size={22} color={COLORS.accent} />
            <Text style={styles.statValue}>+{xpEarned}</Text>
            <Text style={styles.statLabel}>XP Earned</Text>
          </View>
        </View>

        {/* Pass indicator */}
        <View style={[
          styles.passIndicator,
          { backgroundColor: passed ? `${COLORS.success}18` : `${COLORS.error}18` },
          { borderColor: passed ? `${COLORS.success}40` : `${COLORS.error}40` },
        ]}>
          <Ionicons
            name={passed ? 'checkmark-circle' : 'close-circle'}
            size={18}
            color={passed ? COLORS.success : COLORS.error}
          />
          <Text style={[styles.passText, { color: passed ? COLORS.success : COLORS.error }]}>
            {passed ? `Passed — required ${passing_score}%` : `Not passed — required ${passing_score}%`}
          </Text>
        </View>

        {/* Action Buttons */}
        <View style={styles.scoreActions}>
          <TouchableOpacity style={styles.retryBtn} onPress={onRetry} activeOpacity={0.8}>
            <Ionicons name="refresh" size={18} color={COLORS.primary} />
            <Text style={styles.retryBtnText}>Try Again</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.continueBtn} onPress={onContinue} activeOpacity={0.85}>
            <LinearGradient
              colors={['#6C63FF', '#8B5CF6']}
              style={styles.continueBtnGrad}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Text style={styles.continueBtnText}>Continue</Text>
              <Ionicons name="arrow-forward" size={18} color="#fff" />
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </Animated.View>
    </ScrollView>
  );
}

// ── Single Question View ──────────────────────────────────────────────────────
function QuestionView({
  question,
  questionIndex,
  totalQuestions,
  onAnswer,
}: {
  question: QuizQuestion;
  questionIndex: number;
  totalQuestions: number;
  onAnswer: (selectedIndex: number, isCorrect: boolean) => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const slideAnim = useRef(new Animated.Value(30)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const explanationAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Entrance animation
    Animated.parallel([
      Animated.timing(slideAnim, { toValue: 0, duration: 350, useNativeDriver: true }),
      Animated.timing(fadeAnim, { toValue: 1, duration: 350, useNativeDriver: true }),
    ]).start();
  }, [questionIndex, slideAnim, fadeAnim]);

  const handleSelect = (index: number) => {
    if (answered) return;
    const isCorrect = index === question.correct_index;
    setSelectedIndex(index);
    setAnswered(true);

    // Show explanation with animation
    setTimeout(() => {
      setShowExplanation(true);
      Animated.timing(explanationAnim, { toValue: 1, duration: 300, useNativeDriver: true }).start();
    }, 400);
  };

  const getOptionStyle = (index: number) => {
    if (!answered) return styles.option;
    if (index === question.correct_index) return [styles.option, styles.optionCorrect];
    if (index === selectedIndex && index !== question.correct_index) return [styles.option, styles.optionWrong];
    return [styles.option, styles.optionDimmed];
  };

  const getOptionTextColor = (index: number): string => {
    if (!answered) return COLORS.text;
    if (index === question.correct_index) return COLORS.success;
    if (index === selectedIndex) return COLORS.error;
    return COLORS.textMuted;
  };

  const getOptionIcon = (index: number): React.ComponentProps<typeof Ionicons>['name'] | null => {
    if (!answered) return null;
    if (index === question.correct_index) return 'checkmark-circle';
    if (index === selectedIndex) return 'close-circle';
    return null;
  };

  const getOptionIconColor = (index: number): string => {
    if (index === question.correct_index) return COLORS.success;
    return COLORS.error;
  };

  return (
    <Animated.View
      style={[styles.questionContainer, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}
    >
      {/* Progress */}
      <View style={styles.questionProgress}>
        <Text style={styles.questionCounter}>
          Question {questionIndex + 1} of {totalQuestions}
        </Text>
        <View style={{ width: 120 }}>
          <ProgressBar
            progress={((questionIndex + 1) / totalQuestions) * 100}
            height={5}
            colorStart="#6C63FF"
            colorEnd="#8B5CF6"
            animated
          />
        </View>
      </View>

      {/* Question dots */}
      <View style={styles.dotRow}>
        {Array.from({ length: totalQuestions }).map((_, i) => (
          <View
            key={i}
            style={[
              styles.dot,
              i === questionIndex && styles.dotActive,
              i < questionIndex && styles.dotDone,
            ]}
          />
        ))}
      </View>

      {/* Question text */}
      <View style={styles.questionBox}>
        <View style={styles.questionNumberBadge}>
          <Text style={styles.questionNumberText}>Q{questionIndex + 1}</Text>
        </View>
        <Text style={styles.questionText}>{question.question}</Text>
      </View>

      {/* Options */}
      <View style={styles.optionsList}>
        {question.options.map((option, index) => {
          const icon = getOptionIcon(index);
          return (
            <TouchableOpacity
              key={index}
              style={getOptionStyle(index)}
              onPress={() => handleSelect(index)}
              activeOpacity={answered ? 1 : 0.8}
              disabled={answered}
            >
              <View style={[styles.optionLetter, answered && index === question.correct_index && styles.optionLetterCorrect, answered && index === selectedIndex && index !== question.correct_index && styles.optionLetterWrong]}>
                <Text style={[styles.optionLetterText, answered && (index === question.correct_index || index === selectedIndex) && { color: '#fff' }]}>
                  {String.fromCharCode(65 + index)}
                </Text>
              </View>
              <Text style={[styles.optionText, { color: getOptionTextColor(index) }]} numberOfLines={3}>
                {option}
              </Text>
              {icon && (
                <Ionicons name={icon} size={20} color={getOptionIconColor(index)} style={{ marginLeft: 8 }} />
              )}
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Explanation */}
      {showExplanation && (
        <Animated.View style={[styles.explanation, { opacity: explanationAnim }]}>
          <View style={styles.explanationHeader}>
            <Ionicons name="information-circle" size={18} color={COLORS.primary} />
            <Text style={styles.explanationTitle}>Explanation</Text>
          </View>
          <Text style={styles.explanationText}>{question.explanation}</Text>
        </Animated.View>
      )}

      {/* Next button (only visible after answering) */}
      {answered && (
        <TouchableOpacity
          style={styles.nextBtn}
          onPress={() => onAnswer(selectedIndex!, selectedIndex === question.correct_index)}
          activeOpacity={0.85}
        >
          <LinearGradient
            colors={['#6C63FF', '#8B5CF6']}
            style={styles.nextBtnGrad}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.nextBtnText}>
              {questionIndex < totalQuestions - 1 ? 'Next Question' : 'See Results'}
            </Text>
            <Ionicons name="arrow-forward" size={18} color="#fff" />
          </LinearGradient>
        </TouchableOpacity>
      )}
    </Animated.View>
  );
}

// ── Main Quiz Component ───────────────────────────────────────────────────────
export function QuizComponent({ quiz, onComplete }: QuizComponentProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<number[]>([]);
  const [finished, setFinished] = useState(false);

  const percentage = quiz.questions.length > 0
    ? Math.round((score / quiz.questions.length) * 100)
    : 0;
  const xpEarned = percentage === 100
    ? XP_REWARDS.perfect_quiz
    : XP_REWARDS.complete_lesson;

  const handleAnswer = useCallback((selectedIndex: number, isCorrect: boolean) => {
    const newAnswers = [...answers, selectedIndex];
    const newScore = isCorrect ? score + 1 : score;
    setAnswers(newAnswers);
    setScore(newScore);

    if (currentIndex < quiz.questions.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      // Quiz complete
      const finalPct = Math.round((newScore / quiz.questions.length) * 100);
      setFinished(true);
      onComplete(finalPct);
    }
  }, [answers, score, currentIndex, quiz.questions.length, onComplete]);

  const handleRetry = () => {
    setCurrentIndex(0);
    setScore(0);
    setAnswers([]);
    setFinished(false);
  };

  const handleContinue = () => {
    onComplete(percentage);
  };

  if (finished) {
    return (
      <ScoreScreen
        score={score}
        total={quiz.questions.length}
        passing_score={quiz.passing_score}
        xpEarned={xpEarned}
        onRetry={handleRetry}
        onContinue={handleContinue}
      />
    );
  }

  const question = quiz.questions[currentIndex];
  if (!question) return null;

  return (
    <ScrollView
      key={currentIndex}
      style={styles.root}
      contentContainerStyle={styles.scrollContent}
      showsVerticalScrollIndicator={false}
    >
      {/* Quiz header */}
      <View style={styles.quizHeader}>
        <LinearGradient colors={['#6C63FF22', '#6C63FF00']} style={styles.quizHeaderGrad}>
          <Text style={styles.quizTitle}>{quiz.title}</Text>
          <View style={styles.quizMetaRow}>
            <View style={styles.quizMetaChip}>
              <Ionicons name="help-circle-outline" size={14} color={COLORS.primary} />
              <Text style={styles.quizMetaText}>{quiz.questions.length} questions</Text>
            </View>
            <View style={styles.quizMetaChip}>
              <Ionicons name="ribbon-outline" size={14} color={COLORS.warning} />
              <Text style={styles.quizMetaText}>Pass: {quiz.passing_score}%</Text>
            </View>
          </View>
        </LinearGradient>
      </View>

      <QuestionView
        key={currentIndex}
        question={question}
        questionIndex={currentIndex}
        totalQuestions={quiz.questions.length}
        onAnswer={handleAnswer}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollContent: {
    paddingBottom: 40,
  },

  // ── Quiz Header ──
  quizHeader: {
    borderRadius: 0,
    overflow: 'hidden',
  },
  quizHeaderGrad: {
    padding: 20,
    paddingTop: 16,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  quizTitle: {
    color: COLORS.text,
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 10,
  },
  quizMetaRow: {
    flexDirection: 'row',
    gap: 10,
  },
  quizMetaChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: COLORS.surface,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  quizMetaText: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '600',
  },

  // ── Question ──
  questionContainer: {
    padding: 20,
  },
  questionProgress: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  questionCounter: {
    color: COLORS.textMuted,
    fontSize: 13,
    fontWeight: '600',
  },
  dotRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 20,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.surfaceLight,
  },
  dotActive: {
    backgroundColor: COLORS.primary,
    width: 20,
  },
  dotDone: {
    backgroundColor: COLORS.accent,
  },
  questionBox: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    gap: 12,
  },
  questionNumberBadge: {
    backgroundColor: `${COLORS.primary}22`,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: `${COLORS.primary}44`,
  },
  questionNumberText: {
    color: COLORS.primary,
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  questionText: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: '600',
    lineHeight: 24,
  },

  // ── Options ──
  optionsList: {
    gap: 10,
    marginBottom: 16,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    padding: 14,
    borderWidth: 2,
    borderColor: COLORS.border,
    gap: 12,
  },
  optionCorrect: {
    borderColor: COLORS.success,
    backgroundColor: `${COLORS.success}12`,
  },
  optionWrong: {
    borderColor: COLORS.error,
    backgroundColor: `${COLORS.error}12`,
  },
  optionDimmed: {
    opacity: 0.5,
  },
  optionLetter: {
    width: 32,
    height: 32,
    borderRadius: 10,
    backgroundColor: COLORS.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  optionLetterCorrect: {
    backgroundColor: COLORS.success,
    borderColor: COLORS.success,
  },
  optionLetterWrong: {
    backgroundColor: COLORS.error,
    borderColor: COLORS.error,
  },
  optionLetterText: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '800',
  },
  optionText: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
    lineHeight: 20,
  },

  // ── Explanation ──
  explanation: {
    backgroundColor: `${COLORS.primary}12`,
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: `${COLORS.primary}30`,
  },
  explanationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  explanationTitle: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: '700',
  },
  explanationText: {
    color: COLORS.textMuted,
    fontSize: 14,
    lineHeight: 22,
  },

  // ── Next Button ──
  nextBtn: {
    borderRadius: 14,
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  nextBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    height: 52,
    borderRadius: 14,
  },
  nextBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },

  // ── Score Screen ──
  scoreScreen: {
    flexGrow: 1,
    padding: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreContent: {
    width: '100%',
    alignItems: 'center',
  },
  scoreIconWrapper: {
    marginBottom: 20,
  },
  scoreIcon: {
    width: 100,
    height: 100,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#43E97B',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 12,
  },
  scoreEmoji: {
    fontSize: 48,
  },
  scoreHeading: {
    color: COLORS.text,
    fontSize: 26,
    fontWeight: '900',
    marginBottom: 8,
    textAlign: 'center',
  },
  scoreSubHeading: {
    color: COLORS.textMuted,
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 28,
    paddingHorizontal: 20,
  },
  scoreCircle: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 4,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.surface,
    marginBottom: 28,
  },
  scorePercent: {
    fontSize: 38,
    fontWeight: '900',
  },
  scoreLabel: {
    color: COLORS.textMuted,
    fontSize: 13,
    marginTop: 2,
  },

  // ── Stats ──
  statsRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: 16,
    width: '100%',
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
    gap: 6,
  },
  statValue: {
    color: COLORS.text,
    fontSize: 20,
    fontWeight: '800',
  },
  statDivider: {
    width: 1,
    backgroundColor: COLORS.border,
    marginHorizontal: 8,
  },
  statLabel: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '600',
  },

  // ── Pass indicator ──
  passIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1,
    width: '100%',
    marginBottom: 28,
  },
  passText: {
    fontSize: 13,
    fontWeight: '600',
  },

  // ── Score Actions ──
  scoreActions: {
    flexDirection: 'row',
    gap: 12,
    width: '100%',
  },
  retryBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    borderWidth: 2,
    borderColor: COLORS.primary,
    borderRadius: 14,
    height: 52,
    backgroundColor: `${COLORS.primary}12`,
  },
  retryBtnText: {
    color: COLORS.primary,
    fontSize: 15,
    fontWeight: '700',
  },
  continueBtn: {
    flex: 2,
    borderRadius: 14,
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  continueBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    height: 52,
    borderRadius: 14,
  },
  continueBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
});

export default QuizComponent;
