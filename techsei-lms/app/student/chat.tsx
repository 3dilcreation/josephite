// ============================================================
// TechSei LMS — AI Tutor Chatbot Screen
// ============================================================
import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
} from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Animated,
  KeyboardAvoidingView,
  Platform,
  Dimensions,
  Modal,
  ScrollView,
  Pressable,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useChatStore, type ChatMessage } from '../../stores/chatStore';
import { useCourseStore } from '../../stores/courseStore';
import { useAuthStore } from '../../stores/authStore';
import { SUPPORTED_LANGUAGES, type SupportedLanguage } from '../../constants/i18n';
import type { LanguageCode } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─── Colors ───────────────────────────────────────────────────────────────────
const C = {
  bg: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  gold: '#FFD700',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textSec: '#9494B8',
  textMuted: '#5A5A7A',
  border: '#2A2A4A',
  error: '#FF4B4B',
  code: '#0D1117',
  overlay: 'rgba(0,0,0,0.8)',
};

const MAX_CHARS = 1000;

const SUGGESTED_QUESTIONS = [
  'Explain recursion with an example',
  'Help me with Python lists',
  'What is machine learning?',
  'Debug my code',
  'How do APIs work?',
  'Explain Big O notation',
];

// ─── Typing indicator ─────────────────────────────────────────────────────────
function TypingIndicator() {
  const dots = [
    useRef(new Animated.Value(0)).current,
    useRef(new Animated.Value(0)).current,
    useRef(new Animated.Value(0)).current,
  ];

  useEffect(() => {
    const anims = dots.map((dot, i) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(i * 160),
          Animated.timing(dot, { toValue: -6, duration: 300, useNativeDriver: true }),
          Animated.timing(dot, { toValue: 0, duration: 300, useNativeDriver: true }),
          Animated.delay(480),
        ])
      )
    );
    anims.forEach((a) => a.start());
    return () => anims.forEach((a) => a.stop());
  }, []);

  return (
    <View style={styles.typingBubble}>
      <BotAvatar size={28} />
      <View style={styles.typingDots}>
        {dots.map((dot, i) => (
          <Animated.View
            key={i}
            style={[styles.dot, { transform: [{ translateY: dot }] }]}
          />
        ))}
      </View>
    </View>
  );
}

// ─── Bot avatar ───────────────────────────────────────────────────────────────
function BotAvatar({ size = 32 }: { size?: number }) {
  return (
    <LinearGradient
      colors={[C.primary, '#8B5CF6']}
      style={[styles.botAvatar, { width: size, height: size, borderRadius: size / 2 }]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <Ionicons name="sparkles" size={size * 0.5} color={C.text} />
    </LinearGradient>
  );
}

// ─── Code block renderer ──────────────────────────────────────────────────────
function renderMessageContent(content: string) {
  // Split on triple-backtick code blocks
  const parts = content.split(/(```[\s\S]*?```)/g);

  return parts.map((part, i) => {
    if (part.startsWith('```') && part.endsWith('```')) {
      const lines = part.slice(3, -3).split('\n');
      const lang = lines[0]?.trim() || '';
      const code = lines.slice(lang ? 1 : 0).join('\n');
      return (
        <View key={i} style={styles.codeBlock}>
          {lang ? <Text style={styles.codeLang}>{lang}</Text> : null}
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <Text style={styles.codeText}>{code.trim()}</Text>
          </ScrollView>
        </View>
      );
    }
    return (
      <Text key={i} style={styles.msgText}>
        {part}
      </Text>
    );
  });
}

// ─── Single message bubble ────────────────────────────────────────────────────
function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user';
  const timeStr = new Date(msg.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <View style={[styles.msgRow, isUser ? styles.msgRowUser : styles.msgRowBot]}>
      {!isUser && <BotAvatar size={30} />}
      <View style={[styles.bubble, isUser ? styles.bubbleUser : styles.bubbleBot]}>
        {renderMessageContent(msg.content)}
        <Text style={[styles.msgTime, isUser ? styles.msgTimeUser : styles.msgTimeBot]}>
          {timeStr}
        </Text>
      </View>
    </View>
  );
}

// ─── Language picker modal ────────────────────────────────────────────────────
function LanguagePickerModal({
  visible,
  current,
  onSelect,
  onClose,
}: {
  visible: boolean;
  current: LanguageCode;
  onSelect: (code: LanguageCode) => void;
  onClose: () => void;
}) {
  const [search, setSearch] = useState('');
  const filtered = SUPPORTED_LANGUAGES.filter(
    (l) =>
      l.name.toLowerCase().includes(search.toLowerCase()) ||
      l.nativeName.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.langModalBg}>
        <View style={styles.langModalCard}>
          <View style={styles.langModalHeader}>
            <Text style={styles.langModalTitle}>Select Language</Text>
            <TouchableOpacity onPress={onClose}>
              <Ionicons name="close" size={24} color={C.textSec} />
            </TouchableOpacity>
          </View>

          <View style={styles.langSearch}>
            <Ionicons name="search-outline" size={16} color={C.textMuted} />
            <TextInput
              style={styles.langSearchInput}
              placeholder="Search language..."
              placeholderTextColor={C.textMuted}
              value={search}
              onChangeText={setSearch}
            />
          </View>

          <FlatList
            data={filtered}
            keyExtractor={(l) => l.code}
            numColumns={2}
            renderItem={({ item }) => {
              const active = item.code === current;
              return (
                <TouchableOpacity
                  style={[styles.langItem, active && styles.langItemActive]}
                  onPress={() => {
                    onSelect(item.code as LanguageCode);
                    onClose();
                  }}
                >
                  <Text style={styles.langFlag}>{item.flag}</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={[styles.langName, active && { color: C.primary }]}>
                      {item.name}
                    </Text>
                    <Text style={styles.langNative}>{item.nativeName}</Text>
                  </View>
                  {active && <Ionicons name="checkmark-circle" size={16} color={C.primary} />}
                </TouchableOpacity>
              );
            }}
            contentContainerStyle={{ paddingBottom: 24 }}
          />
        </View>
      </View>
    </Modal>
  );
}

// ─── Main Screen ──────────────────────────────────────────────────────────────
export default function ChatbotScreen() {
  const { messages, isTyping, selectedLanguage, sendMessage, clearChat, setLanguage } = useChatStore();
  const { currentCourse } = useCourseStore();
  const { user } = useAuthStore();

  const [input, setInput] = useState('');
  const [langPickerVisible, setLangPickerVisible] = useState(false);
  const [menuVisible, setMenuVisible] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  const sendBtnScale = useRef(new Animated.Value(1)).current;

  const currentLang = SUPPORTED_LANGUAGES.find((l) => l.code === selectedLanguage);

  const studentContext = currentCourse
    ? { courseTitle: currentCourse.title, courseCategory: currentCourse.category }
    : undefined;

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isTyping) return;

    setInput('');
    Animated.sequence([
      Animated.timing(sendBtnScale, { toValue: 0.85, duration: 100, useNativeDriver: true }),
      Animated.spring(sendBtnScale, { toValue: 1, useNativeDriver: true }),
    ]).start();

    await sendMessage(text, studentContext as any);
  }, [input, isTyping, sendMessage, studentContext, sendBtnScale]);

  const handleSuggestion = useCallback((q: string) => {
    setInput(q);
  }, []);

  const isEmpty = messages.length === 0;

  const listData: (ChatMessage | 'typing')[] = isTyping
    ? [...messages, 'typing']
    : messages;

  const renderItem = useCallback(
    ({ item }: { item: ChatMessage | 'typing' }) => {
      if (item === 'typing') return <TypingIndicator />;
      return <MessageBubble msg={item} />;
    },
    []
  );

  const keyExtractor = useCallback(
    (item: ChatMessage | 'typing', idx: number) =>
      item === 'typing' ? 'typing' : item.id,
    []
  );

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <BotAvatar size={40} />
          <View>
            <Text style={styles.headerTitle}>TechSei AI Tutor</Text>
            <View style={styles.headerStatusRow}>
              <View style={styles.onlineDot} />
              <Text style={styles.headerStatus}>Online · Always ready</Text>
            </View>
          </View>
        </View>

        <View style={styles.headerRight}>
          {/* Language selector */}
          <TouchableOpacity
            style={styles.langBtn}
            onPress={() => setLangPickerVisible(true)}
          >
            <Text style={{ fontSize: 16 }}>{currentLang?.flag ?? '🌐'}</Text>
            <Text style={styles.langBtnText}>{currentLang?.name ?? 'English'}</Text>
            <Ionicons name="chevron-down" size={14} color={C.textSec} />
          </TouchableOpacity>

          {/* Menu */}
          <TouchableOpacity
            style={styles.menuBtn}
            onPress={() => setMenuVisible(!menuVisible)}
          >
            <Ionicons name="ellipsis-vertical" size={20} color={C.textSec} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Drop-down menu */}
      {menuVisible && (
        <View style={styles.dropMenu}>
          <TouchableOpacity
            style={styles.dropMenuItem}
            onPress={() => { clearChat(); setMenuVisible(false); }}
          >
            <Ionicons name="trash-outline" size={18} color={C.error} />
            <Text style={[styles.dropMenuText, { color: C.error }]}>Clear Chat</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Language response indicator */}
      <View style={styles.langIndicator}>
        <Text style={styles.langIndicatorText}>
          Responding in: {currentLang?.flag} {currentLang?.name}
          {currentCourse ? `  ·  Course: ${currentCourse.title}` : ''}
        </Text>
      </View>

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={0}
      >
        {/* Suggested questions (shown when empty) */}
        {isEmpty && (
          <View style={styles.suggestionsWrapper}>
            <Text style={styles.suggestionsTitle}>Ask me anything about your studies</Text>
            <View style={styles.suggestionsGrid}>
              {SUGGESTED_QUESTIONS.map((q) => (
                <TouchableOpacity
                  key={q}
                  style={styles.suggestionChip}
                  onPress={() => handleSuggestion(q)}
                >
                  <Ionicons name="bulb-outline" size={14} color={C.primary} />
                  <Text style={styles.suggestionText}>{q}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* Message list */}
        <FlatList
          ref={flatListRef}
          data={listData}
          renderItem={renderItem}
          keyExtractor={keyExtractor}
          contentContainerStyle={styles.messageList}
          showsVerticalScrollIndicator={false}
          onContentSizeChange={() =>
            flatListRef.current?.scrollToEnd({ animated: true })
          }
        />

        {/* Input bar */}
        <View style={styles.inputBar}>
          <View style={styles.inputRow}>
            <TextInput
              style={styles.textInput}
              value={input}
              onChangeText={(t) => setInput(t.slice(0, MAX_CHARS))}
              placeholder="Ask your AI tutor..."
              placeholderTextColor={C.textMuted}
              multiline
              maxLength={MAX_CHARS}
              returnKeyType="default"
            />

            {/* Mic button */}
            <TouchableOpacity style={styles.micBtn}>
              <Ionicons name="mic-outline" size={20} color={C.textSec} />
            </TouchableOpacity>

            {/* Send button */}
            <Animated.View style={{ transform: [{ scale: sendBtnScale }] }}>
              <TouchableOpacity
                style={[styles.sendBtn, (!input.trim() || isTyping) && styles.sendBtnDisabled]}
                onPress={handleSend}
                disabled={!input.trim() || isTyping}
              >
                <LinearGradient
                  colors={[C.primary, '#8B5CF6']}
                  style={styles.sendBtnGrad}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                >
                  <Ionicons name="send" size={18} color={C.text} />
                </LinearGradient>
              </TouchableOpacity>
            </Animated.View>
          </View>

          {/* Char count */}
          <Text style={styles.charCount}>
            {input.length}/{MAX_CHARS}
          </Text>
        </View>
      </KeyboardAvoidingView>

      {/* Language Picker Modal */}
      <LanguagePickerModal
        visible={langPickerVisible}
        current={selectedLanguage}
        onSelect={(code) => setLanguage(code)}
        onClose={() => setLangPickerVisible(false)}
      />
    </SafeAreaView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  headerTitle: { fontSize: 17, fontWeight: '800', color: C.text },
  headerStatusRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2 },
  onlineDot: {
    width: 7, height: 7, borderRadius: 4,
    backgroundColor: C.accent,
  },
  headerStatus: { fontSize: 11, color: C.textSec },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  langBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: C.surfaceLight,
    paddingHorizontal: 10, paddingVertical: 6,
    borderRadius: 12,
  },
  langBtnText: { fontSize: 12, color: C.textSec, fontWeight: '600' },
  menuBtn: {
    width: 34, height: 34, borderRadius: 17,
    backgroundColor: C.surfaceLight,
    alignItems: 'center', justifyContent: 'center',
  },

  // Drop menu
  dropMenu: {
    position: 'absolute',
    top: 70, right: 16,
    backgroundColor: C.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: C.border,
    zIndex: 100,
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    minWidth: 160,
  },
  dropMenuItem: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    paddingHorizontal: 16, paddingVertical: 12,
  },
  dropMenuText: { fontSize: 14, fontWeight: '600' },

  // Lang indicator
  langIndicator: {
    paddingHorizontal: 16, paddingVertical: 6,
    backgroundColor: 'rgba(108,99,255,0.08)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(108,99,255,0.15)',
  },
  langIndicatorText: { fontSize: 11, color: C.textSec },

  // Suggestions
  suggestionsWrapper: {
    paddingHorizontal: 16,
    paddingTop: 24,
    paddingBottom: 12,
  },
  suggestionsTitle: {
    fontSize: 15, fontWeight: '700', color: C.textSec,
    marginBottom: 12, textAlign: 'center',
  },
  suggestionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'center',
  },
  suggestionChip: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: C.surfaceLight,
    borderRadius: 20,
    paddingHorizontal: 14, paddingVertical: 9,
    borderWidth: 1,
    borderColor: 'rgba(108,99,255,0.25)',
  },
  suggestionText: { fontSize: 13, color: C.text, fontWeight: '600' },

  // Messages
  messageList: {
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 8,
    flexGrow: 1,
  },
  msgRow: {
    marginBottom: 14,
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
  },
  msgRowUser: { justifyContent: 'flex-end' },
  msgRowBot: { justifyContent: 'flex-start' },
  bubble: {
    maxWidth: SCREEN_WIDTH * 0.75,
    borderRadius: 18,
    padding: 12,
  },
  bubbleUser: {
    backgroundColor: C.primary,
    borderBottomRightRadius: 4,
  },
  bubbleBot: {
    backgroundColor: C.surface,
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: C.border,
  },
  msgText: {
    fontSize: 14, lineHeight: 20,
    color: C.text,
  },
  msgTime: {
    fontSize: 10, marginTop: 4,
  },
  msgTimeUser: { color: 'rgba(255,255,255,0.55)', textAlign: 'right' },
  msgTimeBot: { color: C.textMuted },

  // Code block
  codeBlock: {
    backgroundColor: C.code,
    borderRadius: 10,
    padding: 12,
    marginVertical: 6,
  },
  codeLang: {
    fontSize: 10, color: C.textMuted,
    fontWeight: '700', marginBottom: 6,
    textTransform: 'uppercase', letterSpacing: 1,
  },
  codeText: {
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 12, lineHeight: 18,
    color: '#E6EDF3',
  },

  // Typing
  typingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 14,
  },
  typingDots: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.surface,
    borderRadius: 18,
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 5,
    borderWidth: 1,
    borderColor: C.border,
  },
  dot: {
    width: 7, height: 7, borderRadius: 4,
    backgroundColor: C.textSec,
  },

  // Bot avatar
  botAvatar: { alignItems: 'center', justifyContent: 'center' },

  // Input bar
  inputBar: {
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: C.border,
    backgroundColor: C.surface,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
  },
  textInput: {
    flex: 1,
    backgroundColor: C.surfaceLight,
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 10,
    fontSize: 14,
    color: C.text,
    maxHeight: 110,
    borderWidth: 1,
    borderColor: C.border,
  },
  micBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: C.surfaceLight,
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: C.border,
  },
  sendBtn: { borderRadius: 20, overflow: 'hidden' },
  sendBtnDisabled: { opacity: 0.4 },
  sendBtnGrad: {
    width: 40, height: 40,
    alignItems: 'center', justifyContent: 'center',
  },
  charCount: {
    fontSize: 10, color: C.textMuted,
    textAlign: 'right',
    marginTop: 4,
    paddingRight: 4,
  },

  // Language picker modal
  langModalBg: {
    flex: 1,
    backgroundColor: C.overlay,
    justifyContent: 'flex-end',
  },
  langModalCard: {
    backgroundColor: C.surface,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    maxHeight: '80%',
    paddingTop: 20,
  },
  langModalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    marginBottom: 16,
  },
  langModalTitle: { fontSize: 18, fontWeight: '800', color: C.text },
  langSearch: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: C.surfaceLight,
    borderRadius: 14,
    paddingHorizontal: 14, paddingVertical: 10,
    marginHorizontal: 16,
    marginBottom: 14,
    borderWidth: 1, borderColor: C.border,
  },
  langSearchInput: {
    flex: 1, fontSize: 14, color: C.text,
  },
  langItem: {
    flex: 1,
    flexDirection: 'row', alignItems: 'center', gap: 10,
    backgroundColor: C.surfaceLight,
    borderRadius: 14,
    padding: 12,
    margin: 4,
    marginHorizontal: 8,
    borderWidth: 1, borderColor: C.border,
  },
  langItemActive: {
    borderColor: C.primary,
    backgroundColor: 'rgba(108,99,255,0.12)',
  },
  langFlag: { fontSize: 22 },
  langName: { fontSize: 13, fontWeight: '700', color: C.text },
  langNative: { fontSize: 11, color: C.textSec },
});
