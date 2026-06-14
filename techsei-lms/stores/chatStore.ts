import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { sendChatMessage } from '../lib/claude';
import type { StudentContext } from '../lib/claude';
import type { LanguageCode, ChatMessage as AppChatMessage } from '../types';

// ─── Types ───────────────────────────────────────────────────────────────────

/**
 * A single turn in the in-app chat UI.
 * Intentionally separate from AppChatMessage (which is the Supabase DB shape)
 * so the chat screen never needs to wait for a round-trip to persist messages.
 */
export interface ChatMessage {
  /** Locally-unique ID (timestamp + random suffix). */
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** ISO 8601 timestamp — used for display and for ordering. */
  timestamp: string;
  language: LanguageCode;
}

// ─── Constants ────────────────────────────────────────────────────────────────

/** Maximum number of messages retained in local state and AsyncStorage. */
const MAX_MESSAGES = 50;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function nowISO(): string {
  return new Date().toISOString();
}

/**
 * Convert our local ChatMessage[] into the AppChatMessage[] shape that
 * lib/claude.ts expects (it reads .message and .response fields).
 *
 * The claude helper builds conversation history from completed user/assistant
 * pairs.  We reconstruct those pairs here by zipping consecutive turns.
 */
function toAppChatMessages(messages: ChatMessage[]): AppChatMessage[] {
  const result: AppChatMessage[] = [];

  for (let i = 0; i < messages.length - 1; i++) {
    const userTurn = messages[i];
    const assistantTurn = messages[i + 1];

    if (userTurn.role === 'user' && assistantTurn?.role === 'assistant') {
      result.push({
        id: userTurn.id,
        student_id: '',         // not needed by claude.ts
        message: userTurn.content,
        response: assistantTurn.content,
        language: userTurn.language as LanguageCode,
        created_at: userTurn.timestamp,
        is_loading: false,
      } satisfies AppChatMessage);
      i++; // advance past the assistant turn we just consumed
    }
  }

  return result;
}

// ─── State & Actions ─────────────────────────────────────────────────────────

interface ChatState {
  messages: ChatMessage[];
  isTyping: boolean;
  selectedLanguage: LanguageCode;
}

interface ChatActions {
  /**
   * Append a user message, then fetch and append the assistant's reply.
   * @param content        The user's message text.
   * @param studentContext Optional learner metadata forwarded to the Claude API.
   */
  sendMessage: (content: string, studentContext?: StudentContext) => Promise<void>;
  /** Wipe all messages from state and AsyncStorage. */
  clearChat: () => void;
  /** Switch the response language for future messages. */
  setLanguage: (code: LanguageCode) => void;
}

type ChatStore = ChatState & ChatActions;

// ─── Store ───────────────────────────────────────────────────────────────────

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // ── Initial state ──────────────────────────────────────────────────────
      messages: [],
      isTyping: false,
      selectedLanguage: 'en',

      // ── Actions ────────────────────────────────────────────────────────────

      sendMessage: async (content: string, studentContext?: StudentContext) => {
        const { selectedLanguage, messages } = get();

        // 1. Append the user message immediately so the UI responds at once.
        const userMessage: ChatMessage = {
          id: generateId(),
          role: 'user',
          content: content.trim(),
          timestamp: nowISO(),
          language: selectedLanguage,
        };

        const withUser = [...messages, userMessage];
        set({ messages: withUser, isTyping: true });

        try {
          // 2. Build the conversation history that claude.ts expects.
          //    We send up to MAX_MESSAGES – 1 previous messages as context
          //    (the current user message is passed separately as the first arg).
          const historyForApi = toAppChatMessages(
            withUser.slice(0, -1).slice(-MAX_MESSAGES),
          );

          // 3. Call the Claude AI helper.
          const assistantContent = await sendChatMessage(
            content.trim(),
            selectedLanguage,
            historyForApi,
            studentContext,
          );

          const assistantMessage: ChatMessage = {
            id: generateId(),
            role: 'assistant',
            content: assistantContent,
            timestamp: nowISO(),
            language: selectedLanguage,
          };

          // 4. Append assistant reply, then prune to MAX_MESSAGES.
          set((state) => {
            const updated = [...state.messages, assistantMessage];
            const pruned =
              updated.length > MAX_MESSAGES
                ? updated.slice(updated.length - MAX_MESSAGES)
                : updated;
            return { messages: pruned, isTyping: false };
          });
        } catch (err: unknown) {
          // Surface an inline error message so the UI never gets stuck.
          const errorMessage: ChatMessage = {
            id: generateId(),
            role: 'assistant',
            content:
              'Sorry, I ran into a problem. Please check your connection and try again.',
            timestamp: nowISO(),
            language: selectedLanguage,
          };

          set((state) => ({
            messages: [...state.messages, errorMessage],
            isTyping: false,
          }));

          console.warn(
            '[chatStore] sendMessage error:',
            err instanceof Error ? err.message : err,
          );
        }
      },

      clearChat: () => set({ messages: [] }),

      setLanguage: (code: LanguageCode) => set({ selectedLanguage: code }),
    }),

    {
      name: 'techsei-chat-store',
      storage: createJSONStorage(() => AsyncStorage),
      // isTyping is transient — never persist it.
      partialize: (state) => ({
        messages: state.messages.slice(-MAX_MESSAGES),
        selectedLanguage: state.selectedLanguage,
      }),
    },
  ),
);

// ─── Typed Selectors ──────────────────────────────────────────────────────────

export const selectMessages = (state: ChatStore) => state.messages;
export const selectIsTyping = (state: ChatStore) => state.isTyping;
export const selectSelectedLanguage = (state: ChatStore) => state.selectedLanguage;

/**
 * Returns the most recent `count` messages (default 20).
 * Useful for feeding a FlatList that renders only the visible window.
 */
export function selectRecentMessages(
  state: ChatStore,
  count = 20,
): ChatMessage[] {
  return state.messages.slice(-count);
}
