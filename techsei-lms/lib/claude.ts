// ============================================================
// TechSei LMS — Claude AI Tutor Integration
// ============================================================

import type { ChatMessage } from '@/types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const CLAUDE_MODEL = 'claude-sonnet-4-6';

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_API_VERSION = '2023-06-01';
const MAX_TOKENS = 1024;

/** Loaded from Expo environment. Replace placeholder before shipping. */
const ANTHROPIC_API_KEY =
  process.env.EXPO_PUBLIC_ANTHROPIC_API_KEY ?? 'your-anthropic-api-key-here';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  system: string;
  messages: AnthropicMessage[];
}

interface AnthropicContentBlock {
  type: 'text' | 'tool_use';
  text?: string;
}

interface AnthropicResponse {
  id: string;
  type: string;
  role: 'assistant';
  content: AnthropicContentBlock[];
  model: string;
  stop_reason: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

interface AnthropicErrorResponse {
  type: 'error';
  error: {
    type: string;
    message: string;
  };
}

export interface StudentContext {
  name?: string;
  level?: number;
  currentCourse?: string;
  streak?: number;
  subscriptionTier?: string;
}

// ---------------------------------------------------------------------------
// Language name mapping (for system prompt)
// ---------------------------------------------------------------------------

const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  hi: 'Hindi (हिन्दी)',
  ta: 'Tamil (தமிழ்)',
  te: 'Telugu (తెలుగు)',
  kn: 'Kannada (ಕನ್ನಡ)',
  bn: 'Bengali (বাংলা)',
  mr: 'Marathi (मराठी)',
  ar: 'Arabic (العربية)',
  fr: 'French (Français)',
  es: 'Spanish (Español)',
};

// ---------------------------------------------------------------------------
// System prompt builder
// ---------------------------------------------------------------------------

function buildSystemPrompt(language: string, studentContext?: StudentContext): string {
  const languageName = LANGUAGE_NAMES[language] ?? 'English';

  const contextLines: string[] = [];

  if (studentContext?.name) {
    contextLines.push(`- Student name: ${studentContext.name}`);
  }
  if (studentContext?.level !== undefined) {
    contextLines.push(`- Student level: ${studentContext.level}`);
  }
  if (studentContext?.currentCourse) {
    contextLines.push(`- Currently studying: ${studentContext.currentCourse}`);
  }
  if (studentContext?.streak !== undefined && studentContext.streak > 0) {
    contextLines.push(`- Learning streak: ${studentContext.streak} day(s)`);
  }
  if (studentContext?.subscriptionTier) {
    contextLines.push(`- Subscription tier: ${studentContext.subscriptionTier}`);
  }

  const contextBlock =
    contextLines.length > 0
      ? `\n\nStudent context:\n${contextLines.join('\n')}`
      : '';

  return `You are TechSei AI Tutor, an expert educational assistant specializing in technology education on the TechSei LMS platform. Your role is to help students learn programming, web development, mobile development, data science, cloud computing, cybersecurity, AI/ML, DevOps, and other technology topics.

Core responsibilities:
1. Answer questions clearly, accurately, and at the appropriate level for the student
2. Provide practical code examples when relevant, using proper formatting
3. Break down complex concepts into simple, digestible explanations
4. Encourage students and celebrate their progress
5. Suggest next steps or related topics to explore
6. Help debug code and explain errors in a constructive, non-condescending way
7. Guide students toward problem-solving skills rather than just giving answers directly

Language instruction:
- ALWAYS respond in ${languageName}
- If the student writes in a different language, still reply in ${languageName}
- Use clear, natural ${languageName} — avoid awkward literal translations
- For code blocks, variable names, and technical terms, you may keep them in English even when responding in other languages, as this is standard practice in technology education${contextBlock}

Tone and style:
- Be warm, encouraging, and patient
- Acknowledge when a question is particularly good or insightful
- Never make the student feel bad for not knowing something
- Keep responses focused and concise — use bullet points and code blocks for clarity
- If you are unsure about something, say so honestly rather than guessing

Platform context:
- You are embedded in the TechSei mobile app (React Native / Expo)
- Students may ask about courses they are taking on the platform
- You can reference general learning paths for technology topics
- You do not have access to real-time internet, but you have strong knowledge of technology topics up to your training cutoff`;
}

// ---------------------------------------------------------------------------
// Conversation history converter
// ---------------------------------------------------------------------------

/**
 * Converts TechSei ChatMessage records into the Anthropic messages format.
 * Only includes completed messages (no loading state).
 * Limits history to the most recent N exchanges to stay within token limits.
 */
function buildConversationHistory(
  history: ChatMessage[],
  maxExchanges = 10
): AnthropicMessage[] {
  const completed = history.filter(
    (msg) => !msg.is_loading && msg.response && msg.message
  );

  // Take the most recent maxExchanges entries
  const recent = completed.slice(-maxExchanges);

  const messages: AnthropicMessage[] = [];

  for (const exchange of recent) {
    messages.push({ role: 'user', content: exchange.message });
    messages.push({ role: 'assistant', content: exchange.response });
  }

  return messages;
}

// ---------------------------------------------------------------------------
// Core chat function
// ---------------------------------------------------------------------------

/**
 * Send a message to the Claude AI Tutor and receive a response.
 *
 * @param message - The student's current message
 * @param language - The student's preferred language code (e.g. 'en', 'hi')
 * @param conversationHistory - Previous exchanges in this session (for context)
 * @param studentContext - Optional student profile data to personalize the response
 * @returns The assistant's response text
 * @throws Error if the API call fails after retries
 */
export async function sendChatMessage(
  message: string,
  language: string,
  conversationHistory: ChatMessage[],
  studentContext?: StudentContext
): Promise<string> {
  if (!message.trim()) {
    throw new Error('Message cannot be empty.');
  }

  const systemPrompt = buildSystemPrompt(language, studentContext);
  const history = buildConversationHistory(conversationHistory);

  // Append the current user message
  const messages: AnthropicMessage[] = [
    ...history,
    { role: 'user', content: message.trim() },
  ];

  const requestBody: AnthropicRequest = {
    model: CLAUDE_MODEL,
    max_tokens: MAX_TOKENS,
    system: systemPrompt,
    messages,
  };

  const response = await fetchWithRetry(requestBody);
  return response;
}

// ---------------------------------------------------------------------------
// HTTP call with exponential-backoff retry
// ---------------------------------------------------------------------------

async function fetchWithRetry(
  requestBody: AnthropicRequest,
  maxRetries = 2
): Promise<string> {
  let lastError: Error = new Error('Unknown error');

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(ANTHROPIC_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': ANTHROPIC_API_KEY,
          'anthropic-version': ANTHROPIC_API_VERSION,
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorBody = (await response.json().catch(() => null)) as AnthropicErrorResponse | null;
        const errorMessage =
          errorBody?.error?.message ?? `HTTP ${response.status}: ${response.statusText}`;

        // Do not retry on 4xx client errors (except 429 rate-limit)
        if (response.status >= 400 && response.status < 500 && response.status !== 429) {
          throw new Error(errorMessage);
        }

        lastError = new Error(errorMessage);

        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000; // 1s, 2s, …
          await sleep(delay);
          continue;
        }

        throw lastError;
      }

      const data = (await response.json()) as AnthropicResponse;

      // Extract text from the first text content block
      const textBlock = data.content.find(
        (block): block is AnthropicContentBlock & { text: string } =>
          block.type === 'text' && typeof block.text === 'string'
      );

      if (!textBlock?.text) {
        throw new Error('The AI returned an empty response. Please try again.');
      }

      return textBlock.text;
    } catch (err) {
      if (err instanceof Error) {
        lastError = err;
      }

      // If it's not a retriable error (e.g. a 400), re-throw immediately
      if (
        err instanceof Error &&
        (err.message.startsWith('HTTP 4') && !err.message.startsWith('HTTP 429'))
      ) {
        throw err;
      }

      if (attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000;
        await sleep(delay);
        continue;
      }
    }
  }

  throw new Error(
    `AI Tutor is temporarily unavailable. Please try again in a moment. (${lastError.message})`
  );
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Additional convenience helpers
// ---------------------------------------------------------------------------

/**
 * Generate a short, contextual greeting message for the chat screen.
 * This is a lightweight call that doesn't consume the full conversation context.
 */
export async function generateGreeting(
  studentName: string,
  language: string,
  currentCourse?: string
): Promise<string> {
  const courseNote = currentCourse
    ? ` I see you're working on "${currentCourse}" — great choice!`
    : '';

  const greetingMessage = `Generate a short, warm, encouraging greeting (2-3 sentences) for a student named ${studentName} who just opened the TechSei AI Tutor chat.${courseNote} Make it feel personal and motivating. Respond in ${LANGUAGE_NAMES[language] ?? 'English'}.`;

  try {
    const response = await sendChatMessage(greetingMessage, language, [], {
      name: studentName,
      currentCourse,
    });
    return response;
  } catch {
    // Fallback greeting if API is unavailable
    return `Hi ${studentName}! I'm your TechSei AI Tutor. How can I help you learn today?`;
  }
}

/**
 * Check if the Anthropic API key is configured (not the placeholder).
 */
export function isApiConfigured(): boolean {
  return (
    ANTHROPIC_API_KEY !== 'your-anthropic-api-key-here' &&
    ANTHROPIC_API_KEY.startsWith('sk-ant-')
  );
}
