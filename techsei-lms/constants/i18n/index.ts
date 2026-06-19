// ============================================================
// TechSei LMS — i18n Module
// ============================================================

import { useCallback } from 'react';
import en, { TranslationKeys } from './en';
import hi from './hi';

// ---------------------------------------------------------------------------
// Supported languages metadata
// ---------------------------------------------------------------------------

export interface SupportedLanguage {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
  rtl: boolean;
}

export const SUPPORTED_LANGUAGES: SupportedLanguage[] = [
  { code: 'en', name: 'English',    nativeName: 'English',    flag: '🇬🇧', rtl: false },
  { code: 'hi', name: 'Hindi',      nativeName: 'हिन्दी',      flag: '🇮🇳', rtl: false },
  { code: 'ta', name: 'Tamil',      nativeName: 'தமிழ்',       flag: '🇮🇳', rtl: false },
  { code: 'te', name: 'Telugu',     nativeName: 'తెలుగు',      flag: '🇮🇳', rtl: false },
  { code: 'kn', name: 'Kannada',    nativeName: 'ಕನ್ನಡ',       flag: '🇮🇳', rtl: false },
  { code: 'bn', name: 'Bengali',    nativeName: 'বাংলা',       flag: '🇮🇳', rtl: false },
  { code: 'mr', name: 'Marathi',    nativeName: 'मराठी',       flag: '🇮🇳', rtl: false },
  { code: 'ar', name: 'Arabic',     nativeName: 'العربية',     flag: '🇸🇦', rtl: true  },
  { code: 'fr', name: 'French',     nativeName: 'Français',    flag: '🇫🇷', rtl: false },
  { code: 'es', name: 'Spanish',    nativeName: 'Español',     flag: '🇪🇸', rtl: false },
];

// ---------------------------------------------------------------------------
// Partial translation type — all keys optional (only en is complete)
// ---------------------------------------------------------------------------

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type PartialTranslations = DeepPartial<TranslationKeys>;

// ---------------------------------------------------------------------------
// Tamil (ta) — basic translations
// ---------------------------------------------------------------------------

const ta: PartialTranslations = {
  app: {
    name: 'TechSei',
    tagline: 'தொழில்நுட்பம் கற்றுக்கொள்ளுங்கள். எப்போதும், எங்கும்.',
    version: 'பதிப்பு',
    powered_by: 'AI ஆல் இயக்கப்படுகிறது',
  },
  auth: {
    login: 'உள்நுழைய',
    signup: 'பதிவு செய்ய',
    logout: 'வெளியேறு',
    email: 'மின்னஞ்சல் முகவரி',
    password: 'கடவுச்சொல்',
    forgot_password: 'கடவுச்சொல் மறந்தீர்களா?',
    google_sign_in: 'Google மூலம் தொடரவும்',
    no_account: 'கணக்கு இல்லையா? பதிவு செய்யுங்கள்',
    have_account: 'கணக்கு உள்ளதா? உள்நுழையவும்',
    invalid_credentials: 'தவறான மின்னஞ்சல் அல்லது கடவுச்சொல்.',
  },
  nav: {
    home: 'முகப்பு',
    courses: 'படிப்புகள்',
    progress: 'முன்னேற்றம்',
    chat: 'AI ஆசிரியர்',
    leaderboard: 'தரவரிசை',
    profile: 'சுயவிவரம்',
    back: 'திரும்பு',
  },
  home: {
    good_morning: 'காலை வணக்கம்',
    good_afternoon: 'மதிய வணக்கம்',
    good_evening: 'மாலை வணக்கம்',
    continue_learning: 'கற்றலை தொடரவும்',
    recommended: 'உங்களுக்கு பரிந்துரைக்கப்பட்டவை',
    explore_courses: 'படிப்புகளை ஆராயவும்',
  },
  courses: {
    browse: 'படிப்புகளை உலாவுங்கள்',
    search: 'படிப்புகளை தேடுங்கள்...',
    enroll: 'இப்போது சேரவும்',
    free: 'இலவசம்',
    premium: 'பிரீமியம்',
    start_lesson: 'பாடத்தை தொடங்கவும்',
    mark_complete: 'முழுமையானது என குறிக்கவும்',
    download_offline: 'ஆஃப்லைனில் பதிவிறக்கவும்',
  },
  common: {
    save: 'சேமி',
    cancel: 'ரத்து செய்',
    delete: 'நீக்கு',
    edit: 'திருத்து',
    loading: 'ஏற்றுகிறது...',
    error: 'ஏதோ தவறு ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.',
    success: 'வெற்றி!',
    retry: 'மீண்டும் முயற்சிக்கவும்',
    back: 'திரும்பு',
  },
};

// ---------------------------------------------------------------------------
// Telugu (te) — basic translations
// ---------------------------------------------------------------------------

const te: PartialTranslations = {
  app: {
    name: 'TechSei',
    tagline: 'సాంకేతిక విద్య నేర్చుకోండి. ఎప్పుడైనా, ఎక్కడైనా.',
    version: 'వెర్షన్',
    powered_by: 'AI ద్వారా నడుపబడుతుంది',
  },
  auth: {
    login: 'లాగిన్ చేయండి',
    signup: 'సైన్ అప్ చేయండి',
    logout: 'లాగ్ అవుట్',
    email: 'ఇమెయిల్ చిరునామా',
    password: 'పాస్‌వర్డ్',
    forgot_password: 'పాస్‌వర్డ్ మర్చిపోయారా?',
    google_sign_in: 'Google తో కొనసాగండి',
    no_account: 'ఖాతా లేదా? సైన్ అప్ చేయండి',
    have_account: 'ఖాతా ఉందా? లాగిన్ చేయండి',
    invalid_credentials: 'తప్పు ఇమెయిల్ లేదా పాస్‌వర్డ్.',
  },
  nav: {
    home: 'హోమ్',
    courses: 'కోర్సులు',
    progress: 'పురోగతి',
    chat: 'AI ట్యూటర్',
    leaderboard: 'లీడర్‌బోర్డ్',
    profile: 'ప్రొఫైల్',
    back: 'వెనుకకు',
  },
  home: {
    good_morning: 'శుభోదయం',
    good_afternoon: 'శుభ మధ్యాహ్నం',
    good_evening: 'శుభ సాయంత్రం',
    continue_learning: 'నేర్చుకోవడం కొనసాగించండి',
    recommended: 'మీకు సిఫార్సు చేయబడినవి',
    explore_courses: 'కోర్సులు అన్వేషించండి',
  },
  courses: {
    browse: 'కోర్సులు బ్రౌజ్ చేయండి',
    search: 'కోర్సులు వెతకండి...',
    enroll: 'ఇప్పుడే చేరండి',
    free: 'ఉచితం',
    premium: 'ప్రీమియం',
    start_lesson: 'పాఠం ప్రారంభించండి',
    mark_complete: 'పూర్తి అయినట్లు గుర్తించండి',
    download_offline: 'ఆఫ్‌లైన్‌లో డౌన్‌లోడ్ చేయండి',
  },
  common: {
    save: 'సేవ్ చేయండి',
    cancel: 'రద్దు చేయండి',
    delete: 'తొలగించండి',
    edit: 'సవరించండి',
    loading: 'లోడ్ అవుతోంది...',
    error: 'ఏదో తప్పు జరిగింది. దయచేసి మళ్ళీ ప్రయత్నించండి.',
    success: 'విజయం!',
    retry: 'మళ్ళీ ప్రయత్నించండి',
    back: 'వెనుకకు',
  },
};

// ---------------------------------------------------------------------------
// Arabic (ar) — basic translations (RTL)
// ---------------------------------------------------------------------------

const ar: PartialTranslations = {
  app: {
    name: 'TechSei',
    tagline: 'تعلّم التكنولوجيا. في أي وقت، في أي مكان.',
    version: 'الإصدار',
    powered_by: 'مدعوم بالذكاء الاصطناعي',
  },
  auth: {
    login: 'تسجيل الدخول',
    signup: 'إنشاء حساب',
    logout: 'تسجيل الخروج',
    email: 'البريد الإلكتروني',
    password: 'كلمة المرور',
    forgot_password: 'نسيت كلمة المرور؟',
    google_sign_in: 'المتابعة مع Google',
    no_account: 'ليس لديك حساب؟ سجّل الآن',
    have_account: 'لديك حساب بالفعل؟ سجّل الدخول',
    invalid_credentials: 'البريد الإلكتروني أو كلمة المرور غير صحيحة.',
  },
  nav: {
    home: 'الرئيسية',
    courses: 'الدورات',
    progress: 'التقدم',
    chat: 'المعلم الذكي',
    leaderboard: 'المتصدرون',
    profile: 'الملف الشخصي',
    back: 'رجوع',
  },
  home: {
    good_morning: 'صباح الخير',
    good_afternoon: 'مساء النور',
    good_evening: 'مساء الخير',
    continue_learning: 'تابع التعلم',
    recommended: 'موصى لك',
    explore_courses: 'استكشف الدورات',
  },
  courses: {
    browse: 'تصفح الدورات',
    search: 'ابحث عن دورات...',
    enroll: 'سجّل الآن',
    free: 'مجاني',
    premium: 'مميز',
    start_lesson: 'ابدأ الدرس',
    mark_complete: 'وضع علامة مكتمل',
    download_offline: 'تحميل للاستخدام دون إنترنت',
  },
  common: {
    save: 'حفظ',
    cancel: 'إلغاء',
    delete: 'حذف',
    edit: 'تعديل',
    loading: 'جارٍ التحميل...',
    error: 'حدث خطأ ما. يرجى المحاولة مرة أخرى.',
    success: 'نجاح!',
    retry: 'إعادة المحاولة',
    back: 'رجوع',
  },
};

// ---------------------------------------------------------------------------
// French (fr) — basic translations
// ---------------------------------------------------------------------------

const fr: PartialTranslations = {
  app: {
    name: 'TechSei',
    tagline: 'Apprenez la technologie. N'importe quand, n'importe où.',
    version: 'Version',
    powered_by: 'Propulsé par l'IA',
  },
  auth: {
    login: 'Se connecter',
    signup: 'S'inscrire',
    logout: 'Se déconnecter',
    email: 'Adresse e-mail',
    password: 'Mot de passe',
    forgot_password: 'Mot de passe oublié ?',
    google_sign_in: 'Continuer avec Google',
    no_account: 'Pas de compte ? Inscrivez-vous',
    have_account: 'Déjà un compte ? Connectez-vous',
    invalid_credentials: 'E-mail ou mot de passe invalide.',
  },
  nav: {
    home: 'Accueil',
    courses: 'Cours',
    progress: 'Progrès',
    chat: 'Tuteur IA',
    leaderboard: 'Classement',
    profile: 'Profil',
    back: 'Retour',
  },
  home: {
    good_morning: 'Bonjour',
    good_afternoon: 'Bon après-midi',
    good_evening: 'Bonsoir',
    continue_learning: 'Continuer à apprendre',
    recommended: 'Recommandé pour vous',
    explore_courses: 'Explorer les cours',
  },
  courses: {
    browse: 'Parcourir les cours',
    search: 'Rechercher des cours...',
    enroll: 'S'inscrire maintenant',
    free: 'Gratuit',
    premium: 'Premium',
    start_lesson: 'Commencer la leçon',
    mark_complete: 'Marquer comme terminé',
    download_offline: 'Télécharger hors ligne',
  },
  common: {
    save: 'Enregistrer',
    cancel: 'Annuler',
    delete: 'Supprimer',
    edit: 'Modifier',
    loading: 'Chargement...',
    error: 'Une erreur s'est produite. Veuillez réessayer.',
    success: 'Succès !',
    retry: 'Réessayer',
    back: 'Retour',
  },
};

// ---------------------------------------------------------------------------
// Spanish (es) — basic translations
// ---------------------------------------------------------------------------

const es: PartialTranslations = {
  app: {
    name: 'TechSei',
    tagline: 'Aprende tecnología. En cualquier momento, en cualquier lugar.',
    version: 'Versión',
    powered_by: 'Impulsado por IA',
  },
  auth: {
    login: 'Iniciar sesión',
    signup: 'Registrarse',
    logout: 'Cerrar sesión',
    email: 'Correo electrónico',
    password: 'Contraseña',
    forgot_password: '¿Olvidaste tu contraseña?',
    google_sign_in: 'Continuar con Google',
    no_account: '¿No tienes cuenta? Regístrate',
    have_account: '¿Ya tienes cuenta? Inicia sesión',
    invalid_credentials: 'Correo o contraseña inválidos.',
  },
  nav: {
    home: 'Inicio',
    courses: 'Cursos',
    progress: 'Progreso',
    chat: 'Tutor IA',
    leaderboard: 'Clasificación',
    profile: 'Perfil',
    back: 'Volver',
  },
  home: {
    good_morning: 'Buenos días',
    good_afternoon: 'Buenas tardes',
    good_evening: 'Buenas noches',
    continue_learning: 'Continuar aprendiendo',
    recommended: 'Recomendado para ti',
    explore_courses: 'Explorar cursos',
  },
  courses: {
    browse: 'Explorar cursos',
    search: 'Buscar cursos...',
    enroll: 'Inscribirse ahora',
    free: 'Gratis',
    premium: 'Premium',
    start_lesson: 'Iniciar lección',
    mark_complete: 'Marcar como completado',
    download_offline: 'Descargar sin conexión',
  },
  common: {
    save: 'Guardar',
    cancel: 'Cancelar',
    delete: 'Eliminar',
    edit: 'Editar',
    loading: 'Cargando...',
    error: 'Algo salió mal. Por favor, inténtalo de nuevo.',
    success: '¡Éxito!',
    retry: 'Reintentar',
    back: 'Volver',
  },
};

// ---------------------------------------------------------------------------
// Translations registry
// ---------------------------------------------------------------------------

export const translations: Record<string, TranslationKeys | PartialTranslations> = {
  en,
  hi,
  ta,
  te,
  kn: {
    app: { name: 'TechSei', tagline: 'ತಂತ್ರಜ್ಞಾನ ಕಲಿಯಿರಿ. ಯಾವಾಗಲಾದರೂ, ಎಲ್ಲಿಯಾದರೂ.', version: 'ಆವೃತ್ತಿ', powered_by: 'AI ಯಿಂದ ಚಾಲಿತ' },
    auth: { login: 'ಲಾಗಿನ್ ಮಾಡಿ', signup: 'ಸೈನ್ ಅಪ್ ಮಾಡಿ', logout: 'ಲಾಗ್ ಔಟ್', email: 'ಇಮೇಲ್ ವಿಳಾಸ', password: 'ಪಾಸ್‌ವರ್ಡ್', forgot_password: 'ಪಾಸ್‌ವರ್ಡ್ ಮರೆತಿರಾ?', google_sign_in: 'Google ನೊಂದಿಗೆ ಮುಂದುವರಿಯಿರಿ', no_account: 'ಖಾತೆ ಇಲ್ಲವೇ? ಸೈನ್ ಅಪ್ ಮಾಡಿ', have_account: 'ಖಾತೆ ಇದೆಯೇ? ಲಾಗಿನ್ ಮಾಡಿ', invalid_credentials: 'ತಪ್ಪಾದ ಇಮೇಲ್ ಅಥವಾ ಪಾಸ್‌ವರ್ಡ್.' },
    nav: { home: 'ಮುಖ್ಯ', courses: 'ಕೋರ್ಸ್‌ಗಳು', progress: 'ಪ್ರಗತಿ', chat: 'AI ಟ್ಯೂಟರ್', leaderboard: 'ಲೀಡರ್‌ಬೋರ್ಡ್', profile: 'ಪ್ರೊಫೈಲ್', back: 'ಹಿಂದೆ' },
    common: { save: 'ಉಳಿಸಿ', cancel: 'ರದ್ದುಗೊಳಿಸಿ', delete: 'ಅಳಿಸಿ', edit: 'ತಿದ್ದಿರಿ', loading: 'ಲೋಡ್ ಆಗುತ್ತಿದೆ...', error: 'ಏನೋ ತಪ್ಪಾಗಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.', success: 'ಯಶಸ್ಸು!', retry: 'ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ', back: 'ಹಿಂದೆ' },
  } as PartialTranslations,
  bn: {
    app: { name: 'TechSei', tagline: 'প্রযুক্তি শিখুন। যেকোনো সময়, যেকোনো জায়গায়।', version: 'সংস্করণ', powered_by: 'AI দ্বারা চালিত' },
    auth: { login: 'লগ ইন করুন', signup: 'সাইন আপ করুন', logout: 'লগ আউট', email: 'ইমেইল ঠিকানা', password: 'পাসওয়ার্ড', forgot_password: 'পাসওয়ার্ড ভুলে গেছেন?', google_sign_in: 'Google দিয়ে চালিয়ে যান', no_account: 'অ্যাকাউন্ট নেই? সাইন আপ করুন', have_account: 'অ্যাকাউন্ট আছে? লগ ইন করুন', invalid_credentials: 'ভুল ইমেইল বা পাসওয়ার্ড।' },
    nav: { home: 'হোম', courses: 'কোর্স', progress: 'অগ্রগতি', chat: 'AI টিউটর', leaderboard: 'লিডারবোর্ড', profile: 'প্রোফাইল', back: 'ফিরে যান' },
    common: { save: 'সংরক্ষণ করুন', cancel: 'বাতিল করুন', delete: 'মুছুন', edit: 'সম্পাদনা করুন', loading: 'লোড হচ্ছে...', error: 'কিছু ভুল হয়েছে। আবার চেষ্টা করুন।', success: 'সফল!', retry: 'আবার চেষ্টা করুন', back: 'ফিরে যান' },
  } as PartialTranslations,
  mr: {
    app: { name: 'TechSei', tagline: 'तंत्रज्ञान शिका. कधीही, कुठेही.', version: 'आवृत्ती', powered_by: 'AI द्वारे समर्थित' },
    auth: { login: 'लॉग इन करा', signup: 'साइन अप करा', logout: 'लॉग आउट', email: 'ईमेल पत्ता', password: 'पासवर्ड', forgot_password: 'पासवर्ड विसरलात?', google_sign_in: 'Google सोबत सुरू ठेवा', no_account: 'खाते नाही? साइन अप करा', have_account: 'खाते आहे? लॉग इन करा', invalid_credentials: 'चुकीचा ईमेल किंवा पासवर्ड.' },
    nav: { home: 'मुखपृष्ठ', courses: 'अभ्यासक्रम', progress: 'प्रगती', chat: 'AI शिक्षक', leaderboard: 'लीडरबोर्ड', profile: 'प्रोफाइल', back: 'मागे' },
    common: { save: 'जतन करा', cancel: 'रद्द करा', delete: 'हटवा', edit: 'संपादित करा', loading: 'लोड होत आहे...', error: 'काहीतरी चुकले. कृपया पुन्हा प्रयत्न करा.', success: 'यश!', retry: 'पुन्हा प्रयत्न करा', back: 'मागे' },
  } as PartialTranslations,
  ar,
  fr,
  es,
};

// ---------------------------------------------------------------------------
// Deep-get a nested key from a translation object with fallback to English
// ---------------------------------------------------------------------------

function getNestedValue(obj: Record<string, unknown>, keys: string[]): string | undefined {
  let current: unknown = obj;
  for (const key of keys) {
    if (current == null || typeof current !== 'object') return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return typeof current === 'string' ? current : undefined;
}

// ---------------------------------------------------------------------------
// useTranslation hook
// This hook reads the current language from the auth/settings store.
// It returns a `t()` function that resolves dot-notation keys like
// "auth.login" or "common.save", with English fallback if key is missing.
// ---------------------------------------------------------------------------

/**
 * Interpolate {{variable}} placeholders in a translation string.
 * e.g. interpolate("Hello {{name}}!", { name: "Ana" }) => "Hello Ana!"
 */
function interpolate(str: string, vars?: Record<string, string | number>): string {
  if (!vars) return str;
  return str.replace(/\{\{(\w+)\}\}/g, (_, key) =>
    vars[key] !== undefined ? String(vars[key]) : `{{${key}}}`
  );
}

/**
 * Retrieves a translation for the given dot-notation key.
 * Falls back to English if the key is not found in the target language.
 */
function resolveKey(
  languageCode: string,
  key: string,
  vars?: Record<string, string | number>
): string {
  const keys = key.split('.');
  const langTranslations = translations[languageCode] as Record<string, unknown> | undefined;
  const enTranslations = translations['en'] as Record<string, unknown>;

  let result: string | undefined;

  if (langTranslations) {
    result = getNestedValue(langTranslations, keys);
  }

  // Fallback to English
  if (result === undefined) {
    result = getNestedValue(enTranslations, keys);
  }

  // If still undefined, return the key itself so developers can spot missing strings
  if (result === undefined) {
    console.warn(`[i18n] Missing translation key: "${key}" for language "${languageCode}"`);
    return key;
  }

  return interpolate(result, vars);
}

// ---------------------------------------------------------------------------
// Standalone translate function (for use outside React components)
// ---------------------------------------------------------------------------

export function createTranslator(languageCode: string) {
  return (key: string, vars?: Record<string, string | number>): string =>
    resolveKey(languageCode, key, vars);
}

// ---------------------------------------------------------------------------
// useTranslation React hook
// Reads language from Zustand auth store; falls back to 'en'.
// ---------------------------------------------------------------------------

export function useTranslation() {
  // We do a lazy require to avoid circular imports. The auth store exports
  // the current user's language_pref. If the store is not yet initialised
  // (e.g. during onboarding) we default to 'en'.
  let languageCode = 'en';
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { useAuthStore } = require('@/stores/authStore') as {
      useAuthStore: () => { user: { language_pref?: string } | null };
    };
    // Note: calling a hook inside another hook is valid in React.
    const { user } = useAuthStore();
    if (user?.language_pref) {
      languageCode = user.language_pref;
    }
  } catch {
    // Store not available — use default
  }

  const t = useCallback(
    (key: string, vars?: Record<string, string | number>): string =>
      resolveKey(languageCode, key, vars),
    [languageCode]
  );

  const isRTL =
    SUPPORTED_LANGUAGES.find((lang) => lang.code === languageCode)?.rtl ?? false;

  return { t, languageCode, isRTL };
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export { en, hi, ta, te, ar, fr, es };
export type { PartialTranslations };
export default translations;
