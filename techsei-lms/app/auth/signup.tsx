// ============================================================
// TechSei LMS — Sign Up Screen
// ============================================================
import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Animated,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '../../stores/authStore';

// ── Constants ─────────────────────────────────────────────────────────────────

const COLORS = {
  bg: '#0A0A1A',
  surface: '#141428',
  surfaceFocused: '#1A1A34',
  primary: '#6C63FF',
  primaryDim: '#4a4480',
  primaryLight: '#8B5CF6',
  accent: '#43E97B',
  border: '#2A2A44',
  borderFocused: '#6C63FF',
  iconDefault: '#555577',
  text: '#FFFFFF',
  textMuted: '#7777AA',
  textDim: '#55557A',
  textDimmer: '#44445A',
  label: '#AAAACC',
  error: '#FF6B6B',
  errorBg: '#FF6B6B18',
  errorBorder: '#FF6B6B44',
  errorText: '#FF8080',
};

// ── Password strength ─────────────────────────────────────────────────────────

type StrengthLevel = 'weak' | 'fair' | 'good' | 'strong';

function getPasswordStrength(password: string): {
  level: StrengthLevel;
  score: number;
  label: string;
} {
  let score = 0;
  if (password.length >= 8) score += 1;
  if (password.length >= 12) score += 1;
  if (/[A-Z]/.test(password)) score += 1;
  if (/[0-9]/.test(password)) score += 1;
  if (/[^A-Za-z0-9]/.test(password)) score += 1;

  if (score <= 1) return { level: 'weak', score, label: 'Weak' };
  if (score === 2) return { level: 'fair', score, label: 'Fair' };
  if (score === 3) return { level: 'good', score, label: 'Good' };
  return { level: 'strong', score, label: 'Strong' };
}

const STRENGTH_COLORS: Record<StrengthLevel, string> = {
  weak: '#FF6B6B',
  fair: '#FFB547',
  good: '#43E97B',
  strong: '#00C9FF',
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function SignupScreen() {
  const router = useRouter();
  const { signUp, isLoading, error, clearError, user, isAuthenticated } = useAuthStore();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [role, setRole] = useState<'student' | 'admin'>('student');
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  const [nameFocused, setNameFocused] = useState(false);
  const [emailFocused, setEmailFocused] = useState(false);
  const [passwordFocused, setPasswordFocused] = useState(false);

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 500, useNativeDriver: true }),
    ]).start();
  }, [fadeAnim, slideAnim]);

  // New users always go through onboarding
  useEffect(() => {
    if (isAuthenticated && user) {
      router.replace('/auth/onboarding');
    }
  }, [isAuthenticated, user, router]);

  const passwordStrength = getPasswordStrength(password);

  const handleSignUp = async () => {
    clearError();
    if (!name.trim()) {
      Alert.alert('Missing name', 'Please enter your full name.');
      return;
    }
    if (!email.trim()) {
      Alert.alert('Missing email', 'Please enter your email address.');
      return;
    }
    if (password.length < 8) {
      Alert.alert('Weak password', 'Password must be at least 8 characters.');
      return;
    }
    if (!agreedToTerms) {
      Alert.alert('Terms required', 'Please agree to the Terms & Conditions to continue.');
      return;
    }
    try {
      await signUp(email.trim().toLowerCase(), password, name.trim(), role);
    } catch {
      // Error displayed via store.error banner
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView
        contentContainerStyle={styles.scroll}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {/* Ambient background gradient */}
        <LinearGradient
          colors={['#43E97B18', '#0A0A1A00']}
          style={styles.bgBlob}
          start={{ x: 0.5, y: 0 }}
          end={{ x: 0.5, y: 1 }}
        />

        <Animated.View
          style={[styles.content, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}
        >
          {/* ── Back Button ───────────────────────────────────── */}
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => router.back()}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Ionicons name="arrow-back" size={22} color="#AAAACC" />
          </TouchableOpacity>

          {/* ── Header ────────────────────────────────────────── */}
          <Text style={styles.heading}>Create Account</Text>
          <Text style={styles.subHeading}>Join TechSei and start learning today</Text>

          {/* ── Error Banner ──────────────────────────────────── */}
          {error ? (
            <View style={styles.errorBanner}>
              <Ionicons name="alert-circle" size={16} color={COLORS.error} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          {/* ── Role Selector ─────────────────────────────────── */}
          <View style={styles.roleContainer}>
            <TouchableOpacity
              style={[styles.rolePill, role === 'student' && styles.rolePillActive]}
              onPress={() => setRole('student')}
              activeOpacity={0.75}
            >
              <Ionicons
                name="school-outline"
                size={16}
                color={role === 'student' ? '#fff' : '#6666AA'}
              />
              <Text style={[styles.rolePillText, role === 'student' && styles.rolePillTextActive]}>
                I'm a Student
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.rolePill, role === 'admin' && styles.rolePillActive]}
              onPress={() => setRole('admin')}
              activeOpacity={0.75}
            >
              <Ionicons
                name="shield-checkmark-outline"
                size={16}
                color={role === 'admin' ? '#fff' : '#6666AA'}
              />
              <Text style={[styles.rolePillText, role === 'admin' && styles.rolePillTextActive]}>
                I'm an Admin
              </Text>
            </TouchableOpacity>
          </View>

          {/* ── Full Name ─────────────────────────────────────── */}
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Full Name</Text>
            <View style={[styles.inputWrapper, nameFocused && styles.inputWrapperFocused]}>
              <Ionicons
                name="person-outline"
                size={20}
                color={nameFocused ? COLORS.primary : COLORS.iconDefault}
                style={styles.inputIcon}
              />
              <TextInput
                style={styles.input}
                placeholder="Your full name"
                placeholderTextColor={COLORS.textDimmer}
                value={name}
                onChangeText={setName}
                onFocus={() => setNameFocused(true)}
                onBlur={() => setNameFocused(false)}
                autoCapitalize="words"
                autoComplete="name"
                returnKeyType="next"
              />
            </View>
          </View>

          {/* ── Email ─────────────────────────────────────────── */}
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Email</Text>
            <View style={[styles.inputWrapper, emailFocused && styles.inputWrapperFocused]}>
              <Ionicons
                name="mail-outline"
                size={20}
                color={emailFocused ? COLORS.primary : COLORS.iconDefault}
                style={styles.inputIcon}
              />
              <TextInput
                style={styles.input}
                placeholder="you@example.com"
                placeholderTextColor={COLORS.textDimmer}
                value={email}
                onChangeText={setEmail}
                onFocus={() => setEmailFocused(true)}
                onBlur={() => setEmailFocused(false)}
                keyboardType="email-address"
                autoCapitalize="none"
                autoComplete="email"
                autoCorrect={false}
                returnKeyType="next"
              />
            </View>
          </View>

          {/* ── Password ──────────────────────────────────────── */}
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Password</Text>
            <View style={[styles.inputWrapper, passwordFocused && styles.inputWrapperFocused]}>
              <Ionicons
                name="lock-closed-outline"
                size={20}
                color={passwordFocused ? COLORS.primary : COLORS.iconDefault}
                style={styles.inputIcon}
              />
              <TextInput
                style={styles.input}
                placeholder="Min. 8 characters"
                placeholderTextColor={COLORS.textDimmer}
                value={password}
                onChangeText={setPassword}
                onFocus={() => setPasswordFocused(true)}
                onBlur={() => setPasswordFocused(false)}
                secureTextEntry={!showPassword}
                autoComplete="new-password"
                returnKeyType="done"
              />
              <TouchableOpacity
                onPress={() => setShowPassword((v) => !v)}
                style={styles.eyeBtn}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              >
                <Ionicons
                  name={showPassword ? 'eye-outline' : 'eye-off-outline'}
                  size={20}
                  color={COLORS.iconDefault}
                />
              </TouchableOpacity>
            </View>

            {/* Password strength indicator */}
            {password.length > 0 && (
              <View style={styles.strengthContainer}>
                <View style={styles.strengthBars}>
                  {[1, 2, 3, 4].map((i) => (
                    <View
                      key={i}
                      style={[
                        styles.strengthBar,
                        i <= passwordStrength.score
                          ? { backgroundColor: STRENGTH_COLORS[passwordStrength.level] }
                          : null,
                      ]}
                    />
                  ))}
                </View>
                <Text
                  style={[
                    styles.strengthLabel,
                    { color: STRENGTH_COLORS[passwordStrength.level] },
                  ]}
                >
                  {passwordStrength.label}
                </Text>
              </View>
            )}
          </View>

          {/* ── Terms Checkbox ────────────────────────────────── */}
          <TouchableOpacity
            style={styles.termsRow}
            onPress={() => setAgreedToTerms((v) => !v)}
            activeOpacity={0.7}
          >
            <View style={[styles.checkbox, agreedToTerms && styles.checkboxChecked]}>
              {agreedToTerms && <Ionicons name="checkmark" size={13} color="#fff" />}
            </View>
            <Text style={styles.termsText}>
              I agree to the{' '}
              <Text style={styles.termsLink}>Terms & Conditions</Text>
              {' '}and{' '}
              <Text style={styles.termsLink}>Privacy Policy</Text>
            </Text>
          </TouchableOpacity>

          {/* ── Create Account Button ─────────────────────────── */}
          <TouchableOpacity
            onPress={handleSignUp}
            disabled={isLoading}
            activeOpacity={0.85}
            style={styles.primaryBtnWrapper}
          >
            <LinearGradient
              colors={isLoading ? [COLORS.primaryDim, '#5e4fa3'] : [COLORS.primary, COLORS.primaryLight]}
              style={styles.primaryBtn}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              {isLoading ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Text style={styles.primaryBtnText}>Create Account</Text>
              )}
            </LinearGradient>
          </TouchableOpacity>

          {/* ── Sign In Link ──────────────────────────────────── */}
          <View style={styles.footerRow}>
            <Text style={styles.footerText}>Already have an account? </Text>
            <TouchableOpacity onPress={() => router.replace('/auth/login')}>
              <Text style={styles.footerLink}>Sign In</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  scroll: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingBottom: 48,
  },
  bgBlob: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 280,
  },
  content: {
    flex: 1,
    paddingTop: 60,
  },

  // ── Back ──
  backBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: COLORS.surface,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  // ── Headings ──
  heading: {
    fontSize: 28,
    fontWeight: '700',
    color: COLORS.text,
    marginBottom: 6,
  },
  subHeading: {
    fontSize: 15,
    color: COLORS.textMuted,
    marginBottom: 28,
  },

  // ── Error ──
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.errorBg,
    borderWidth: 1,
    borderColor: COLORS.errorBorder,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginBottom: 18,
    gap: 8,
  },
  errorText: {
    color: COLORS.errorText,
    fontSize: 13,
    flex: 1,
  },

  // ── Role Pills ──
  roleContainer: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 4,
    marginBottom: 24,
    gap: 4,
  },
  rolePill: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 9,
    gap: 6,
  },
  rolePillActive: {
    backgroundColor: COLORS.primary,
  },
  rolePillText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6666AA',
  },
  rolePillTextActive: {
    color: COLORS.text,
  },

  // ── Inputs ──
  inputGroup: {
    marginBottom: 18,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: COLORS.label,
    marginBottom: 8,
    letterSpacing: 0.3,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 14,
    height: 52,
  },
  inputWrapperFocused: {
    borderColor: COLORS.borderFocused,
    backgroundColor: COLORS.surfaceFocused,
  },
  inputIcon: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    fontSize: 15,
    color: COLORS.text,
    height: '100%',
  },
  eyeBtn: {
    padding: 4,
  },

  // ── Password Strength ──
  strengthContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    gap: 10,
  },
  strengthBars: {
    flex: 1,
    flexDirection: 'row',
    gap: 4,
  },
  strengthBar: {
    flex: 1,
    height: 4,
    borderRadius: 2,
    backgroundColor: COLORS.border,
  },
  strengthLabel: {
    fontSize: 12,
    fontWeight: '600',
    minWidth: 44,
    textAlign: 'right',
  },

  // ── Terms ──
  termsRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 24,
    gap: 12,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: '#3A3A5A',
    backgroundColor: 'transparent',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
    flexShrink: 0,
  },
  checkboxChecked: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  termsText: {
    flex: 1,
    fontSize: 13,
    color: COLORS.textMuted,
    lineHeight: 20,
  },
  termsLink: {
    color: COLORS.primary,
    fontWeight: '600',
  },

  // ── Primary Button ──
  primaryBtnWrapper: {
    marginBottom: 24,
    borderRadius: 14,
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
  },
  primaryBtn: {
    height: 54,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 14,
  },
  primaryBtnText: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },

  // ── Footer ──
  footerRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  footerText: {
    color: COLORS.textDim,
    fontSize: 14,
  },
  footerLink: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: '700',
  },
});
