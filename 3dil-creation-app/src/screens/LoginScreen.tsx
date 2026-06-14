import React, { useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, TextInput,
  ActivityIndicator, Alert, KeyboardAvoidingView, Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, FontSize, Spacing, BorderRadius } from '../theme';
import { useAuth } from '../context/AuthContext';

type Nav = NativeStackNavigationProp<RootStackParamList>;

const LoginScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please enter email and password.');
      return;
    }
    setLoading(true);
    const success = await login(email, password);
    setLoading(false);
    if (success) {
      navigation.goBack();
    } else {
      Alert.alert('Login Failed', 'Invalid credentials. Please try again.');
    }
  };

  return (
    <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <LinearGradient colors={[Colors.primary, Colors.secondary]} style={styles.header}>
        <TouchableOpacity style={styles.closeBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.logo}>🖨️</Text>
        <Text style={styles.brandName}>3DIL CREATION</Text>
        <Text style={styles.tagline}>Sign in to your account</Text>
      </LinearGradient>

      <View style={styles.form}>
        <Text style={styles.welcomeText}>Welcome Back! 👋</Text>

        <View style={styles.inputWrapper}>
          <Ionicons name="mail-outline" size={20} color={Colors.textLight} style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Email address"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            placeholderTextColor={Colors.textLight}
          />
        </View>

        <View style={styles.inputWrapper}>
          <Ionicons name="lock-closed-outline" size={20} color={Colors.textLight} style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Password"
            value={password}
            onChangeText={setPassword}
            secureTextEntry={!showPass}
            placeholderTextColor={Colors.textLight}
          />
          <TouchableOpacity onPress={() => setShowPass(v => !v)} style={styles.eyeBtn}>
            <Ionicons name={showPass ? 'eye-off-outline' : 'eye-outline'} size={20} color={Colors.textLight} />
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.forgotBtn}>
          <Text style={styles.forgotText}>Forgot Password?</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.loginBtn} onPress={handleLogin} disabled={loading}>
          <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.loginBtnGradient}>
            {loading ? (
              <ActivityIndicator color={Colors.white} />
            ) : (
              <Text style={styles.loginBtnText}>Sign In</Text>
            )}
          </LinearGradient>
        </TouchableOpacity>

        <View style={styles.divider}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>or continue with</Text>
          <View style={styles.dividerLine} />
        </View>

        <View style={styles.socialBtns}>
          <TouchableOpacity style={styles.socialBtn}>
            <Text style={styles.socialBtnEmoji}>🔵</Text>
            <Text style={styles.socialBtnText}>Google</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.socialBtn}>
            <Text style={styles.socialBtnEmoji}>📱</Text>
            <Text style={styles.socialBtnText}>Phone OTP</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.registerRow}>
          <Text style={styles.registerText}>Don't have an account? </Text>
          <TouchableOpacity onPress={() => navigation.replace('Register')}>
            <Text style={styles.registerLink}>Sign Up →</Text>
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  header: { paddingTop: 52, paddingBottom: 32, paddingHorizontal: Spacing.md, alignItems: 'center' },
  closeBtn: { position: 'absolute', top: 52, right: 16, width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  logo: { fontSize: 48, marginBottom: 8 },
  brandName: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900', letterSpacing: 3, marginBottom: 4 },
  tagline: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md },
  form: { flex: 1, padding: Spacing.lg, backgroundColor: Colors.background },
  welcomeText: { fontSize: FontSize.xxl, fontWeight: '900', color: Colors.textPrimary, marginBottom: 24 },
  inputWrapper: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.card, borderRadius: BorderRadius.md, borderWidth: 1.5, borderColor: Colors.border, paddingHorizontal: 14, marginBottom: 14 },
  inputIcon: { marginRight: 10 },
  input: { flex: 1, paddingVertical: 14, fontSize: FontSize.md, color: Colors.textPrimary },
  eyeBtn: { padding: 4 },
  forgotBtn: { alignSelf: 'flex-end', marginBottom: 20 },
  forgotText: { color: Colors.primary, fontWeight: '600', fontSize: FontSize.md },
  loginBtn: { borderRadius: BorderRadius.lg, overflow: 'hidden', marginBottom: 20 },
  loginBtnGradient: { paddingVertical: 18, alignItems: 'center', borderRadius: BorderRadius.lg },
  loginBtnText: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
  divider: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 20 },
  dividerLine: { flex: 1, height: 1, backgroundColor: Colors.border },
  dividerText: { color: Colors.textSecondary, fontSize: FontSize.sm, fontWeight: '600' },
  socialBtns: { flexDirection: 'row', gap: 12, marginBottom: 24 },
  socialBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: Colors.card, borderRadius: BorderRadius.md, paddingVertical: 14, borderWidth: 1.5, borderColor: Colors.border },
  socialBtnEmoji: { fontSize: 20 },
  socialBtnText: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  registerRow: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center' },
  registerText: { color: Colors.textSecondary, fontSize: FontSize.md },
  registerLink: { color: Colors.primary, fontWeight: '800', fontSize: FontSize.md },
});

export default LoginScreen;
