import React, { useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, TextInput,
  ActivityIndicator, Alert, ScrollView, KeyboardAvoidingView, Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, FontSize, Spacing, BorderRadius } from '../theme';
import { useAuth } from '../context/AuthContext';

type Nav = NativeStackNavigationProp<RootStackParamList>;

const RegisterScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { register } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const [referralCode, setReferralCode] = useState('');
  const [agreed, setAgreed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showPass, setShowPass] = useState(false);

  const handleRegister = async () => {
    if (!name || !email || !phone || !password) {
      Alert.alert('Error', 'Please fill in all required fields.');
      return;
    }
    if (!agreed) {
      Alert.alert('Terms Required', 'Please accept the terms and conditions.');
      return;
    }
    if (password.length < 6) {
      Alert.alert('Weak Password', 'Password must be at least 6 characters.');
      return;
    }
    setLoading(true);
    const success = await register(name, email, phone, password, referralCode);
    setLoading(false);
    if (success) {
      Alert.alert('🎉 Welcome!', `Account created successfully! You earned 100 welcome bonus points!`, [
        { text: 'Start Shopping', onPress: () => navigation.navigate('Main') },
      ]);
    } else {
      Alert.alert('Error', 'Registration failed. Please try again.');
    }
  };

  return (
    <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <LinearGradient colors={[Colors.secondary, Colors.primary]} style={styles.header}>
        <TouchableOpacity style={styles.closeBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.logo}>🖨️</Text>
        <Text style={styles.brandName}>3DIL CREATION</Text>
        <Text style={styles.tagline}>Create your account</Text>
      </LinearGradient>

      <ScrollView style={styles.form} showsVerticalScrollIndicator={false}>
        <Text style={styles.welcomeText}>Join 3DIL Family! 🎉</Text>
        <Text style={styles.welcomeDesc}>Get 100 welcome points + exclusive deals</Text>

        {[
          { placeholder: 'Full Name *', value: name, set: setName, icon: 'person-outline', type: 'default' },
          { placeholder: 'Email Address *', value: email, set: setEmail, icon: 'mail-outline', type: 'email-address' },
          { placeholder: 'Phone Number *', value: phone, set: setPhone, icon: 'call-outline', type: 'phone-pad' },
        ].map((f, i) => (
          <View key={i} style={styles.inputWrapper}>
            <Ionicons name={f.icon as any} size={20} color={Colors.textLight} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder={f.placeholder}
              value={f.value}
              onChangeText={f.set}
              keyboardType={f.type as any}
              autoCapitalize={f.type === 'email-address' ? 'none' : 'words'}
              placeholderTextColor={Colors.textLight}
            />
          </View>
        ))}

        <View style={styles.inputWrapper}>
          <Ionicons name="lock-closed-outline" size={20} color={Colors.textLight} style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Password * (min 6 characters)"
            value={password}
            onChangeText={setPassword}
            secureTextEntry={!showPass}
            placeholderTextColor={Colors.textLight}
          />
          <TouchableOpacity onPress={() => setShowPass(v => !v)}>
            <Ionicons name={showPass ? 'eye-off-outline' : 'eye-outline'} size={20} color={Colors.textLight} />
          </TouchableOpacity>
        </View>

        <View style={styles.inputWrapper}>
          <Ionicons name="pricetag-outline" size={20} color={Colors.textLight} style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Referral Code (Optional)"
            value={referralCode}
            onChangeText={setReferralCode}
            autoCapitalize="characters"
            placeholderTextColor={Colors.textLight}
          />
        </View>

        {referralCode.length > 0 && (
          <View style={styles.referralInfo}>
            <Ionicons name="gift-outline" size={16} color={Colors.success} />
            <Text style={styles.referralInfoText}>+200 bonus points will be added!</Text>
          </View>
        )}

        <TouchableOpacity style={styles.termsRow} onPress={() => setAgreed(v => !v)}>
          <View style={[styles.checkbox, agreed && styles.checkboxActive]}>
            {agreed && <Ionicons name="checkmark" size={14} color={Colors.white} />}
          </View>
          <Text style={styles.termsText}>
            I agree to the{' '}
            <Text style={styles.termsLink}>Terms & Conditions</Text> and{' '}
            <Text style={styles.termsLink}>Privacy Policy</Text>
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.registerBtn} onPress={handleRegister} disabled={loading}>
          <LinearGradient colors={[Colors.secondary, Colors.primary]} style={styles.registerBtnGradient}>
            {loading ? (
              <ActivityIndicator color={Colors.white} />
            ) : (
              <Text style={styles.registerBtnText}>Create Account 🚀</Text>
            )}
          </LinearGradient>
        </TouchableOpacity>

        <View style={styles.loginRow}>
          <Text style={styles.loginText}>Already have an account? </Text>
          <TouchableOpacity onPress={() => navigation.replace('Login')}>
            <Text style={styles.loginLink}>Sign In →</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  header: { paddingTop: 52, paddingBottom: 24, paddingHorizontal: Spacing.md, alignItems: 'center' },
  closeBtn: { position: 'absolute', top: 52, right: 16, width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  logo: { fontSize: 40, marginBottom: 8 },
  brandName: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900', letterSpacing: 3, marginBottom: 4 },
  tagline: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md },
  form: { flex: 1, padding: Spacing.lg, backgroundColor: Colors.background },
  welcomeText: { fontSize: FontSize.xxl, fontWeight: '900', color: Colors.textPrimary, marginBottom: 4 },
  welcomeDesc: { fontSize: FontSize.md, color: Colors.textSecondary, marginBottom: 24 },
  inputWrapper: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.card, borderRadius: BorderRadius.md, borderWidth: 1.5, borderColor: Colors.border, paddingHorizontal: 14, marginBottom: 12 },
  inputIcon: { marginRight: 10 },
  input: { flex: 1, paddingVertical: 14, fontSize: FontSize.md, color: Colors.textPrimary },
  referralInfo: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: Colors.success + '15', borderRadius: BorderRadius.sm, padding: 10, marginBottom: 12 },
  referralInfoText: { color: Colors.success, fontWeight: '700', fontSize: FontSize.sm },
  termsRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 20 },
  checkbox: { width: 22, height: 22, borderRadius: 6, borderWidth: 2, borderColor: Colors.border, alignItems: 'center', justifyContent: 'center', marginTop: 2 },
  checkboxActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  termsText: { flex: 1, fontSize: FontSize.sm, color: Colors.textSecondary, lineHeight: 20 },
  termsLink: { color: Colors.primary, fontWeight: '700' },
  registerBtn: { borderRadius: BorderRadius.lg, overflow: 'hidden', marginBottom: 20 },
  registerBtnGradient: { paddingVertical: 18, alignItems: 'center', borderRadius: BorderRadius.lg },
  registerBtnText: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
  loginRow: { flexDirection: 'row', justifyContent: 'center', paddingBottom: 32 },
  loginText: { color: Colors.textSecondary, fontSize: FontSize.md },
  loginLink: { color: Colors.primary, fontWeight: '800', fontSize: FontSize.md },
});

export default RegisterScreen;
