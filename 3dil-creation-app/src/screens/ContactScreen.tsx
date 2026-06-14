import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  TextInput, Linking, Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';

const ContactScreen: React.FC = () => {
  const navigation = useNavigation();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [subject, setSubject] = useState('');
  const [sending, setSending] = useState(false);

  const handleSend = async () => {
    if (!name || !email || !message) {
      Alert.alert('Missing Info', 'Please fill in name, email and message.');
      return;
    }
    setSending(true);
    await new Promise(r => setTimeout(r, 1500));
    setSending(false);
    Alert.alert('✅ Message Sent!', 'We\'ll get back to you within 2 business hours.');
    setName(''); setEmail(''); setMessage(''); setSubject('');
  };

  const contactInfo = [
    { icon: 'call', label: 'Phone', value: '+91 99999 99999', action: () => Linking.openURL('tel:+919999999999') },
    { icon: 'mail', label: 'Email', value: 'hello@3dilcreation.in', action: () => Linking.openURL('mailto:hello@3dilcreation.in') },
    { icon: 'location', label: 'Address', value: 'Pune, Maharashtra, India 411001', action: () => {} },
    { icon: 'time', label: 'Hours', value: 'Mon–Sat: 9 AM – 7 PM IST', action: () => {} },
  ];

  const socials = [
    { icon: '📘', label: 'Facebook', url: 'https://facebook.com' },
    { icon: '📷', label: 'Instagram', url: 'https://instagram.com' },
    { icon: '▶️', label: 'YouTube', url: 'https://youtube.com' },
    { icon: '🐦', label: 'Twitter', url: 'https://twitter.com' },
  ];

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.secondary, Colors.primary]} style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Contact Us</Text>
        <Text style={styles.headerSubtitle}>We typically reply within 2 hours</Text>
      </LinearGradient>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* WhatsApp CTA */}
        <TouchableOpacity
          style={styles.whatsappCard}
          onPress={() => Linking.openURL('https://wa.me/919999999999?text=Hi 3DIL Creation, I need help with a 3D printing order.')}
        >
          <LinearGradient colors={['#25D366', '#128C7E']} style={styles.whatsappGradient}>
            <Text style={styles.whatsappEmoji}>💬</Text>
            <View>
              <Text style={styles.whatsappTitle}>Chat on WhatsApp</Text>
              <Text style={styles.whatsappDesc}>Fastest response — usually within minutes</Text>
            </View>
            <Ionicons name="arrow-forward" size={24} color={Colors.white} />
          </LinearGradient>
        </TouchableOpacity>

        {/* Contact Info */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Contact Information</Text>
          {contactInfo.map((item, i) => (
            <TouchableOpacity key={i} style={styles.contactRow} onPress={item.action}>
              <View style={styles.contactIcon}>
                <Ionicons name={item.icon as any} size={22} color={Colors.primary} />
              </View>
              <View>
                <Text style={styles.contactLabel}>{item.label}</Text>
                <Text style={styles.contactValue}>{item.value}</Text>
              </View>
              <Ionicons name="chevron-forward" size={16} color={Colors.textLight} style={{ marginLeft: 'auto' }} />
            </TouchableOpacity>
          ))}
        </View>

        {/* Map Placeholder */}
        <View style={styles.mapPlaceholder}>
          <LinearGradient colors={['#E8F5E9', '#C8E6C9']} style={styles.mapGradient}>
            <Text style={styles.mapEmoji}>🗺️</Text>
            <Text style={styles.mapText}>3DIL Creation, Pune</Text>
            <Text style={styles.mapSubtext}>Maharashtra, India</Text>
            <TouchableOpacity style={styles.directionsBtn}>
              <Text style={styles.directionsBtnText}>Get Directions →</Text>
            </TouchableOpacity>
          </LinearGradient>
        </View>

        {/* Contact Form */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Send Us a Message</Text>
          <TextInput
            style={styles.input}
            placeholder="Your Name"
            value={name}
            onChangeText={setName}
            placeholderTextColor={Colors.textLight}
          />
          <TextInput
            style={styles.input}
            placeholder="Email Address"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            placeholderTextColor={Colors.textLight}
          />
          <TextInput
            style={styles.input}
            placeholder="Subject (Optional)"
            value={subject}
            onChangeText={setSubject}
            placeholderTextColor={Colors.textLight}
          />
          <TextInput
            style={styles.textArea}
            placeholder="Your message..."
            value={message}
            onChangeText={setMessage}
            multiline
            numberOfLines={5}
            placeholderTextColor={Colors.textLight}
            textAlignVertical="top"
          />
          <TouchableOpacity style={styles.sendBtn} onPress={handleSend} disabled={sending}>
            <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.sendBtnGradient}>
              <Ionicons name="send" size={18} color={Colors.white} />
              <Text style={styles.sendBtnText}>{sending ? 'Sending...' : 'Send Message'}</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>

        {/* Social Media */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Follow Us</Text>
          <View style={styles.socialsGrid}>
            {socials.map((s, i) => (
              <TouchableOpacity key={i} style={styles.socialCard} onPress={() => Linking.openURL(s.url)}>
                <Text style={styles.socialEmoji}>{s.icon}</Text>
                <Text style={styles.socialLabel}>{s.label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 20, paddingHorizontal: Spacing.md },
  backBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', marginBottom: 10 },
  headerTitle: { color: Colors.white, fontSize: FontSize.xxxl, fontWeight: '900', marginBottom: 4 },
  headerSubtitle: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md },
  whatsappCard: { margin: 12, borderRadius: BorderRadius.lg, overflow: 'hidden', ...Shadows.medium },
  whatsappGradient: { flexDirection: 'row', alignItems: 'center', padding: 20, gap: 12, borderRadius: BorderRadius.lg },
  whatsappEmoji: { fontSize: 36 },
  whatsappTitle: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800', marginBottom: 2 },
  whatsappDesc: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.sm },
  section: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  sectionTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 14 },
  contactRow: { flexDirection: 'row', alignItems: 'center', gap: 14, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: Colors.border },
  contactIcon: { width: 44, height: 44, borderRadius: 22, backgroundColor: Colors.primary + '15', alignItems: 'center', justifyContent: 'center' },
  contactLabel: { fontSize: FontSize.xs, color: Colors.textSecondary, marginBottom: 2 },
  contactValue: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  mapPlaceholder: { margin: 12, borderRadius: BorderRadius.md, overflow: 'hidden' },
  mapGradient: { padding: 32, alignItems: 'center', borderRadius: BorderRadius.md },
  mapEmoji: { fontSize: 56, marginBottom: 8 },
  mapText: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.secondary, marginBottom: 4 },
  mapSubtext: { fontSize: FontSize.md, color: Colors.textSecondary, marginBottom: 16 },
  directionsBtn: { backgroundColor: Colors.secondary, paddingHorizontal: 20, paddingVertical: 10, borderRadius: 20 },
  directionsBtnText: { color: Colors.white, fontWeight: '700', fontSize: FontSize.md },
  input: { borderWidth: 1.5, borderColor: Colors.border, borderRadius: BorderRadius.md, paddingHorizontal: 14, paddingVertical: 12, fontSize: FontSize.md, color: Colors.textPrimary, marginBottom: 10 },
  textArea: { borderWidth: 1.5, borderColor: Colors.border, borderRadius: BorderRadius.md, paddingHorizontal: 14, paddingVertical: 12, fontSize: FontSize.md, color: Colors.textPrimary, marginBottom: 16, minHeight: 100 },
  sendBtn: { borderRadius: BorderRadius.lg, overflow: 'hidden' },
  sendBtnGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, gap: 8, borderRadius: BorderRadius.lg },
  sendBtnText: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '800' },
  socialsGrid: { flexDirection: 'row', gap: 10, flexWrap: 'wrap' },
  socialCard: { flex: 1, minWidth: 70, backgroundColor: Colors.background, borderRadius: BorderRadius.md, padding: 16, alignItems: 'center', gap: 6, borderWidth: 1.5, borderColor: Colors.border },
  socialEmoji: { fontSize: 28 },
  socialLabel: { fontSize: FontSize.xs, color: Colors.textSecondary, fontWeight: '700' },
});

export default ContactScreen;
