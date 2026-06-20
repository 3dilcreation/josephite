import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  TextInput, KeyboardAvoidingView, Platform, Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';
import { CHAT_MESSAGES } from '../data/mockData';

const QUICK_REPLIES = [
  '✂️ I want a taper fade',
  '💈 Can I come in now?',
  '🕐 What time are you free?',
  '📸 Can I show you a pic?',
  '💰 What\'s the price?',
];

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const scrollRef = useRef(null);
  const [messages, setMessages] = useState(CHAT_MESSAGES);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);

  const send = (text = input) => {
    if (!text.trim()) return;
    const newMsg = {
      id: `m${Date.now()}`,
      sender: 'user',
      text: text.trim(),
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setMessages(m => [...m, newMsg]);
    setInput('');

    // Simulate barber typing
    setTyping(true);
    setTimeout(() => {
      setTyping(false);
      const reply = {
        id: `m${Date.now() + 1}`,
        sender: 'barber',
        name: 'Marcus',
        text: getAutoReply(text),
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        avatar: 'https://i.pravatar.cc/150?img=12',
      };
      setMessages(m => [...m, reply]);
      scrollRef.current?.scrollToEnd({ animated: true });
    }, 1500);

    scrollRef.current?.scrollToEnd({ animated: true });
  };

  const getAutoReply = (text) => {
    const t = text.toLowerCase();
    if (t.includes('fade')) return "I'm great with fades! I'll prep the guards. See you tomorrow! 💈";
    if (t.includes('price') || t.includes('cost')) return "Haircut starts at $28, fade from $35, full combo $50. All worth it! 🏆";
    if (t.includes('time') || t.includes('free') || t.includes('available')) return "I have a slot open at 10:30 AM and 2:00 PM tomorrow. Which works for you?";
    if (t.includes('pic') || t.includes('photo') || t.includes('image')) return "Yes! Send it over and I'll let you know if we can replicate it. Usually no problem 💪";
    return "Got it! Looking forward to seeing you 💈 Let me know if you need anything else.";
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={[styles.container, { paddingTop: insets.top }]}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
            <MaterialCommunityIcons name="arrow-left" size={22} color={Colors.textPrimary} />
          </TouchableOpacity>

          <View style={styles.barberHeader}>
            <View style={styles.barberAvatarWrap}>
              <Image source={{ uri: 'https://i.pravatar.cc/150?img=12' }} style={styles.barberAvatar} />
              <View style={styles.onlineDot} />
            </View>
            <View>
              <Text style={styles.barberName}>Marcus "The Blade"</Text>
              <Text style={styles.barberStatus}>
                {typing ? '✏️ typing...' : 'Online · Usually replies in minutes'}
              </Text>
            </View>
          </View>

          <TouchableOpacity style={styles.callBtn}>
            <MaterialCommunityIcons name="video-outline" size={22} color={Colors.primary} />
          </TouchableOpacity>
        </View>

        {/* Appointment Banner */}
        <TouchableOpacity style={styles.apptBanner} activeOpacity={0.85}>
          <LinearGradient colors={Colors.gradientGold} style={StyleSheet.absoluteFill} borderRadius={Radius.md} />
          <MaterialCommunityIcons name="calendar-check" size={18} color="#0A0A0F" />
          <Text style={styles.apptBannerText}>Tomorrow · 10:30 AM · Mid Taper Fade</Text>
          <MaterialCommunityIcons name="chevron-right" size={16} color="#0A0A0F60" />
        </TouchableOpacity>

        {/* Messages */}
        <ScrollView
          ref={scrollRef}
          style={styles.messages}
          contentContainerStyle={styles.messagesContent}
          showsVerticalScrollIndicator={false}
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        >
          {/* Date separator */}
          <View style={styles.dateSep}>
            <View style={styles.dateSepLine} />
            <Text style={styles.dateSepText}>Today</Text>
            <View style={styles.dateSepLine} />
          </View>

          {messages.map((msg) => (
            <View
              key={msg.id}
              style={[styles.msgRow, msg.sender === 'user' && styles.msgRowUser]}
            >
              {msg.sender === 'barber' && (
                <Image source={{ uri: msg.avatar }} style={styles.msgAvatar} />
              )}
              <View style={[
                styles.msgBubble,
                msg.sender === 'user' ? styles.msgBubbleUser : styles.msgBubbleBarber,
              ]}>
                {msg.sender === 'user' ? (
                  <LinearGradient
                    colors={Colors.gradientGold}
                    style={StyleSheet.absoluteFill}
                    borderRadius={Radius.lg}
                  />
                ) : (
                  <LinearGradient
                    colors={Colors.gradientCard}
                    style={StyleSheet.absoluteFill}
                    borderRadius={Radius.lg}
                  />
                )}
                <Text style={[
                  styles.msgText,
                  msg.sender === 'user' && styles.msgTextUser,
                ]}>
                  {msg.text}
                </Text>
                <Text style={[styles.msgTime, msg.sender === 'user' && styles.msgTimeUser]}>
                  {msg.time}
                </Text>
              </View>
            </View>
          ))}

          {typing && (
            <View style={styles.msgRow}>
              <View style={styles.msgAvatarPlaceholder} />
              <View style={[styles.msgBubble, styles.msgBubbleBarber, styles.typingBubble]}>
                <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.lg} />
                <View style={styles.typingDots}>
                  {[0, 1, 2].map(i => (
                    <View key={i} style={styles.typingDot} />
                  ))}
                </View>
              </View>
            </View>
          )}
        </ScrollView>

        {/* Quick Replies */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.quickReplies}
          contentContainerStyle={styles.quickRepliesContent}
        >
          {QUICK_REPLIES.map((qr) => (
            <TouchableOpacity key={qr} style={styles.quickReply} onPress={() => send(qr)} activeOpacity={0.8}>
              <Text style={styles.quickReplyText}>{qr}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Input */}
        <View style={[styles.inputArea, { paddingBottom: insets.bottom + Spacing.sm }]}>
          <TouchableOpacity style={styles.attachBtn}>
            <MaterialCommunityIcons name="image-plus" size={22} color={Colors.textSecondary} />
          </TouchableOpacity>
          <TextInput
            style={styles.textInput}
            placeholder="Message your barber..."
            placeholderTextColor={Colors.textMuted}
            value={input}
            onChangeText={setInput}
            multiline
            maxLength={500}
          />
          <TouchableOpacity
            style={[styles.sendBtn, input.trim() && styles.sendBtnActive]}
            onPress={() => send()}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={input.trim() ? Colors.gradientGold : [Colors.bgCardAlt, Colors.bgCardAlt]}
              style={styles.sendBtnGrad}
            >
              <MaterialCommunityIcons
                name="send"
                size={18}
                color={input.trim() ? '#0A0A0F' : Colors.textMuted}
              />
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    gap: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
  },
  backBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.bgGlass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  barberHeader: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 10 },
  barberAvatarWrap: { position: 'relative' },
  barberAvatar: { width: 40, height: 40, borderRadius: 20, borderWidth: 2, borderColor: Colors.border },
  onlineDot: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 11,
    height: 11,
    borderRadius: 6,
    backgroundColor: Colors.success,
    borderWidth: 2,
    borderColor: Colors.bg,
  },
  barberName: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700' },
  barberStatus: { color: Colors.textSecondary, fontSize: 11, marginTop: 1 },
  callBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.bgGlass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  apptBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginHorizontal: Spacing.md,
    marginVertical: 8,
    padding: Spacing.sm + 4,
    borderRadius: Radius.md,
    overflow: 'hidden',
  },
  apptBannerText: { flex: 1, color: '#0A0A0F', fontSize: 13, fontWeight: '700' },
  messages: { flex: 1 },
  messagesContent: { padding: Spacing.md, gap: 8 },
  dateSep: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginVertical: 8,
  },
  dateSepLine: { flex: 1, height: 1, backgroundColor: Colors.borderSubtle },
  dateSepText: { color: Colors.textMuted, fontSize: 11 },
  msgRow: { flexDirection: 'row', alignItems: 'flex-end', gap: 8, marginBottom: 4 },
  msgRowUser: { flexDirection: 'row-reverse' },
  msgAvatar: { width: 28, height: 28, borderRadius: 14 },
  msgAvatarPlaceholder: { width: 28 },
  msgBubble: {
    maxWidth: '75%',
    padding: 12,
    borderRadius: Radius.lg,
    overflow: 'hidden',
    position: 'relative',
  },
  msgBubbleUser: { borderBottomRightRadius: 4 },
  msgBubbleBarber: { borderBottomLeftRadius: 4 },
  msgText: { color: Colors.textPrimary, fontSize: 14, lineHeight: 20 },
  msgTextUser: { color: '#0A0A0F' },
  msgTime: { color: Colors.textMuted, fontSize: 10, marginTop: 4, textAlign: 'right' },
  msgTimeUser: { color: '#0A0A0F60' },
  typingBubble: { paddingVertical: 14, paddingHorizontal: 16 },
  typingDots: { flexDirection: 'row', gap: 4, alignItems: 'center' },
  typingDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: Colors.textMuted },
  quickReplies: { borderTopWidth: 1, borderTopColor: Colors.borderSubtle },
  quickRepliesContent: { padding: 8, gap: 6 },
  quickReply: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: Radius.full,
    backgroundColor: Colors.bgCardAlt,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  quickReplyText: { color: Colors.textPrimary, fontSize: 13 },
  inputArea: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
    paddingHorizontal: Spacing.md,
    paddingTop: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
    backgroundColor: Colors.bg,
  },
  attachBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  textInput: {
    flex: 1,
    color: Colors.textPrimary,
    fontSize: 15,
    backgroundColor: Colors.bgCardAlt,
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 10,
    maxHeight: 100,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
  },
  sendBtn: { width: 40, height: 40, borderRadius: 20, overflow: 'hidden' },
  sendBtnActive: {},
  sendBtnGrad: { flex: 1, alignItems: 'center', justifyContent: 'center' },
});
