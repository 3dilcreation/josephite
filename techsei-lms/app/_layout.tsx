// ============================================================
// TechSei LMS — Root Layout
// ============================================================
import { Stack } from 'expo-router';
import { useEffect, useCallback } from 'react';
import { useFonts } from 'expo-font';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { View, StyleSheet } from 'react-native';
import { useAuthStore } from '../stores/authStore';

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const { loadUser } = useAuthStore();

  const [fontsLoaded, fontError] = useFonts({
    // Extend here with custom fonts when added to assets/fonts/
  });

  useEffect(() => {
    // Re-hydrate the persisted auth session on every cold start.
    loadUser();
  }, [loadUser]);

  const onLayoutRootView = useCallback(async () => {
    if (fontsLoaded || fontError) {
      await SplashScreen.hideAsync();
    }
  }, [fontsLoaded, fontError]);

  // Hold render until fonts are ready (or errored) to prevent font flash.
  if (!fontsLoaded && !fontError) {
    return null;
  }

  return (
    <GestureHandlerRootView style={styles.root}>
      <StatusBar style="light" backgroundColor="#0A0A1A" />
      <View style={styles.root} onLayout={onLayoutRootView}>
        <Stack screenOptions={{ headerShown: false, animation: 'fade' }}>
          {/* Entry redirect */}
          <Stack.Screen name="index" options={{ headerShown: false }} />

          {/* Auth flow */}
          <Stack.Screen name="auth/login" options={{ headerShown: false, animation: 'fade' }} />
          <Stack.Screen name="auth/signup" options={{ headerShown: false, animation: 'slide_from_right' }} />
          <Stack.Screen name="auth/onboarding" options={{ headerShown: false, animation: 'slide_from_right' }} />

          {/* Student tab navigator */}
          <Stack.Screen name="student" options={{ headerShown: false }} />

          {/* Admin stack */}
          <Stack.Screen name="admin" options={{ headerShown: false }} />
        </Stack>
      </View>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#0A0A1A',
  },
});
