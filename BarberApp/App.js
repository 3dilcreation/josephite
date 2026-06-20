import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import AppNavigator from './src/navigation/AppNavigator';
import OnboardingScreen from './src/screens/OnboardingScreen';

export default function App() {
  const [onboarded, setOnboarded] = useState(false);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <StatusBar style="light" backgroundColor="#0A0A0F" />
        {onboarded ? (
          <NavigationContainer>
            <AppNavigator />
          </NavigationContainer>
        ) : (
          <OnboardingScreen onFinish={() => setOnboarded(true)} />
        )}
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
