// ============================================================
// TechSei LMS — Entry Point (auth-aware redirect)
// ============================================================
import { useEffect } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useAuthStore } from '../stores/authStore';

/**
 * Invisible splash-level redirect:
 *   • Not authenticated  → /auth/login
 *   • Student            → /student/home
 *   • Admin              → /admin/dashboard
 *
 * The ActivityIndicator is shown while `isLoading` is true (i.e. while the
 * auth store is re-hydrating the Supabase session from AsyncStorage).
 */
export default function Index() {
  const router = useRouter();
  const { isAuthenticated, isLoading, user } = useAuthStore();

  useEffect(() => {
    if (isLoading) return; // Wait for the session check to complete.

    if (!isAuthenticated || !user) {
      router.replace('/auth/login');
    } else if (user.role === 'admin') {
      router.replace('/admin/dashboard');
    } else {
      router.replace('/student/home');
    }
  }, [isAuthenticated, isLoading, user, router]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#6C63FF" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A1A',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
