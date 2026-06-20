import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors } from '../theme/colors';

import HomeScreen from '../screens/HomeScreen';
import BookingScreen from '../screens/BookingScreen';
import SocialFeedScreen from '../screens/SocialFeedScreen';
import LoyaltyScreen from '../screens/LoyaltyScreen';
import ProfileScreen from '../screens/ProfileScreen';
import ARTryOnScreen from '../screens/ARTryOnScreen';
import QueueScreen from '../screens/QueueScreen';
import ChatScreen from '../screens/ChatScreen';
import SubscriptionScreen from '../screens/SubscriptionScreen';
import HairJourneyScreen from '../screens/HairJourneyScreen';
import HairAnalysisScreen from '../screens/HairAnalysisScreen';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function CustomTabBar({ state, descriptors, navigation }) {
  const tabs = [
    { name: 'Home', icon: 'home-variant', iconActive: 'home-variant' },
    { name: 'Book', icon: 'calendar-plus', iconActive: 'calendar-plus' },
    { name: 'Feed', icon: 'view-dashboard', iconActive: 'view-dashboard' },
    { name: 'Loyalty', icon: 'crown-outline', iconActive: 'crown' },
    { name: 'Profile', icon: 'account-circle-outline', iconActive: 'account-circle' },
  ];

  return (
    <View style={styles.tabBar}>
      {state.routes.map((route, index) => {
        const focused = state.index === index;
        const tab = tabs[index];

        return (
          <TouchableOpacity
            key={route.key}
            style={styles.tabItem}
            onPress={() => navigation.navigate(route.name)}
            activeOpacity={0.7}
          >
            {focused && <View style={styles.tabActiveIndicator} />}
            <MaterialCommunityIcons
              name={focused ? tab.iconActive : tab.icon}
              size={24}
              color={focused ? Colors.primary : Colors.textMuted}
            />
            <Text style={[styles.tabLabel, focused && styles.tabLabelActive]}>
              {route.name}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

function HomeTabs() {
  return (
    <Tab.Navigator
      tabBar={(props) => <CustomTabBar {...props} />}
      screenOptions={{ headerShown: false }}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Book" component={BookingScreen} />
      <Tab.Screen name="Feed" component={SocialFeedScreen} />
      <Tab.Screen name="Loyalty" component={LoyaltyScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Main" component={HomeTabs} />
      <Stack.Screen name="ARTryOn" component={ARTryOnScreen} />
      <Stack.Screen name="Queue" component={QueueScreen} />
      <Stack.Screen name="Chat" component={ChatScreen} />
      <Stack.Screen name="Subscription" component={SubscriptionScreen} />
      <Stack.Screen name="HairJourney" component={HairJourneyScreen} />
      <Stack.Screen name="HairAnalysis" component={HairAnalysisScreen} />
    </Stack.Navigator>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    flexDirection: 'row',
    backgroundColor: Colors.bgCard,
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
    paddingBottom: 20,
    paddingTop: 10,
    paddingHorizontal: 8,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  tabActiveIndicator: {
    position: 'absolute',
    top: -10,
    width: 32,
    height: 3,
    backgroundColor: Colors.primary,
    borderRadius: 2,
  },
  tabLabel: {
    fontSize: 10,
    color: Colors.textMuted,
    marginTop: 4,
    fontWeight: '500',
  },
  tabLabelActive: {
    color: Colors.primary,
    fontWeight: '700',
  },
});
