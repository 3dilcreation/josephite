import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { RootStackParamList } from './types';
import TabNavigator from './TabNavigator';

// Screens
import SplashScreen from '../screens/SplashScreen';
import OnboardingScreen from '../screens/OnboardingScreen';
import ProductDetailScreen from '../screens/ProductDetailScreen';
import ServiceDetailScreen from '../screens/ServiceDetailScreen';
import CustomOrderScreen from '../screens/CustomOrderScreen';
import QuoteCalculatorScreen from '../screens/QuoteCalculatorScreen';
import CartScreen from '../screens/CartScreen';
import CheckoutScreen from '../screens/CheckoutScreen';
import OrderTrackingScreen from '../screens/OrderTrackingScreen';
import ARViewerScreen from '../screens/ARViewerScreen';
import PortfolioScreen from '../screens/PortfolioScreen';
import BlogScreen from '../screens/BlogScreen';
import ContactScreen from '../screens/ContactScreen';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';
import SubscriptionScreen from '../screens/SubscriptionScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

const AppNavigator: React.FC = () => {
  return (
    <Stack.Navigator
      initialRouteName="Splash"
      screenOptions={{ headerShown: false }}
    >
      <Stack.Screen name="Splash" component={SplashScreen} />
      <Stack.Screen name="Onboarding" component={OnboardingScreen} />
      <Stack.Screen name="Main" component={TabNavigator} />
      <Stack.Screen
        name="ProductDetail"
        component={ProductDetailScreen}
        options={{ presentation: 'card' }}
      />
      <Stack.Screen
        name="ServiceDetail"
        component={ServiceDetailScreen}
        options={{ presentation: 'card' }}
      />
      <Stack.Screen
        name="CustomOrder"
        component={CustomOrderScreen}
        options={{ presentation: 'modal' }}
      />
      <Stack.Screen
        name="QuoteCalculator"
        component={QuoteCalculatorScreen}
        options={{ presentation: 'modal' }}
      />
      <Stack.Screen
        name="Cart"
        component={CartScreen}
        options={{ presentation: 'modal' }}
      />
      <Stack.Screen
        name="Checkout"
        component={CheckoutScreen}
        options={{ presentation: 'modal' }}
      />
      <Stack.Screen
        name="OrderTracking"
        component={OrderTrackingScreen}
        options={{ presentation: 'card' }}
      />
      <Stack.Screen
        name="ARViewer"
        component={ARViewerScreen}
        options={{ presentation: 'fullScreenModal' }}
      />
      <Stack.Screen
        name="Portfolio"
        component={PortfolioScreen}
        options={{ presentation: 'card' }}
      />
      <Stack.Screen
        name="Blog"
        component={BlogScreen}
        options={{ presentation: 'card' }}
      />
      <Stack.Screen
        name="Contact"
        component={ContactScreen}
        options={{ presentation: 'card' }}
      />
      <Stack.Screen
        name="Login"
        component={LoginScreen}
        options={{ presentation: 'modal' }}
      />
      <Stack.Screen
        name="Register"
        component={RegisterScreen}
        options={{ presentation: 'modal' }}
      />
      <Stack.Screen
        name="Subscription"
        component={SubscriptionScreen}
        options={{ presentation: 'modal' }}
      />
    </Stack.Navigator>
  );
};

export default AppNavigator;
