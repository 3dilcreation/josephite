import { NavigatorScreenParams } from '@react-navigation/native';

export type TabParamList = {
  Home: undefined;
  Products: undefined;
  Orders: undefined;
  Loyalty: undefined;
  Profile: undefined;
};

export type RootStackParamList = {
  Splash: undefined;
  Onboarding: undefined;
  Main: NavigatorScreenParams<TabParamList>;
  ProductDetail: { productId: string };
  ServiceDetail: { serviceId: string };
  CustomOrder: { serviceId?: string };
  QuoteCalculator: undefined;
  Cart: undefined;
  Checkout: undefined;
  OrderTracking: { orderId?: string };
  ARViewer: { productId?: string };
  Portfolio: undefined;
  Blog: undefined;
  BlogDetail: { postId: string };
  Contact: undefined;
  Login: undefined;
  Register: undefined;
  Subscription: undefined;
};
