export interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  originalPrice: number;
  rating: number;
  reviewCount: number;
  material: string;
  colors: string[];
  sizes: string[];
  description: string;
  features: string[];
  inStock: boolean;
  isCustomizable: boolean;
  isBestseller: boolean;
  emoji: string;
}

export interface Service {
  id: string;
  name: string;
  description: string;
  icon: string;
  startingPrice: number;
  features: string[];
  category: string;
  turnaround: string;
  longDescription: string;
  process: string[];
  faqs: { question: string; answer: string }[];
}

export interface PortfolioItem {
  id: string;
  title: string;
  category: string;
  description: string;
  tags: string[];
  emoji: string;
  client?: string;
  completedDate?: string;
}

export type OrderStatus = 'placed' | 'designing' | 'printing' | 'quality_check' | 'shipped' | 'delivered' | 'cancelled';

export interface Order {
  id: string;
  productName: string;
  status: OrderStatus;
  placedDate: string;
  estimatedDelivery: string;
  totalAmount: number;
  quantity: number;
  items: CartItem[];
  trackingId?: string;
  address: string;
}

export interface CartItem {
  product: Product;
  quantity: number;
  selectedMaterial?: string;
  selectedColor?: string;
  selectedSize?: string;
  customNote?: string;
}

export type LoyaltyTier = 'Bronze' | 'Silver' | 'Gold' | 'Platinum';

export interface User {
  id: string;
  name: string;
  email: string;
  phone: string;
  avatar?: string;
  loyaltyPoints: number;
  loyaltyTier: LoyaltyTier;
  referralCode: string;
  totalOrders: number;
  memberSince: string;
  savedAddresses: Address[];
}

export interface Address {
  id: string;
  label: string;
  line1: string;
  line2?: string;
  city: string;
  state: string;
  pincode: string;
  isDefault: boolean;
}

export interface SubscriptionPlan {
  id: string;
  name: string;
  monthlyPrice: number;
  annualPrice: number;
  features: string[];
  freePrints: number;
  discount: number;
  color: string;
  isPopular: boolean;
}

export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  category: string;
  readTime: number;
  date: string;
  emoji: string;
  author: string;
}

export interface Testimonial {
  id: string;
  name: string;
  location: string;
  rating: number;
  review: string;
  service: string;
  avatar: string;
}

export interface QuoteRequest {
  serviceType: string;
  description: string;
  quantity: number;
  material: string;
  dimensions: { length: number; width: number; height: number };
  complexity: 'Simple' | 'Moderate' | 'Complex' | 'Highly Detailed';
  finish: 'Raw' | 'Sanded' | 'Painted' | 'Premium';
  referenceImage?: string;
  engravingText?: string;
  contactName: string;
  contactPhone: string;
  contactEmail: string;
  deadline?: string;
}
