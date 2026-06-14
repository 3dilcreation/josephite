import React, { createContext, useContext, useState, useCallback } from 'react';
import { User, LoyaltyTier } from '../types';

interface AuthContextType {
  user: User | null;
  isLoggedIn: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  register: (name: string, email: string, phone: string, password: string, referralCode?: string) => Promise<boolean>;
  updateProfile: (updates: Partial<User>) => void;
  loyaltyPoints: number;
  loyaltyTier: LoyaltyTier;
  addPoints: (points: number) => void;
  redeemPoints: (points: number) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const getTier = (points: number): LoyaltyTier => {
  if (points >= 5000) return 'Platinum';
  if (points >= 2000) return 'Gold';
  if (points >= 500) return 'Silver';
  return 'Bronze';
};

const MOCK_USER: User = {
  id: 'u1',
  name: 'Rajesh Kumar',
  email: 'rajesh@example.com',
  phone: '+91 98765 43210',
  loyaltyPoints: 1250,
  loyaltyTier: 'Silver',
  referralCode: '3DIL-RK1250',
  totalOrders: 8,
  memberSince: 'January 2024',
  savedAddresses: [
    {
      id: 'a1',
      label: 'Home',
      line1: '42, Shivaji Nagar',
      line2: 'Near Central Mall',
      city: 'Pune',
      state: 'Maharashtra',
      pincode: '411005',
      isDefault: true,
    },
    {
      id: 'a2',
      label: 'Office',
      line1: 'Tech Park, Tower B, 4th Floor',
      city: 'Pune',
      state: 'Maharashtra',
      pincode: '411045',
      isDefault: false,
    },
  ],
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  const login = useCallback(async (email: string, _password: string): Promise<boolean> => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    if (email && _password.length >= 6) {
      setUser({ ...MOCK_USER, email });
      return true;
    }
    return false;
  }, []);

  const logout = useCallback(() => {
    setUser(null);
  }, []);

  const register = useCallback(async (name: string, email: string, phone: string, _password: string, _referralCode?: string): Promise<boolean> => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    const newUser: User = {
      id: `u_${Date.now()}`,
      name,
      email,
      phone,
      loyaltyPoints: _referralCode ? 100 : 50,
      loyaltyTier: 'Bronze',
      referralCode: `3DIL-${name.substring(0, 2).toUpperCase()}${Date.now().toString().slice(-4)}`,
      totalOrders: 0,
      memberSince: new Date().toLocaleDateString('en-IN', { month: 'long', year: 'numeric' }),
      savedAddresses: [],
    };
    setUser(newUser);
    return true;
  }, []);

  const updateProfile = useCallback((updates: Partial<User>) => {
    setUser(prev => prev ? { ...prev, ...updates } : null);
  }, []);

  const addPoints = useCallback((points: number) => {
    setUser(prev => {
      if (!prev) return null;
      const newPoints = prev.loyaltyPoints + points;
      return { ...prev, loyaltyPoints: newPoints, loyaltyTier: getTier(newPoints) };
    });
  }, []);

  const redeemPoints = useCallback((points: number): boolean => {
    if (!user || user.loyaltyPoints < points) return false;
    setUser(prev => {
      if (!prev) return null;
      const newPoints = prev.loyaltyPoints - points;
      return { ...prev, loyaltyPoints: newPoints, loyaltyTier: getTier(newPoints) };
    });
    return true;
  }, [user]);

  return (
    <AuthContext.Provider value={{
      user,
      isLoggedIn: !!user,
      login,
      logout,
      register,
      updateProfile,
      loyaltyPoints: user?.loyaltyPoints || 0,
      loyaltyTier: user?.loyaltyTier || 'Bronze',
      addPoints,
      redeemPoints,
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};
