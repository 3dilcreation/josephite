import React, { createContext, useContext, useState, useCallback } from 'react';
import { CartItem, Product } from '../types';

interface CartContextType {
  cartItems: CartItem[];
  addToCart: (product: Product, options?: Partial<CartItem>) => void;
  removeFromCart: (productId: string) => void;
  updateQuantity: (productId: string, quantity: number) => void;
  clearCart: () => void;
  cartTotal: number;
  cartCount: number;
  applyCoupon: (code: string) => boolean;
  discount: number;
  couponCode: string;
  removeCoupon: () => void;
}

const CartContext = createContext<CartContextType | undefined>(undefined);

const VALID_COUPONS: Record<string, number> = {
  'FIRST10': 10,
  'DIWALI20': 20,
  '3DIL15': 15,
  'WELCOME25': 25,
};

export const CartProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [couponCode, setCouponCode] = useState('');
  const [discount, setDiscount] = useState(0);

  const addToCart = useCallback((product: Product, options?: Partial<CartItem>) => {
    setCartItems(prev => {
      const existingIndex = prev.findIndex(
        item => item.product.id === product.id &&
          item.selectedMaterial === (options?.selectedMaterial || product.material) &&
          item.selectedColor === (options?.selectedColor || product.colors[0]) &&
          item.selectedSize === (options?.selectedSize || product.sizes[0])
      );
      if (existingIndex >= 0) {
        const updated = [...prev];
        updated[existingIndex] = {
          ...updated[existingIndex],
          quantity: updated[existingIndex].quantity + (options?.quantity || 1),
        };
        return updated;
      }
      return [...prev, {
        product,
        quantity: options?.quantity || 1,
        selectedMaterial: options?.selectedMaterial || product.material,
        selectedColor: options?.selectedColor || product.colors[0],
        selectedSize: options?.selectedSize || product.sizes[0],
        customNote: options?.customNote,
      }];
    });
  }, []);

  const removeFromCart = useCallback((productId: string) => {
    setCartItems(prev => prev.filter(item => item.product.id !== productId));
  }, []);

  const updateQuantity = useCallback((productId: string, quantity: number) => {
    if (quantity <= 0) {
      removeFromCart(productId);
      return;
    }
    setCartItems(prev =>
      prev.map(item =>
        item.product.id === productId ? { ...item, quantity } : item
      )
    );
  }, [removeFromCart]);

  const clearCart = useCallback(() => {
    setCartItems([]);
    setCouponCode('');
    setDiscount(0);
  }, []);

  const applyCoupon = useCallback((code: string): boolean => {
    const upperCode = code.toUpperCase().trim();
    if (VALID_COUPONS[upperCode]) {
      setCouponCode(upperCode);
      setDiscount(VALID_COUPONS[upperCode]);
      return true;
    }
    return false;
  }, []);

  const removeCoupon = useCallback(() => {
    setCouponCode('');
    setDiscount(0);
  }, []);

  const cartTotal = cartItems.reduce((sum, item) => sum + item.product.price * item.quantity, 0);
  const cartCount = cartItems.reduce((sum, item) => sum + item.quantity, 0);

  return (
    <CartContext.Provider value={{
      cartItems,
      addToCart,
      removeFromCart,
      updateQuantity,
      clearCart,
      cartTotal,
      cartCount,
      applyCoupon,
      discount,
      couponCode,
      removeCoupon,
    }}>
      {children}
    </CartContext.Provider>
  );
};

export const useCart = (): CartContextType => {
  const context = useContext(CartContext);
  if (!context) throw new Error('useCart must be used within CartProvider');
  return context;
};
