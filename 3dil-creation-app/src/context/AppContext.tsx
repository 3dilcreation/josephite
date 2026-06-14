import React, { createContext, useContext, useState } from 'react';
import { CartProvider } from './CartContext';
import { AuthProvider } from './AuthContext';

interface AppContextType {
  notificationsEnabled: boolean;
  setNotificationsEnabled: (v: boolean) => void;
  isDarkMode: boolean;
  setIsDarkMode: (v: boolean) => void;
  appVersion: string;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);

  return (
    <AppContext.Provider value={{
      notificationsEnabled,
      setNotificationsEnabled,
      isDarkMode,
      setIsDarkMode,
      appVersion: '1.0.0',
    }}>
      <AuthProvider>
        <CartProvider>
          {children}
        </CartProvider>
      </AuthProvider>
    </AppContext.Provider>
  );
};

export const useApp = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within AppProvider');
  return context;
};
