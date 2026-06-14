module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      [
        'module-resolver',
        {
          root: ['.'],
          extensions: ['.ios.js', '.android.js', '.js', '.ts', '.tsx', '.json', '.jsx'],
          alias: {
            '@': '.',
            '@/components': './components',
            '@/lib': './lib',
            '@/stores': './stores',
            '@/constants': './constants',
            '@/types': './types',
            '@/assets': './assets',
            '@/app': './app',
            '@/hooks': './hooks',
          },
        },
      ],
      'react-native-reanimated/plugin',
    ],
  };
};
