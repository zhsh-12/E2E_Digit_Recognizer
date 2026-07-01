import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

import path from 'path'

const frontendDir = path.resolve(__dirname)

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(frontendDir, 'src'),
    },
  },
  server: {
    port: 5173,
    fs: {
      allow: ['..', '.'],
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/batch_predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // https://vitest.dev/config/
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/**/*.{test,spec}.{js,ts}'],
    exclude: ['tests/e2e/**'],
    server: {
      deps: {
        fallbackCJS: true,
      },
    },
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{js,vue}'],
      exclude: ['src/main.js'],
    },
  },
})
