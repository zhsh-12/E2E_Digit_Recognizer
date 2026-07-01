import { createApp } from 'vue'
import { createI18n } from 'vue-i18n'
import App from './App.vue'
import './style.css'

import zhCN from './locales/zh-CN.json'
import en from './locales/en.json'

const i18n = createI18n({
  legacy: false, 
  locale: 'zh-CN', // Default language
  fallbackLocale: 'en',
  messages: {
    'zh-CN': zhCN,
    'en': en,
  },
})

const app = createApp(App)
app.use(i18n)
app.mount('#app')
