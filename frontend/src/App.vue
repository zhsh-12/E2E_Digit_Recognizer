<template>
  <div id="app-container">
    <div class="header-row">
      <h1 style="margin: 0; color: #1a1a1a;">{{ $t('app.title') }}</h1>
      <button class="lang-switch" @click="toggleLang">
        {{ locale === 'zh-CN' ? '🌐 EN' : '🌐 中文' }}
      </button>
    </div>

    <div class="nav-bar">
      <button
        class="nav-btn"
        :class="{ active: currentTab === 'single' }"
        @click="currentTab = 'single'"
      >
        {{ $t('nav.single') }}
      </button>
      <button
        class="nav-btn"
        :class="{ active: currentTab === 'batch' }"
        @click="currentTab = 'batch'"
      >
        {{ $t('nav.batch') }}
      </button>
      <button
        class="nav-btn"
        :class="{ active: currentTab === 'history' }"
        @click="currentTab = 'history'"
      >
        {{ $t('nav.history') }}
      </button>
    </div>

    <SinglePredict v-if="currentTab === 'single'" />
    <BatchPredict v-else-if="currentTab === 'batch'" />
    <History v-else />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import SinglePredict from './components/SinglePredict.vue'
import BatchPredict from './components/BatchPredict.vue'
import History from './components/History.vue'

const { locale } = useI18n()
const currentTab = ref('single')

function toggleLang() {
  locale.value = locale.value === 'zh-CN' ? 'en' : 'zh-CN'
}
</script>

<style scoped>
.header-row {
  position: relative;
  text-align: center;
  margin-bottom: 24px;
}
.lang-switch {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  padding: 6px 14px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}
.lang-switch:hover {
  border-color: #4096ff;
  color: #4096ff;
}
</style>

