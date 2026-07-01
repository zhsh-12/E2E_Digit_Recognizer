/**
 * Tests for App.vue — tab switching and language switching
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import App from '../../src/App.vue'

// Mock child components to simplify testing
vi.mock('../../src/components/SinglePredict.vue', () => ({
  default: {
    name: 'SinglePredict',
    template: '<div class="mock-single">SinglePredict</div>',
  },
}))
vi.mock('../../src/components/BatchPredict.vue', () => ({
  default: {
    name: 'BatchPredict',
    template: '<div class="mock-batch">BatchPredict</div>',
  },
}))
vi.mock('../../src/components/History.vue', () => ({
  default: {
    name: 'History',
    template: '<div class="mock-history">History</div>',
  },
}))

const i18n = createI18n({
  legacy: false,
  locale: 'zh-CN',
  fallbackLocale: 'en',
  messages: {
    'zh-CN': {
      app: { title: '数字识别系统' },
      nav: { single: '单张预测', batch: '批量预测', history: '历史记录' },
    },
    en: {
      app: { title: 'Digit Recognition System' },
      nav: { single: 'Single Prediction', batch: 'Batch Prediction', history: 'History' },
    },
  },
})

function mountApp() {
  return mount(App, {
    global: {
      plugins: [i18n],
    },
  })
}

describe('App.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the app title', () => {
    const wrapper = mountApp()
    expect(wrapper.text()).toContain('数字识别系统')
  })

  it('shows SinglePredict by default', () => {
    const wrapper = mountApp()
    expect(wrapper.find('.mock-single').exists()).toBe(true)
    expect(wrapper.find('.mock-batch').exists()).toBe(false)
    expect(wrapper.find('.mock-history').exists()).toBe(false)
  })

  it('switches to BatchPredict tab on batch button click', async () => {
    const wrapper = mountApp()
    const buttons = wrapper.findAll('.nav-btn')
    // Second button is "批量预测"
    await buttons[1].trigger('click')

    expect(wrapper.find('.mock-single').exists()).toBe(false)
    expect(wrapper.find('.mock-batch').exists()).toBe(true)
    expect(wrapper.find('.mock-history').exists()).toBe(false)
  })

  it('switches to History tab on history button click', async () => {
    const wrapper = mountApp()
    const buttons = wrapper.findAll('.nav-btn')
    // Third button is "历史记录"
    await buttons[2].trigger('click')

    expect(wrapper.find('.mock-single').exists()).toBe(false)
    expect(wrapper.find('.mock-batch').exists()).toBe(false)
    expect(wrapper.find('.mock-history').exists()).toBe(true)
  })

  it('highlights active tab', async () => {
    const wrapper = mountApp()
    const buttons = wrapper.findAll('.nav-btn')

    // Default: single is active
    expect(buttons[0].classes()).toContain('active')
    expect(buttons[1].classes()).not.toContain('active')
    expect(buttons[2].classes()).not.toContain('active')

    // Click batch
    await buttons[1].trigger('click')
    expect(buttons[0].classes()).not.toContain('active')
    expect(buttons[1].classes()).toContain('active')
    expect(buttons[2].classes()).not.toContain('active')
  })

  it('toggles language on lang switch button click', async () => {
    const wrapper = mountApp()
    const langBtn = wrapper.find('.lang-switch')

    // Default is zh-CN
    expect(wrapper.text()).toContain('单张预测')

    // Click to switch to EN
    await langBtn.trigger('click')
    expect(wrapper.text()).toContain('Single Prediction')

    // Click again to switch back to zh-CN
    await langBtn.trigger('click')
    expect(wrapper.text()).toContain('单张预测')
  })
})
