/**
 * Tests for History.vue component
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import History from '../../src/components/History.vue'

// Mock the API module
vi.mock('../../src/api/predict.js', () => ({
  getHistory: vi.fn(),
}))

import { getHistory } from '../../src/api/predict.js'

const i18n = createI18n({
  legacy: false,
  locale: 'zh-CN',
  fallbackLocale: 'en',
  messages: {
    'zh-CN': {
      history: {
        title: '预测历史记录',
        loading: '加载中...',
        empty: '暂无历史记录',
        total: '共 {count} 条记录',
        id: 'ID',
        filename: '文件名',
        prediction: '预测值',
        confidence: '置信度',
        batch: '批次',
        time: '时间',
      },
    },
    en: {
      history: {
        title: 'Prediction History',
        loading: 'Loading...',
        empty: 'No records yet',
        total: '{count} records total',
        id: 'ID',
        filename: 'Filename',
        prediction: 'Prediction',
        confidence: 'Confidence',
        batch: 'Batch',
        time: 'Time',
      },
    },
  },
})

function mountComponent() {
  return mount(History, {
    global: {
      plugins: [i18n],
    },
  })
}

describe('History.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the title', () => {
    getHistory.mockResolvedValue({ records: [], count: 0 })
    const wrapper = mountComponent()
    expect(wrapper.text()).toContain('预测历史记录')
  })

  it('shows loading state initially', () => {
    // Don't resolve the promise to keep loading state
    getHistory.mockReturnValue(new Promise(() => {}))

    const wrapper = mountComponent()
    expect(wrapper.text()).toContain('加载中')
  })

  it('shows empty state when no records', async () => {
    getHistory.mockResolvedValue({ records: [], count: 0 })

    const wrapper = mountComponent()
    await new Promise(process.nextTick)

    expect(wrapper.text()).toContain('暂无历史记录')
  })

  it('renders records in a table', async () => {
    const mockRecords = [
      {
        id: 1,
        filename: 'test.png',
        predicted_label: 5,
        confidence: 0.95,
        batch_id: null,
        created_at: '2026-06-30 12:00:00',
      },
      {
        id: 2,
        filename: 'img.png',
        predicted_label: 3,
        confidence: 0.88,
        batch_id: 'batch_001',
        created_at: '2026-06-30 12:01:00',
      },
    ]
    getHistory.mockResolvedValue({ records: mockRecords, count: 2 })

    const wrapper = mountComponent()
    await new Promise(process.nextTick)

    expect(wrapper.text()).toContain('共 2 条记录')

    const rows = wrapper.findAll('tbody tr')
    expect(rows).toHaveLength(2)

    // First row
    expect(rows[0].text()).toContain('1')
    expect(rows[0].text()).toContain('test.png')
    expect(rows[0].text()).toContain('5')
    expect(rows[0].text()).toContain('95.0%')

    // Second row with batch
    expect(rows[1].text()).toContain('batch_001')
  })

  it('shows error message when loading fails', async () => {
    getHistory.mockRejectedValue({
      response: { data: { detail: 'Database error' } },
      message: 'Database error',
    })

    const wrapper = mountComponent()
    await new Promise(process.nextTick)

    expect(wrapper.text()).toContain('Database error')
  })

  it('calls getHistory with limit=100 on mount', async () => {
    getHistory.mockResolvedValue({ records: [], count: 0 })

    mountComponent()
    await new Promise(process.nextTick)

    expect(getHistory).toHaveBeenCalledWith(100)
  })

  it('formats confidence correctly', async () => {
    const mockRecords = [
      {
        id: 1,
        filename: 'test.png',
        predicted_label: 5,
        confidence: 0.856,
        batch_id: null,
        created_at: '2026-06-30 12:00:00',
      },
    ]
    getHistory.mockResolvedValue({ records: mockRecords, count: 1 })

    const wrapper = mountComponent()
    await new Promise(process.nextTick)

    expect(wrapper.text()).toContain('85.6%')
  })

  it('formats timestamp correctly', async () => {
    const mockRecords = [
      {
        id: 1,
        filename: 'test.png',
        predicted_label: 5,
        confidence: 0.9,
        batch_id: null,
        created_at: '2026-06-30 14:05:30',
      },
    ]
    getHistory.mockResolvedValue({ records: mockRecords, count: 1 })

    const wrapper = mountComponent()
    await new Promise(process.nextTick)

    expect(wrapper.text()).toContain('2026-06-30 14:05')
  })
})
