/**
 * Tests for BatchPredict.vue component
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import BatchPredict from '../../src/components/BatchPredict.vue'

// Mock the API module
vi.mock('../../src/api/predict.js', () => ({
  predictBatch: vi.fn(),
  saveBatchResults: vi.fn(),
  exportCsv: vi.fn(),
}))

import { predictBatch, saveBatchResults, exportCsv } from '../../src/api/predict.js'

const i18n = createI18n({
  legacy: false,
  locale: 'zh-CN',
  fallbackLocale: 'en',
  messages: {
    'zh-CN': {
      batch: {
        title: '批量图片预测',
        upload_text: '点击选择多张图片，或拖拽图片到此处',
        upload_hint: '支持 JPG / PNG / JPEG 格式，可多选',
        predicting: '正在预测... {processed} / {total}',
        save: '保存',
        saving: '保存中...',
        download_csv: '下载 CSV',
        downloading: '下载中...',
        saved: '保存成功 (批次: {batch})',
      },
      stats: { total: '总数' },
    },
    en: {
      batch: {
        title: 'Batch Image Prediction',
        upload_text: 'Click to select multiple images, or drag them here',
        upload_hint: 'Supports JPG / PNG / JPEG, multiple selection',
        predicting: 'Predicting... {processed} / {total}',
        save: 'Save',
        saving: 'Saving...',
        download_csv: 'Download CSV',
        downloading: 'Downloading...',
        saved: 'Saved successfully (Batch: {batch})',
      },
      stats: { total: 'Total' },
    },
  },
})

function mountComponent() {
  return mount(BatchPredict, {
    global: {
      plugins: [i18n],
    },
  })
}

describe('BatchPredict.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the title', () => {
    const wrapper = mountComponent()
    expect(wrapper.text()).toContain('批量图片预测')
  })

  it('shows upload area', () => {
    const wrapper = mountComponent()
    expect(wrapper.find('.upload-area').exists()).toBe(true)
    expect(wrapper.text()).toContain('点击选择多张图片')
  })

  it('shows dragover class on drag over', async () => {
    const wrapper = mountComponent()
    const uploadArea = wrapper.find('.upload-area')

    await uploadArea.trigger('dragover')
    expect(uploadArea.classes()).toContain('dragover')

    await uploadArea.trigger('dragleave')
    expect(uploadArea.classes()).not.toContain('dragover')
  })

  it('processes files on drop and calls predictBatch', async () => {
    predictBatch.mockResolvedValue({
      results: [
        { filename: 'a.png', prediction: 1, confidence: 0.9, error: null },
        { filename: 'b.png', prediction: 2, confidence: 0.8, error: null },
      ],
    })

    const wrapper = mountComponent()
    const files = [
      new File(['a'], 'a.png', { type: 'image/png' }),
      new File(['b'], 'b.png', { type: 'image/png' }),
    ]

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files },
    })

    await new Promise(process.nextTick)

    expect(predictBatch).toHaveBeenCalledWith(files, expect.any(Function))
    // Should show stats bar with total count
    expect(wrapper.text()).toContain('2')
  })

  it('processes files on file select and calls predictBatch', async () => {
    predictBatch.mockResolvedValue({
      results: [
        { filename: 'a.png', prediction: 1, confidence: 0.9, error: null },
      ],
    })

    const wrapper = mountComponent()
    const file = new File(['a'], 'a.png', { type: 'image/png' })

    const fileInput = wrapper.find('input[type="file"]')
    Object.defineProperty(fileInput.element, 'files', {
      value: [file],
    })
    await fileInput.trigger('change')

    await new Promise(process.nextTick)

    expect(predictBatch).toHaveBeenCalled()
  })

  it('shows progress bar during processing', async () => {
    // Don't resolve immediately to see processing state
    predictBatch.mockReturnValue(new Promise(() => {}))

    const wrapper = mountComponent()
    const files = [
      new File(['a'], 'a.png', { type: 'image/png' }),
    ]

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files },
    })

    await wrapper.vm.$nextTick()

    expect(wrapper.find('.progress-bar').exists()).toBe(true)
    expect(wrapper.text()).toContain('正在预测')
  })

  it('shows error message when batch prediction fails', async () => {
    predictBatch.mockRejectedValue({
      response: { data: { detail: 'Server error' } },
      message: 'Server error',
    })

    const wrapper = mountComponent()
    const files = [new File(['a'], 'a.png', { type: 'image/png' })]

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files },
    })

    await new Promise(process.nextTick)

    expect(wrapper.text()).toContain('Server error')
  })

  it('calls saveBatchResults on save button click', async () => {
    predictBatch.mockResolvedValue({
      results: [
        { filename: 'a.png', prediction: 1, confidence: 0.9, error: null },
      ],
    })
    saveBatchResults.mockResolvedValue({ message: 'Batch saved' })

    const wrapper = mountComponent()
    const files = [new File(['a'], 'a.png', { type: 'image/png' })]

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files },
    })
    await new Promise(process.nextTick)

    // Click save button
    const saveBtn = wrapper.find('.btn-primary')
    expect(saveBtn.exists()).toBe(true)
    await saveBtn.trigger('click')
    await new Promise(process.nextTick)

    expect(saveBatchResults).toHaveBeenCalled()
    const saveCall = saveBatchResults.mock.calls[0][0]
    expect(saveCall).toHaveProperty('batch_id')
    expect(saveCall.batch_id).toMatch(/^batch_/)
    expect(saveCall.results).toHaveLength(1)
    expect(saveCall.results[0].filename).toBe('a.png')
  })

  it('calls exportCsv on download CSV button click', async () => {
    predictBatch.mockResolvedValue({
      results: [
        { filename: 'a.png', prediction: 1, confidence: 0.9, error: null },
      ],
    })
    saveBatchResults.mockResolvedValue({ message: 'Batch saved' })

    const wrapper = mountComponent()
    const files = [new File(['a'], 'a.png', { type: 'image/png' })]

    // Upload and predict
    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files },
    })
    await new Promise(process.nextTick)

    // Save first
    await wrapper.find('.btn-primary').trigger('click')
    await new Promise(process.nextTick)

    // Click download CSV button
    const downloadBtn = wrapper.findAll('.btn-secondary').find(b => b.text().includes('CSV'))
    expect(downloadBtn).toBeTruthy()
    await downloadBtn.trigger('click')
    await new Promise(process.nextTick)

    expect(exportCsv).toHaveBeenCalled()
    const csvCall = exportCsv.mock.calls[0][0]
    expect(csvCall).toHaveProperty('batch_id')
    expect(csvCall).toHaveProperty('results')
  })

  it('renders result cards with predictions', async () => {
    predictBatch.mockResolvedValue({
      results: [
        { filename: 'a.png', prediction: 7, confidence: 0.95, error: null },
        { filename: 'b.png', prediction: 3, confidence: 0.88, error: null },
      ],
    })

    const wrapper = mountComponent()
    const files = [
      new File(['a'], 'a.png', { type: 'image/png' }),
      new File(['b'], 'b.png', { type: 'image/png' }),
    ]

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files },
    })
    await new Promise(process.nextTick)

    const cards = wrapper.findAll('.preview-card')
    expect(cards).toHaveLength(2)
    expect(cards[0].text()).toContain('7')
    expect(cards[0].text()).toContain('95.0%')
    expect(cards[1].text()).toContain('3')
    expect(cards[1].text()).toContain('88.0%')
  })
})
