/**
 * Tests for SinglePredict.vue component
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import SinglePredict from '../../src/components/SinglePredict.vue'

// Mock the API module
vi.mock('../../src/api/predict.js', () => ({
  predictSingle: vi.fn(),
  savePrediction: vi.fn(),
  exportCsv: vi.fn(),
}))

import { predictSingle, savePrediction, exportCsv } from '../../src/api/predict.js'

// Create i18n instance for tests
const i18n = createI18n({
  legacy: false,
  locale: 'zh-CN',
  fallbackLocale: 'en',
  messages: {
    'zh-CN': {
      single: {
        title: '单张图片预测',
        upload_text: '点击选择图片，或拖拽图片到此处',
        upload_hint: '支持 JPG / PNG / JPEG 格式',
        prediction_label: '预测结果',
        confidence: '置信度',
        save: '保存',
        saving: '保存中...',
        download_csv: '下载 CSV',
        downloading: '下载中...',
        saved: '保存成功 (ID: {id})',
      },
    },
    en: {
      single: {
        title: 'Single Image Prediction',
        upload_text: 'Click or drag an image here',
        upload_hint: 'Supports JPG / PNG / JPEG',
        prediction_label: 'Prediction',
        confidence: 'Confidence',
        save: 'Save',
        saving: 'Saving...',
        download_csv: 'Download CSV',
        downloading: 'Downloading...',
        saved: 'Saved successfully (ID: {id})',
      },
    },
  },
})

function mountComponent() {
  return mount(SinglePredict, {
    global: {
      plugins: [i18n],
    },
  })
}

describe('SinglePredict.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the title', () => {
    const wrapper = mountComponent()
    expect(wrapper.text()).toContain('单张图片预测')
  })

  it('shows upload area', () => {
    const wrapper = mountComponent()
    expect(wrapper.find('.upload-area').exists()).toBe(true)
    expect(wrapper.text()).toContain('点击选择图片')
  })

  it('triggers file input click on upload area click', async () => {
    const wrapper = mountComponent()
    const fileInput = wrapper.find('input[type="file"]')
    const clickSpy = vi.spyOn(fileInput.element, 'click')

    await wrapper.find('.upload-area').trigger('click')
    expect(clickSpy).toHaveBeenCalled()
  })

  it('shows dragover class on drag over', async () => {
    const wrapper = mountComponent()
    const uploadArea = wrapper.find('.upload-area')

    await uploadArea.trigger('dragover')
    expect(uploadArea.classes()).toContain('dragover')

    await uploadArea.trigger('dragleave')
    expect(uploadArea.classes()).not.toContain('dragover')
  })

  it('processes file on drop and calls predictSingle', async () => {
    predictSingle.mockResolvedValue({ prediction: 7, confidence: 0.92 })

    const wrapper = mountComponent()
    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files: [file] },
    })

    // Wait for async prediction
    await wrapper.vm.$nextTick()
    // Wait for the async doPredict to complete
    await new Promise(process.nextTick)

    expect(predictSingle).toHaveBeenCalledWith(file)
    expect(wrapper.text()).toContain('7')
    expect(wrapper.text()).toContain('92.0%')
  })

  it('processes file on file select and calls predictSingle', async () => {
    predictSingle.mockResolvedValue({ prediction: 3, confidence: 0.85 })

    const wrapper = mountComponent()
    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })

    const fileInput = wrapper.find('input[type="file"]')
    Object.defineProperty(fileInput.element, 'files', {
      value: [file],
    })
    await fileInput.trigger('change')

    await new Promise(process.nextTick)

    expect(predictSingle).toHaveBeenCalledWith(file)
    expect(wrapper.text()).toContain('3')
  })

  it('shows error message when prediction fails', async () => {
    predictSingle.mockRejectedValue({
      response: { data: { detail: 'Model error' } },
      message: 'Model error',
    })

    const wrapper = mountComponent()
    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files: [file] },
    })

    await new Promise(process.nextTick)

    expect(wrapper.text()).toContain('Model error')
  })

  it('calls savePrediction on save button click', async () => {
    predictSingle.mockResolvedValue({ prediction: 5, confidence: 0.95 })
    savePrediction.mockResolvedValue({ id: 42, message: 'Saved successfully' })

    const wrapper = mountComponent()
    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })

    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files: [file] },
    })
    await new Promise(process.nextTick)

    // Click save button
    const saveBtn = wrapper.find('.btn-primary')
    expect(saveBtn.exists()).toBe(true)
    await saveBtn.trigger('click')
    await new Promise(process.nextTick)

    expect(savePrediction).toHaveBeenCalledWith({
      predicted_label: 5,
      confidence: 0.95,
      filename: 'test.png',
    })
    expect(wrapper.text()).toContain('保存成功')
  })

  it('calls exportCsv on download CSV button click', async () => {
    predictSingle.mockResolvedValue({ prediction: 5, confidence: 0.95 })
    savePrediction.mockResolvedValue({ id: 42, message: 'Saved successfully' })

    const wrapper = mountComponent()
    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })

    // Upload and predict
    await wrapper.find('.upload-area').trigger('drop', {
      dataTransfer: { files: [file] },
    })
    await new Promise(process.nextTick)

    // Save first to show download button
    await wrapper.find('.btn-primary').trigger('click')
    await new Promise(process.nextTick)

    // Click download CSV button
    const downloadBtn = wrapper.findAll('.btn-secondary').find(b => b.text().includes('CSV'))
    expect(downloadBtn).toBeTruthy()
    await downloadBtn.trigger('click')
    await new Promise(process.nextTick)

    expect(exportCsv).toHaveBeenCalledWith({
      filename: 'test.png',
      predicted_label: 5,
      confidence: 0.95,
    })
  })
})
