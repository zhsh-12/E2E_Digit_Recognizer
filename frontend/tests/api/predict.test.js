/**
 * Tests for frontend/src/api/predict.js
 *
 * Uses vi.mock to mock axios so no real HTTP requests are made.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Use vi.hoisted to define mock functions that can be accessed in vi.mock factory
const { mockPost, mockGet, mockCreate } = vi.hoisted(() => {
  const mockPost = vi.fn()
  const mockGet = vi.fn()
  const mockCreate = vi.fn(() => ({
    post: mockPost,
    get: mockGet,
  }))
  return { mockPost, mockGet, mockCreate }
})

// Mock axios before importing the module under test
vi.mock('axios', () => ({
  default: {
    create: mockCreate,
  },
  create: mockCreate,
}))

// Import after mocking - use @ alias defined in vite.config.js
import {
  predictSingle,
  predictBatch,
  savePrediction,
  saveBatchResults,
  getHistory,
  exportCsv,
} from '@/api/predict.js'

describe('predictSingle', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should POST /predict with FormData and return prediction data', async () => {
    mockPost.mockResolvedValue({ data: { prediction: 5, confidence: 0.95, inference_time_ms: 12.3 } })

    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })
    const result = await predictSingle(file)

    expect(mockPost).toHaveBeenCalledWith('/predict', expect.any(FormData))
    expect(result).toEqual({ prediction: 5, confidence: 0.95, inference_time_ms: 12.3 })
  })

  it('should propagate errors', async () => {
    mockPost.mockRejectedValue(new Error('Network error'))

    const file = new File(['fake-image'], 'test.png', { type: 'image/png' })
    await expect(predictSingle(file)).rejects.toThrow('Network error')
  })
})

describe('predictBatch', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should POST /batch_predict with multiple files', async () => {
    mockPost.mockResolvedValue({
      data: {
        results: [
          { filename: 'a.png', prediction: 1, confidence: 0.9 },
          { filename: 'b.png', prediction: 2, confidence: 0.8 },
        ],
      },
    })

    const files = [
      new File(['a'], 'a.png', { type: 'image/png' }),
      new File(['b'], 'b.png', { type: 'image/png' }),
    ]
    const result = await predictBatch(files)

    expect(mockPost).toHaveBeenCalledWith('/batch_predict', expect.any(FormData))
    expect(result.results).toHaveLength(2)
    expect(result.results[0].filename).toBe('a.png')
  })

  it('should return empty results for empty files array', async () => {
    const result = await predictBatch([])
    expect(result).toEqual({ results: [] })
  })

  it('should call onProgress callback', async () => {
    mockPost.mockResolvedValue({
      data: {
        results: [
          { filename: 'a.png', prediction: 1, confidence: 0.9 },
        ],
      },
    })

    const onProgress = vi.fn()
    const files = [new File(['a'], 'a.png', { type: 'image/png' })]
    await predictBatch(files, onProgress)

    expect(onProgress).toHaveBeenCalledTimes(1)
    expect(onProgress).toHaveBeenCalledWith(1, 1)
  })

  it('should auto-split into batches when files exceed MAX_BATCH_SIZE', async () => {
    // MAX_BATCH_SIZE = 120, so 150 files should be split into 2 requests
    mockPost
      .mockResolvedValueOnce({
        data: {
          results: Array.from({ length: 120 }, (_, i) => ({
            filename: `img_${i}.png`,
            prediction: i % 10,
            confidence: 0.9,
          })),
        },
      })
      .mockResolvedValueOnce({
        data: {
          results: Array.from({ length: 30 }, (_, i) => ({
            filename: `img_${i + 120}.png`,
            prediction: i % 10,
            confidence: 0.9,
          })),
        },
      })

    const files = Array.from({ length: 150 }, (_, i) =>
      new File([`content-${i}`], `img_${i}.png`, { type: 'image/png' })
    )
    const result = await predictBatch(files)

    expect(mockPost).toHaveBeenCalledTimes(2)
    expect(result.results).toHaveLength(150)
  })
})

describe('savePrediction', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should POST /api/save_prediction with prediction data', async () => {
    mockPost.mockResolvedValue({ data: { id: 42, message: 'Saved successfully' } })

    const result = await savePrediction({
      predicted_label: 5,
      confidence: 0.95,
      filename: 'test.png',
    })

    expect(mockPost).toHaveBeenCalledWith('/api/save_prediction', {
      predicted_label: 5,
      confidence: 0.95,
      filename: 'test.png',
    })
    expect(result).toEqual({ id: 42, message: 'Saved successfully' })
  })
})

describe('saveBatchResults', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should POST /api/save_batch_results with batch data', async () => {
    mockPost.mockResolvedValue({ data: { message: 'Batch batch_001 saved' } })

    const result = await saveBatchResults({
      batch_id: 'batch_001',
      results: [{ filename: 'a.png', predicted_label: 1, confidence: 0.9 }],
    })

    expect(mockPost).toHaveBeenCalledWith('/api/save_batch_results', {
      batch_id: 'batch_001',
      results: [{ filename: 'a.png', predicted_label: 1, confidence: 0.9 }],
    })
    expect(result).toEqual({ message: 'Batch batch_001 saved' })
  })
})

describe('getHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should GET /api/prediction_history with pagination params', async () => {
    mockGet.mockResolvedValue({
      data: {
        records: [{ id: 1, predicted_label: 5, confidence: 0.95 }],
        count: 1,
      },
    })

    const result = await getHistory(10, 0)

    expect(mockGet).toHaveBeenCalledWith('/api/prediction_history', {
      params: { limit: 10, offset: 0 },
    })
    expect(result.records).toHaveLength(1)
  })

  it('should use default limit=50 and offset=0', async () => {
    mockGet.mockResolvedValue({ data: { records: [], count: 0 } })

    await getHistory()

    expect(mockGet).toHaveBeenCalledWith('/api/prediction_history', {
      params: { limit: 50, offset: 0 },
    })
  })
})

describe('exportCsv', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Mock URL.createObjectURL and document.createElement
    global.URL.createObjectURL = vi.fn(() => 'blob:test')
    global.URL.revokeObjectURL = vi.fn()
  })

  it('should POST /api/export_csv and trigger download for single prediction', async () => {
    const blobContent = 'filename,prediction,confidence\ntest.png,5,0.95\n'
    mockPost.mockResolvedValue({
      data: new Blob([blobContent], { type: 'text/csv;charset=utf-8;' }),
      headers: {
        'content-disposition': 'attachment; filename="predict_test_img.csv"',
      },
    })

    // Mock DOM manipulation
    const mockLink = {
      href: '',
      setAttribute: vi.fn(),
      click: vi.fn(),
    }
    document.createElement = vi.fn(() => mockLink)
    document.body.appendChild = vi.fn()
    document.body.removeChild = vi.fn()

    await exportCsv({
      filename: 'test.png',
      predicted_label: 5,
      confidence: 0.95,
    })

    expect(mockPost).toHaveBeenCalledWith('/api/export_csv', {
      filename: 'test.png',
      predicted_label: 5,
      confidence: 0.95,
    }, { responseType: 'blob' })
    expect(mockLink.click).toHaveBeenCalled()
    expect(mockLink.setAttribute).toHaveBeenCalledWith('download', 'predict_test_img.csv')
  })

  it('should use default filename when Content-Disposition is missing', async () => {
    mockPost.mockResolvedValue({
      data: new Blob(['test'], { type: 'text/csv;charset=utf-8;' }),
      headers: {},
    })

    const mockLink = {
      href: '',
      setAttribute: vi.fn(),
      click: vi.fn(),
    }
    document.createElement = vi.fn(() => mockLink)
    document.body.appendChild = vi.fn()
    document.body.removeChild = vi.fn()

    await exportCsv({ filename: 'test.png', predicted_label: 5, confidence: 0.95 })

    expect(mockLink.setAttribute).toHaveBeenCalledWith('download', 'predict_result.csv')
  })
})
