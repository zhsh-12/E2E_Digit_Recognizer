import axios from 'axios'

const api = axios.create({
  baseURL: '',
  timeout: 120000, // Increased timeout to 120s for large batch uploads
})

// Maximum images per batch prediction
const MAX_BATCH_SIZE = 120

/**
 * Predict a single image
 * @param {File} file - Image file
 * @returns {Promise<{prediction: number, confidence: number}>}
 */
export async function predictSingle(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post('/predict', formData)
  return response.data
}

/**
 * Batch predict images (supports auto-batching)
 *
 * - File count <= MAX_BATCH_SIZE: single request
 * - File count > MAX_BATCH_SIZE: auto-split into batches of MAX_BATCH_SIZE
 *
 * @param {File[]} files - Array of image files
 * @param {function} [onProgress] - Progress callback (batchResults, processed, total) => void
 * @returns {Promise<{results: Array<{filename: string, prediction: number, confidence: number, error: string|null}>}>}
 */
export async function predictBatch(files, onProgress) {
  if (!files || files.length === 0) {
    return { results: [] }
  }

  // Small number of images: single request
  if (files.length <= MAX_BATCH_SIZE) {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })
    const response = await api.post('/batch_predict', formData)
    if (onProgress) onProgress(files.length, files.length)
    return response.data
  }

  // Large number of images: auto-split into batches
  // Prevents memory overflow or timeout from sending all files in one request
  const allResults = []
  const total = files.length

  for (let i = 0; i < total; i += MAX_BATCH_SIZE) {
    const chunk = files.slice(i, i + MAX_BATCH_SIZE)
    const formData = new FormData()
    chunk.forEach((file) => {
      formData.append('files', file)
    })

    const response = await api.post('/batch_predict', formData)
    allResults.push(...response.data.results)

    if (onProgress) {
      onProgress(response.data.results, Math.min(i + MAX_BATCH_SIZE, total), total)
    }
  }

  return { results: allResults }
}

/**
 * Save single prediction result
 * @param {{predicted_label: number, confidence: number, filename?: string, batch_id?: string}} data
 * @returns {Promise<{id: number, message: string}>}
 */
export async function savePrediction(data) {
  const response = await api.post('/api/save_prediction', data)
  return response.data
}

/**
 * Save batch prediction results
 * @param {{batch_id: string, results: Array<{filename?: string, predicted_label: number, confidence: number}>}} data
 * @returns {Promise<{message: string}>}
 */
export async function saveBatchResults(data) {
  const response = await api.post('/api/save_batch_results', data)
  return response.data
}

/**
 * Query prediction history
 * @param {number} limit
 * @param {number} offset
 * @returns {Promise<{records: Array, count: number}>}
 */
export async function getHistory(limit = 50, offset = 0) {
  const response = await api.get('/api/prediction_history', { params: { limit, offset } })
  return response.data
}

/**
 * Export prediction results as CSV file (triggers browser download)
 * @param {Object} data
 *  Single: { filename, predicted_label, confidence }
 *  Batch: { batch_id, results: [{filename, predicted_label, confidence}] }
 * @returns {Promise<void>}
 */
export async function exportCsv(data) {
  const response = await api.post('/api/export_csv', data, {
    responseType: 'blob',
  })
  // Extract filename from Content-Disposition header
  const disposition = response.headers['content-disposition']
  let filename = 'predict_result.csv'
  if (disposition) {
    const match = disposition.match(/filename="?(.+?)"?$/)
    if (match) filename = match[1]
  }
  // Create download link and trigger download
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'text/csv;charset=utf-8;' }))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', filename)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}
