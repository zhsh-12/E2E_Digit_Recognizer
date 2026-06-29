<template>
  <div class="card">
    <div class="card-title">{{ $t('batch.title') }}</div>

    <!-- Upload area -->
    <div
      class="upload-area"
      :class="{ dragover: isDragover }"
      @click="triggerUpload"
      @dragover.prevent="isDragover = true"
      @dragleave="isDragover = false"
      @drop.prevent="handleDrop"
    >
      <div class="upload-icon">📁</div>
      <div class="upload-text">{{ $t('batch.upload_text') }}</div>
      <div class="upload-hint">{{ $t('batch.upload_hint') }}</div>
      <input
        ref="fileInput"
        type="file"
        multiple
        accept="image/png,image/jpeg,image/jpg"
        style="display: none"
        @change="handleFileSelect"
      />
    </div>

    <!-- Statistics -->
    <div v-if="results.length > 0" class="stats-bar">
      <div class="stat-item">
        <div class="stat-value">{{ totalCount }}</div>
        <div class="stat-label">{{ $t('stats.total') }}</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color: #52c41a;">{{ correctCount }}</div>
        <div class="stat-label">{{ $t('stats.correct') }}</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color: #ff4d4f;">{{ wrongCount }}</div>
        <div class="stat-label">{{ $t('stats.wrong') }}</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :style="{ color: accuracyColor }">{{ accuracy }}%</div>
        <div class="stat-label">{{ $t('stats.accuracy') }}</div>
      </div>
    </div>

    <!-- Progress bar -->
    <div v-if="processing" class="progress-bar">
      <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
    </div>

    <div v-if="processing" style="text-align: center; color: #666; font-size: 14px; margin-bottom: 12px;">
      {{ $t('batch.predicting', { processed: processedCount, total: totalCount }) }}
    </div>

    <!-- Save button -->
    <div v-if="results.length > 0 && !processing" style="text-align: center; margin-bottom: 16px;">
      <button class="btn btn-primary" @click="saveBatch" :disabled="saving">
        {{ saving ? $t('batch.saving') : $t('batch.save') }}
      </button>
      <button v-if="saveSuccess" class="btn btn-secondary" style="margin-left: 8px;" @click="downloadCsv" :disabled="csvLoading">
        {{ csvLoading ? $t('batch.downloading') : $t('batch.download_csv') }}
      </button>
      <div v-if="saveMsg" style="margin-top: 8px; font-size: 14px; color: #52c41a;">{{ saveMsg }}</div>
      <div v-if="saveError" style="margin-top: 8px; font-size: 14px; color: #ff4d4f;">{{ saveError }}</div>
    </div>

    <!-- Result grid -->
    <div v-if="results.length > 0" class="preview-container">
      <div
        v-for="item in results"
        :key="item.filename"
        class="preview-card"
        :class="{
          correct: item.verdict === true,
          wrong: item.verdict === false,
        }"
      >
        <img :src="item.previewUrl" :alt="item.filename" />
        <div class="filename">{{ item.filename }}</div>
        <div
          class="prediction"
          :class="{
            correct: item.verdict === true,
            wrong: item.verdict === false,
          }"
        >
          {{ item.prediction !== null ? item.prediction : '?' }}
        </div>
        <div v-if="item.trueLabel !== null" class="true-label">
          {{ $t('batch.true_label', { label: item.trueLabel }) }}
        </div>
        <div v-if="item.error" class="error-msg" style="font-size: 12px;">{{ item.error }}</div>
      </div>
    </div>

    <div v-if="errorMsg" class="error-msg">{{ errorMsg }}</div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { predictBatch, saveBatchResults, exportCsv } from '../api/predict.js'

const { t } = useI18n()

const fileInput = ref(null)
const isDragover = ref(false)
const results = ref([])
const processing = ref(false)
const errorMsg = ref('')
const saving = ref(false)
const saveMsg = ref('')
const saveError = ref('')
const saveSuccess = ref(false)
const csvLoading = ref(false)
let savedBatchId = ''
let savedBatchResults = []
let savedBatchAccuracy = 0

function triggerUpload() {
  fileInput.value.click()
}

function handleFileSelect(e) {
  const files = Array.from(e.target.files)
  if (files.length > 0) processFiles(files)
}

function handleDrop(e) {
  isDragover.value = false
  const files = Array.from(e.dataTransfer.files).filter((f) =>
    /\.(png|jpe?g)$/i.test(f.name)
  )
  if (files.length > 0) processFiles(files)
}

function extractTrueLabel(filename) {
  const match = filename.match(/\[(\d+)\]/)
  return match ? parseInt(match[1]) : null
}

async function processFiles(files) {
  errorMsg.value = ''
  results.value = []
  processing.value = true
  saveMsg.value = ''
  saveError.value = ''

  // Create preview URLs first
  const fileItems = files.map((file) => ({
    file,
    filename: file.name,
    previewUrl: URL.createObjectURL(file),
    prediction: null,
    trueLabel: extractTrueLabel(file.name),
    verdict: null,
    error: null,
  }))
  results.value = fileItems

  try {
    const data = await predictBatch(files, (processed, total) => {
      // Progress callback
    })
    data.results.forEach((res, index) => {
      if (index < results.value.length) {
        results.value[index].prediction = res.prediction
        results.value[index].error = res.error
        if (res.prediction !== null && results.value[index].trueLabel !== null) {
          results.value[index].verdict = res.prediction === results.value[index].trueLabel
        }
      }
    })
  } catch (err) {
    errorMsg.value = 'Batch prediction failed: ' + (err.response?.data?.detail || err.message)
  } finally {
    processing.value = false
  }
}

async function saveBatch() {
  saving.value = true
  saveMsg.value = ''
  saveError.value = ''
  saveSuccess.value = false

  const batchId = 'frontend_' + new Date().toISOString().slice(0, 10).replace(/-/g, '') +
    '_' + Date.now().toString(36)

  const batchResults = results.value.map((item) => ({
    filename: item.filename,
    predicted_label: item.prediction,
    true_label: item.trueLabel,
  }))

  const hasLabels = results.value.some(item => item.trueLabel !== null)
  const acc = hasLabels ? parseFloat(accuracy.value) : null

  try {
    const result = await saveBatchResults({
      batch_id: batchId,
      results: batchResults,
      batch_accuracy: acc,
    })
    saveMsg.value = t('batch.saved', { batch: batchId })
    saveSuccess.value = true
    savedBatchId = batchId
    savedBatchResults = batchResults
    savedBatchAccuracy = acc
  } catch (err) {
    saveError.value = 'Save failed: ' + (err.response?.data?.detail || err.message)
  } finally {
    saving.value = false
  }
}

async function downloadCsv() {
  csvLoading.value = true
  try {
    await exportCsv({
      batch_id: savedBatchId,
      results: savedBatchResults,
      batch_accuracy: savedBatchAccuracy,
    })
  } catch (err) {
    saveError.value = 'CSV download failed: ' + (err.response?.data?.detail || err.message)
  } finally {
    csvLoading.value = false
  }
}

const totalCount = computed(() => results.value.length)
const processedCount = computed(() => results.value.filter((r) => r.prediction !== null).length)
const progressPercent = computed(() =>
  totalCount.value > 0 ? (processedCount.value / totalCount.value) * 100 : 0
)
const correctCount = computed(() => results.value.filter((r) => r.verdict === true).length)
const wrongCount = computed(() => results.value.filter((r) => r.verdict === false).length)
const accuracy = computed(() => {
  const total = correctCount.value + wrongCount.value
  return total > 0 ? ((correctCount.value / total) * 100).toFixed(1) : '-'
})
const accuracyColor = computed(() => {
  const val = parseFloat(accuracy.value)
  if (val >= 80) return '#52c41a'
  if (val >= 60) return '#faad14'
  return '#ff4d4f'
})
</script>
