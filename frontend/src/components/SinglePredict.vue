<template>
  <div class="card">
    <div class="card-title">{{ $t('single.title') }}</div>

    <!-- Upload area -->
    <div
      class="upload-area"
      :class="{ dragover: isDragover }"
      @click="triggerUpload"
      @dragover.prevent="isDragover = true"
      @dragleave="isDragover = false"
      @drop.prevent="handleDrop"
    >
      <div class="upload-icon">🖼️</div>
      <div class="upload-text">{{ $t('single.upload_text') }}</div>
      <div class="upload-hint">{{ $t('single.upload_hint') }}</div>
      <input
        ref="fileInput"
        type="file"
        accept="image/png,image/jpeg,image/jpg"
        style="display: none"
        @change="handleFileSelect"
      />
    </div>

    <!-- Image preview -->
    <div v-if="previewUrl" style="text-align: center; margin-top: 20px;">
      <img :src="previewUrl" style="max-width: 200px; max-height: 200px; border-radius: 8px; border: 2px solid #e8e8e8;" />
    </div>

    <!-- Prediction result -->
    <div v-if="prediction !== null" class="single-result">
      <div class="digit">{{ prediction }}</div>
      <div class="label">{{ $t('single.prediction_label') }}</div>
      <div v-if="confidence !== null" class="confidence">
        {{ $t('single.confidence') }}: {{ formatConfidence(confidence) }}
      </div>

      <div class="input-group">
        <button class="btn btn-primary" @click="saveResult" :disabled="saving">
          {{ saving ? $t('single.saving') : $t('single.save') }}
        </button>
      </div>

      <div v-if="saveMsg" style="margin-top: 8px; font-size: 14px; color: #52c41a;">{{ saveMsg }}</div>
      <div v-if="saveError" style="margin-top: 8px; font-size: 14px; color: #ff4d4f;">{{ saveError }}</div>
      <div v-if="saveSuccess" style="margin-top: 8px;">
        <button class="btn btn-secondary" @click="downloadCsv" :disabled="csvLoading">
          {{ csvLoading ? $t('single.downloading') : $t('single.download_csv') }}
        </button>
      </div>
    </div>

    <div v-if="errorMsg" class="error-msg">{{ errorMsg }}</div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { predictSingle, savePrediction, exportCsv } from '../api/predict.js'

const { t } = useI18n()

const fileInput = ref(null)
const isDragover = ref(false)
const previewUrl = ref(null)
const selectedFile = ref(null)
const prediction = ref(null)
const confidence = ref(null)
const errorMsg = ref('')
const saving = ref(false)
const saveMsg = ref('')
const saveError = ref('')
const saveSuccess = ref(false)
const csvLoading = ref(false)

function formatConfidence(val) {
  return val !== null && val !== undefined ? (val * 100).toFixed(1) + '%' : '-'
}

function triggerUpload() {
  fileInput.value.click()
}

function handleFileSelect(e) {
  const file = e.target.files[0]
  if (file) processFile(file)
}

function handleDrop(e) {
  isDragover.value = false
  const file = e.dataTransfer.files[0]
  if (file) processFile(file)
}

function processFile(file) {
  errorMsg.value = ''
  prediction.value = null
  confidence.value = null
  saveMsg.value = ''
  saveError.value = ''
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  doPredict(file)
}

async function doPredict(file) {
  try {
    const result = await predictSingle(file)
    prediction.value = result.prediction
    confidence.value = result.confidence
  } catch (err) {
    errorMsg.value = 'Prediction failed: ' + (err.response?.data?.detail || err.message)
  }
}

async function saveResult() {
  saving.value = true
  saveMsg.value = ''
  saveError.value = ''
  saveSuccess.value = false
  try {
    const result = await savePrediction({
      predicted_label: prediction.value,
      confidence: confidence.value,
      filename: selectedFile.value?.name || null,
    })
    saveMsg.value = t('single.saved', { id: result.id })
    saveSuccess.value = true
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
      filename: selectedFile.value?.name || null,
      predicted_label: prediction.value,
      confidence: confidence.value,
    })
  } catch (err) {
    saveError.value = 'CSV download failed: ' + (err.response?.data?.detail || err.message)
  } finally {
    csvLoading.value = false
  }
}
</script>
