<template>
  <div class="card">
    <div class="card-title">{{ $t('history.title') }}</div>

    <div v-if="loading" style="text-align: center; padding: 40px; color: #999;">{{ $t('history.loading') }}</div>

    <div v-else-if="records.length === 0" style="text-align: center; padding: 40px; color: #999;">
      {{ $t('history.empty') }}
    </div>

    <div v-else>
      <div style="margin-bottom: 12px; color: #666; font-size: 14px;">
        {{ $t('history.total', { count: records.length }) }}
      </div>

      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <thead>
          <tr style="background: #fafafa; border-bottom: 2px solid #e8e8e8;">
            <th style="padding: 10px 12px; text-align: left;">{{ $t('history.id') }}</th>
            <th style="padding: 10px 12px; text-align: left;">{{ $t('history.filename') }}</th>
            <th style="padding: 10px 12px; text-align: center;">{{ $t('history.prediction') }}</th>
            <th style="padding: 10px 12px; text-align: center;">{{ $t('history.true_label') }}</th>
            <th style="padding: 10px 12px; text-align: center;">{{ $t('history.result') }}</th>
            <th style="padding: 10px 12px; text-align: center;">{{ $t('history.batch') }}</th>
            <th style="padding: 10px 12px; text-align: center;">{{ $t('history.accuracy') }}</th>
            <th style="padding: 10px 12px; text-align: right;">{{ $t('history.time') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(record, index) in records"
            :key="record.id"
            :style="{ background: index % 2 === 0 ? '#fff' : '#fafafa', borderBottom: '1px solid #f0f0f0' }"
          >
            <td style="padding: 10px 12px; color: #999;">{{ record.id }}</td>
            <td style="padding: 10px 12px;">{{ record.filename || '-' }}</td>
            <td style="padding: 10px 12px; text-align: center; font-weight: 600;">{{ record.predicted_label }}</td>
            <td style="padding: 10px 12px; text-align: center;">
              {{ record.true_label !== null ? record.true_label : '-' }}
            </td>
            <td style="padding: 10px 12px; text-align: center;">
              <span v-if="record.is_correct === true" style="color: #52c41a;">✓</span>
              <span v-else-if="record.is_correct === false" style="color: #ff4d4f;">✗</span>
              <span v-else style="color: #ccc;">-</span>
            </td>
            <td style="padding: 10px 12px; text-align: center; font-size: 12px; color: #999;">
              {{ record.batch_id || '-' }}
            </td>
            <td style="padding: 10px 12px; text-align: center;">
              {{ record.batch_accuracy !== null ? record.batch_accuracy + '%' : '-' }}
            </td>
            <td style="padding: 10px 12px; text-align: right; font-size: 12px; color: #999; white-space: nowrap;">
              {{ formatTime(record.created_at) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="errorMsg" class="error-msg">{{ errorMsg }}</div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getHistory } from '../api/predict.js'

const records = ref([])
const loading = ref(true)
const errorMsg = ref('')

onMounted(async () => {
  try {
    const data = await getHistory(100)
    records.value = data.records
  } catch (err) {
    errorMsg.value = 'Failed to load history: ' + (err.response?.data?.detail || err.message)
  } finally {
    loading.value = false
  }
})

function formatTime(timestamp) {
  if (!timestamp) return '-'
  const d = new Date(timestamp)
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
</script>
