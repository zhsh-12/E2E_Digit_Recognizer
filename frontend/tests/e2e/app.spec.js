/**
 * E2E tests for the Digit Recognizer application.
 *
 * Prerequisites:
 *   - Backend running at http://localhost:8000
 *   - Frontend dev server running at http://localhost:5173
 *
 * Run with: npx playwright test
 */
import { test, expect } from '@playwright/test'

test.describe('Digit Recognizer App', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should display the app title', async ({ page }) => {
    await expect(page.locator('h1')).toHaveText(/数字识别系统/)
  })

  test('should show SinglePredict tab by default', async ({ page }) => {
    await expect(page.locator('text=点击选择图片，或拖拽图片到此处')).toBeVisible()
  })

  test('should switch to BatchPredict tab', async ({ page }) => {
    await page.click('button:has-text("批量预测")')
    await expect(page.locator('text=点击选择多张图片，或拖拽图片到此处')).toBeVisible()
  })

  test('should switch to History tab', async ({ page }) => {
    await page.click('button:has-text("历史记录")')
    await expect(page.locator('.card-title:has-text("预测历史记录")')).toBeVisible()
  })

  test('should toggle language', async ({ page }) => {
    const langBtn = page.locator('button:has-text("EN"), button:has-text("中文")')
    await langBtn.click()
    await expect(page.locator('h1')).toHaveText(/Digit Recognition System/)
  })

  test('should upload a single image and show prediction result', async ({ page }) => {
    // Upload a test image - use absolute path
    const fileInput = page.locator('input[type="file"]').first()
    await fileInput.setInputFiles('/Users/zhshaop/Model_train/Digit_recognizer_quant/test_imgs/batch_1/img_1[2].jpg')

    // Wait for prediction result to appear
    await expect(page.locator('text=预测结果')).toBeVisible({ timeout: 15000 })
    await expect(page.locator('text=置信度')).toBeVisible()
  })

  test('should upload batch images and show results', async ({ page }) => {
    await page.click('button:has-text("批量预测")')

    const fileInput = page.locator('input[type="file"]').first()
    await fileInput.setInputFiles([
      '/Users/zhshaop/Model_train/Digit_recognizer_quant/test_imgs/batch_1/img_1[2].jpg',
      '/Users/zhshaop/Model_train/Digit_recognizer_quant/test_imgs/batch_1/img_2[1].jpg',
      '/Users/zhshaop/Model_train/Digit_recognizer_quant/test_imgs/batch_1/img_3[4].jpg',
    ])

    // Wait for results - check for preview cards
    await expect(page.locator('.preview-card')).toHaveCount(3, { timeout: 30000 })
  })

  test('should show error for invalid file type', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]').first()
    await fileInput.setInputFiles({
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('not an image'),
    })

    // The error message is shown in the error-msg div
    await expect(page.locator('.error-msg')).toBeVisible({ timeout: 5000 })
  })
})
