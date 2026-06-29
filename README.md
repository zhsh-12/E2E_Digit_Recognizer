<p align="center">
  <h1 align="center">🖍️ 数字识别系统 / Digit Recognizer</h1>
  <p align="center">
    从模型训练到服务部署的全栈数字识别应用 ·
    <em>End-to-end digit recognition from model training to deployment</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Frontend-Vue%203-4fc08d?logo=vuedotjs" alt="Vue 3"/>
  <img src="https://img.shields.io/badge/Database-SQLite-003b57?logo=sqlite" alt="SQLite"/>
  <img src="https://img.shields.io/badge/Docker-✔-2496ed?logo=docker" alt="Docker"/>
  <img src="https://img.shields.io/badge/Tests-pytest-0a9edc?logo=pytest" alt="pytest"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License"/>
</p>

---

## 📖 项目简介 / Introduction

**数字识别系统** 是一个从零构建的全栈 AI 应用项目，覆盖了**模型训练 → 服务封装 → Web 前端 → 容器化部署**的完整链路。该项目的初衷不仅是实现一个数字识别功能，更是展示模型设计、数据预处理、前后端开发、容器化部署、自动化测试等综合工程过程。

核心是一套**自定义超轻量 ResNet-CNN**（仅约 49K 参数），在 MNIST、EMNIST、SVHN 等混合数据集上从零训练和微调，实现 0-9 手写/印刷体数字的高精度识别。后端基于 FastAPI 提供高性能推理服务，支持单图和批量并发预测。前端使用 Vue 3 + Vite 构建交互式界面。通过 Docker Compose 可一键部署。

---

## ✨ 功能特性 / Features

| 特性 | 说明 |
|------|------|
| 🎯 **数字识别** | 支持 0-9 手写数字和印刷体数字识别 |
| 🖼️ **单图预测** | 拖拽上传或点击选择图片，实时预览 + 识别结果显示 |
| 📦 **批量预测** | 一次最多 50 张图片，asyncio + ThreadPool 并发推理，大幅缩短等待时间 |
| ✅ **结果校验** | 支持输入真实标签进行比对，显示 ✓/✗ 判断 |
| 📊 **CSV 导出** | 预测结果一键导出为结构化 CSV 文件 |
| 📋 **历史记录** | 保存预测历史，支持分页查询（默认显示最近 50 条） |
| 🌏 **中英双语** | Vue I18n 实现的中英文界面实时切换 |
| 🐳 **容器化部署** | Docker Compose 一键启动，Nginx 反向代理前端 API 请求 |
| 🧪 **测试覆盖** | pytest + coverage，覆盖 API、数据库、预处理、CSV 导出等核心模块 |

---

## 🛠️ 技术栈 / Tech Stack

| 层级 / Layer | 技术 / Technology |
|-------------|-------------------|
| **模型 / Model** | PyTorch, 自定义 ResNet-CNN, TensorBoard, scikit-learn (混淆矩阵) |
| **后端 / Backend** | FastAPI, Uvicorn, asyncio, ThreadPoolExecutor |
| **前端 / Frontend** | Vue 3 (Composition API), Vite, vue-i18n, Axios |
| **数据库 / Database** | SQLite (PRAGMA user_version 版本迁移) |
| **容器化 / Container** | Docker (多阶段构建), Docker Compose, Nginx |
| **开发工具 / Tools** | uv (包管理), pytest, coverage, httpx, matplotlib / seaborn (可视化) |

---

## 🏗️ 系统架构 / Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────────┐
│   Vue 3      │────▶│   Nginx     │────▶│   FastAPI        │
│   Frontend   │     │   (Proxy)   │     │   /predict       │
│  port:5173   │     │   port:80   │     │   /batch_predict │
└──────────────┘     └─────────────┘     │   /api/*         │
         │                                │   /health        │
         │                                └────────┬─────────┘
         │                                         │
         │                                ┌────────▼─────────┐
         │                                │  PyTorch Model   │
         │                                │  DigitRecognizer │
         │                                │  (ResNet-CNN)    │
         │                                └────────┬─────────┘
         │                                         │
         │                                ┌────────▼─────────┐
         └────────────────────────────────▶   SQLite DB     │
                                            (history logs)  └──────────────────┘
```

**推理流程 / Inference Flow:**
1. 用户上传图片 → Vue 前端 / CLI 客户端
2. 请求发送到 FastAPI 后端
3. 后端预处理（灰度→RGB、resize 28×28、归一化到 [-1, 1]）
4. PyTorch 模型推理 → argmax 获取预测值
5. 返回结果给前端展示，同时可选存入 SQLite

---

## 🧠 模型架构 / Model Architecture

自定义 **超轻量 ResNet-CNN**，专为数字识别场景设计：

```
Input: 3×28×28
    │
    ▼
┌─────────────┐    Conv2d(3→32, k=3) + BN + ReLU
│    Stem     │
└──────┬──────┘
       │
┌─────────────┐    ResidualBlock(32→32) + MaxPool2d(2) → 14×14
│  Layer 1    │
└──────┬──────┘
       │
┌─────────────┐    ResidualBlock(32→64) + MaxPool2d(2) → 7×7
│  Layer 2    │
└──────┬──────┘
       │
┌─────────────┐    ResidualBlock(64→128) → 7×7
│  Layer 3    │
└──────┬──────┘
       │
┌─────────────┐    AdaptiveAvgPool2d(1) → 128-d
│     GAP     │
└──────┬──────┘
       │
┌─────────────┐    FC(128→64) → ReLU → Dropout(0.3) → FC(64→10)
│ Classifier  │
└──────┬──────┘
       │
    Output: 10 classes (0-9)
```

**设计亮点 / Highlights:**
- **超轻量**：仅约 **49K 参数**，单张 CPU 推理 < 15ms
- **残差连接**：3 层 Residual Block 缓解梯度消失，加速收敛
- **全局平均池化**（GAP）：替代全连接层展平，大幅减少参数量
- **多通道兼容**：预处理自动适配 L(灰度)、RGB、RGBA 三种图像模式
- **学习率调度 + Early Stopping**：训练中自动保存最佳模型

---

## 📁 项目结构 / Project Structure

```
Digit_recognizer_app/
├── api/                        # FastAPI 后端服务
│   ├── main.py                 #   API 入口，路由定义
│   ├── services.py             #   批量推理并发服务
│   ├── schemas.py              #   Pydantic 数据模型
│   └── csv_export.py           #   CSV 导出功能
├── app/                        # 模型核心
│   ├── model.py                #   DigitRecognizer 模型定义 + 单例加载
│   └── preprocess.py           #   图像预处理 (L/RGB/RGBA 适配)
├── cli/                        # 命令行客户端
│   ├── client.py               #   HTTP 请求封装
│   ├── batch_test.py           #   批量测试 (自动检测标签)
│   ├── single_test.py          #   单图测试
│   └── utils.py                #   图片读取、标签提取
├── common/
# 公共模块
│   ├── config.py
# 全局配置
│   └── database.py
# SQLite 数据库 (含版本迁移)
├── frontend/
# Vue 3 前端
│   ├── src/
# 源码
│   │   ├── components/
#     SinglePredict/BatchPredict/History
│   │   ├── api/predict.js
#     Axios API 封装
│   │   ├── locales/
#     中英文 JSON
│   │   ├── App.vue / main.js
#     入口
│   │   └── style.css
│   ├── index.html
│   ├── vite.config.js
#   Vite 配置 (开发代理)
│   └── package.json
├── training/
# 模型训练
│   ├── initial_training/
#   初始训练 (baseline)
│   │   └── main.py / model.py / trainer.py / ...
│   └── Full_FT/
#   微调训练 (FineTuning)
│       └── main.py / model.py / trainer.py / ...
├── docker/
# Docker 部署配置
│   ├── Dockerfile
#   后端 Dockerfile
│   ├── Dockerfile.frontend
#   前端多阶段构建
│   ├── docker-compose.yml
#   服务编排
│   └── nginx.conf
#   Nginx 反向代理
├── tests/
# 测试 (8 个测试文件)
│   ├── conftest.py
#   pytest fixtures + mock 模型
│   ├── test_api_endpoints.py
#   API 端点测试
│   ├── test_database.py
#   数据库测试
│   └── ...
├── models/
# 训练好的模型权重
│   └── digit_recognizer.pth
├── pyproject.toml
# 项目配置 + 依赖
├── uv.lock
# 锁定依赖版本
├── Makefile
# 常用命令
├── LICENSE
# MIT 许可证
└── README.md
# 项目文档
```

---

## 🚀 快速开始 / Quick Start

### 方式一：本地运行 / Run Locally

**前置要求 / Prerequisites:**
- Python >= 3.11，uv（推荐）
- Node.js >= 20（前端开发用）
- 模型权重文件 models/digit_recognizer.pth

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd Digit_recognizer_app

# 2. 安装后端依赖
uv sync

# 3. 准备模型权重
#    方式 A：将训练好的 digit_recognizer.pth 放入 models/
#    方式 B：自行训练（参见训练模型章节）

# 4. 启动后端（终端 1）
uv run uvicorn api.main:app --reload --port 8000

# 5. 安装前端依赖并启动（终端 2）
cd frontend
npm install
npm run dev

# 6. 打开浏览器访问 http://localhost:5173
```

**CLI 测试（不需要前端）:**
```bash
# 确保后端正在运行
# 单图预测
uv run python -m cli.single_test

# 批量预测
uv run python -m cli.batch_test
```

### 方式二：Docker 部署 / Deploy with Docker（推荐）

```bash
# 一键启动前后端服务
make docker-up

# 或者手动执行
docker compose -f docker/docker-compose.yml up -d --build

# 访问 http://localhost:8080
# API 地址 http://localhost:8000

# 停止服务
make docker-down
```

---

## 📡 API 文档 / API Reference

| 端点 / Endpoint | 方法 | 说明 / Description |
|----------------|:----:|-------------------|
| /health | GET | 健康检查 / Health check |
| /predict | POST | 单图预测 / Single image prediction |
| /batch_predict | POST | 批量预测（≤50张）/ Batch prediction (max 50) |
| /api/save_prediction | POST | 保存单次预测 / Save single prediction |
| /api/save_batch_results | POST | 保存批量预测 / Save batch results |
| /api/prediction_history | GET | 查询预测历史 / Query history |
| /api/export_csv | POST | 导出 CSV / Export CSV |

**单图预测请求示例:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.png"
# Response: {"prediction": 2}
```

**批量预测请求示例:**
```bash
curl -X POST http://localhost:8000/batch_predict \
  -F "files=@img_1.png" -F "files=@img_2.png"
# Response: {"results": [{"filename": "img_1.png", "prediction": 2, ...}, ...]}
```

---

## 🏋️ 训练模型 / Model Training

### 初始训练 / Initial Training

在 training/initial_training/ 目录下，基于 MNIST + EMNIST(digits) + SVHN 混合数据集训练。

```bash
# baseline 模式：标准训练
uv run python -m training.initial_training.main --mode baseline \
  --epochs 30 --lr 1e-3 --batch_size 128
```

**特性:**
- 支持按比例混合三个数据集（--sample_ratio 1:1:1）
- 每个域单独记录准确率（MNIST / EMNIST / SVHN）
- 混淆矩阵可视化
- TensorBoard 日志 + Early Stopping + 学习率调度

### 微调训练 / Fine-Tuning (Full_FT)

在 training/Full_FT/ 目录下，基于初始训练的最佳模型进行微调。

```bash
# FineTuning 模式：加载旧模型，混合新旧数据训练
uv run python -m training.Full_FT.main --mode FineTuning \
  --epochs 10 --lr 1e-4 --sample_ratio 1:1
```

**特性:**
- 新旧数据按比例混合训练（--sample_ratio n:o）
- 支持断点续训（从 last.pth 恢复）
- 在新旧数据上分别评估

---

## 🧪 测试 / Testing

```bash
# 运行所有测试
make test
# 或者：uv run pytest

# 带覆盖率报告
make coverage
# 或者：uv run pytest --cov --cov-report=html

# 查看覆盖率 HTML 报告
open htmlcov/index.html
```

测试覆盖模块：API 端点、预处理、批量推理服务、数据库操作、CSV 导出、数据模型校验、CLI 工具函数。

---

## 📄 许可证 / License

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

<p align="center">
  <sub>Built with using PyTorch, FastAPI, Vue 3, and Docker</sub>
</p>

