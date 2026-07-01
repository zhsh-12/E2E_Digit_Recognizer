# ============================================
# Makefile - Digit Recognizer Quant
# Environment: macOS + Colima + docker-compose
# ============================================

# ─── Docker 服务管理 ─────────────────────────

.PHONY: build up down restart ps logs

## 构建镜像（所有服务）
build:
	docker-compose build

## 构建镜像（不使用缓存）
build-no-cache:
	docker-compose build --no-cache

## 构建并启动服务
up:
	docker-compose up -d --build

## 启动服务（使用已有镜像）
start:
	docker-compose up -d

## 停止服务（不删除容器）
stop:
	docker-compose stop

## 停止并删除容器
down:
	docker-compose down

## 重启服务
restart:
	docker-compose restart

## 查看运行状态
ps:
	docker-compose ps

## 查看所有日志
logs:
	docker-compose logs -f

## 查看后端日志
logs-backend:
	docker-compose logs -f backend

## 查看前端日志
logs-frontend:
	docker-compose logs -f frontend


# ─── Docker 镜像管理 ─────────────────────────

.PHONY: images rmi prune

## 查看镜像列表
images:
	docker images

## 删除项目镜像
rmi:
	docker rmi digit_recognizer_quant-backend digit_recognizer_quant-frontend

## 清除构建缓存
prune:
	docker builder prune -f


# ─── Docker 完全清理 ─────────────────────────

.PHONY: clean clean-all

## 停止并删除容器 + 删除镜像
clean: down rmi

## 停止并删除容器 + 删除镜像 + 清除缓存
clean-all: down rmi prune


# ─── 本地开发 ────────────────────────────────

.PHONY: dev-backend dev-frontend install install-frontend

## 启动本地后端（Gunicorn）
dev-backend:
	gunicorn api.main:app -c gunicorn.conf.py

## 启动本地前端（Vite 开发服务器）
dev-frontend:
	cd frontend && npm run dev

## 安装后端依赖（uv）
install:
	uv sync

## 安装前端依赖（npm）
install-frontend:
	cd frontend && npm install


# ─── 测试 ────────────────────────────────────

.PHONY: test test-backend test-frontend

## 运行后端测试
test-backend:
	uv run pytest

## 运行前端测试
test-frontend:
	cd frontend && npm run test


# ─── 帮助 ────────────────────────────────────

.PHONY: help

## 显示帮助信息
help:
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║  Digit Recognizer - Makefile Help           ║"
	@echo "╠══════════════════════════════════════════════╣"
	@echo "║                                              ║"
	@echo "║  ┌─ Docker 服务管理 ─────────────────────┐  ║"
	@echo "║  │  make build       构建镜像             │  ║"
	@echo "║  │  make build-no-cache  不使用缓存构建   │  ║"
	@echo "║  │  make up          构建并启动服务       │  ║"
	@echo "║  │  make start       启动（已有镜像）     │  ║"
	@echo "║  │  make stop        停止服务             │  ║"
	@echo "║  │  make down        停止并删除容器       │  ║"
	@echo "║  │  make restart     重启服务             │  ║"
	@echo "║  │  make ps          查看运行状态         │  ║"
	@echo "║  │  make logs        查看所有日志         │  ║"
	@echo "║  │  make logs-backend   查看后端日志      │  ║"
	@echo "║  │  make logs-frontend  查看前端日志      │  ║"
	@echo "║  └──────────────────────────────────────┘  ║"
	@echo "║                                              ║"
	@echo "║  ┌─ Docker 镜像管理 ─────────────────────┐  ║"
	@echo "║  │  make images     查看镜像列表          │  ║"
	@echo "║  │  make rmi        删除项目镜像          │  ║"
	@echo "║  │  make prune      清除构建缓存          │  ║"
	@echo "║  └──────────────────────────────────────┘  ║"
	@echo "║                                              ║"
	@echo "║  ┌─ Docker 完全清理 ─────────────────────┐  ║"
	@echo "║  │  make clean      停止+删除容器+镜像    │  ║"
	@echo "║  │  make clean-all  完全清理（含缓存）    │  ║"
	@echo "║  └──────────────────────────────────────┘  ║"
	@echo "║                                              ║"
	@echo "║  ┌─ 本地开发 ───────────────────────────┐  ║"
	@echo "║  │  make dev-backend    启动本地后端     │  ║"
	@echo "║  │  make dev-frontend   启动本地前端     │  ║"
	@echo "║  │  make install       安装后端依赖      │  ║"
	@echo "║  │  make install-frontend  安装前端依赖  │  ║"
	@echo "║  └──────────────────────────────────────┘  ║"
	@echo "║                                              ║"
	@echo "║  ┌─ 测试 ───────────────────────────────┐  ║"
	@echo "║  │  make test-backend   运行后端测试     │  ║"
	@echo "║  │  make test-frontend  运行前端测试     │  ║"
	@echo "║  └──────────────────────────────────────┘  ║"
	@echo "╚══════════════════════════════════════════════╝"
