# ============================================
# Frontend Dockerfile - Digit Recognizer UI
# Multi-stage build: Build (Node) + Serve (Nginx)
# ============================================
# ---- Stage 1: Build ----
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files and install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source code and build
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Nginx ----
FROM nginx:alpine

# Copy built static files from stage 1
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom nginx config for API proxy
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
