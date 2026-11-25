# ==============================================================================
# OPV-PredictNet — Frontend Dockerfile
# Multi-stage build: Node.js 20 build stage -> Nginx Alpine production image
# ==============================================================================

# Build Stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy frontend source files
COPY frontend/ ./

# Build production bundle with Vite
RUN npm run build

# Production Stage
FROM nginx:alpine AS runner

# Remove default nginx configurations
RUN rm -rf /etc/nginx/conf.d/*

# Copy custom Nginx SPA and reverse proxy configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
