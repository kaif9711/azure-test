# Simple Nginx-based static build for the Fraud Detection Frontend
# Multi-stage kept minimal in case future asset build is added.

FROM nginx:1.27-alpine

# Labels
LABEL maintainer="project-team" \
      description="Static frontend for Fraudulent Claim Detection Agent" \
      org.opencontainers.image.source="https://example.com/repo"

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static assets
COPY index.html /usr/share/nginx/html/
COPY css /usr/share/nginx/html/css
COPY js /usr/share/nginx/html/js

# Add entrypoint to inject runtime configuration (API base URL etc.)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV API_BASE_URL="http://localhost:8000" \
    ENABLE_STRICT_CSP="false"

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget -q -O - http://localhost/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
