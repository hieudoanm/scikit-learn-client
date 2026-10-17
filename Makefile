.PHONY: help up down start stop e2e e2e-demo

FRONTEND_DIR := packages/frontend
BACKEND_DIR := packages/backend
DEMO_DIR := demo
DEMO_VIDEO := $(DEMO_DIR)/demo.mp4
BACKEND_PORT := 8000
FRONTEND_PORT := 3000
BACKEND_LOG := /tmp/skl-backend.log
FRONTEND_LOG := /tmp/skl-frontend.log
BACKEND_PID := /tmp/skl-backend.pid
FRONTEND_PID := /tmp/skl-frontend.pid
# The compose stack publishes Postgres here by default (see docker-compose.yaml
# and .env.example); override with `make start POSTGRES_PORT=5433` to keep 5432
# free for another local PostgreSQL.
POSTGRES_PORT ?= 5432
DATABASE_URL ?= postgresql+asyncpg://skl:skl@localhost:$(POSTGRES_PORT)/skl
# The onnx extra is optional in pyproject.toml, so `uv run` alone would leave it out.
BACKEND_UV_RUN := uv run --python 3.14.8 --extra onnx
# Stops host-mode services and whatever still holds their ports (make start /
# make stop only — never run it against the compose stack).
STOP_SERVICES = test -f $(BACKEND_PID) && xargs -r kill -9 < $(BACKEND_PID) 2>/dev/null || true; test -f $(FRONTEND_PID) && xargs -r kill -9 < $(FRONTEND_PID) 2>/dev/null || true; lsof -ti :$(BACKEND_PORT) | xargs -r kill -9 2>/dev/null || true; lsof -ti :$(FRONTEND_PORT) | xargs -r kill -9 2>/dev/null || true; rm -f $(BACKEND_PID) $(FRONTEND_PID)
# Force-frees the published ports (backend, frontend, Postgres) so `make up` /
# `make start` never trip over a stale host process or container. This kills
# whatever holds the port, including a non-Docker Postgres on $(POSTGRES_PORT).
FREE_PORTS = for port in $(BACKEND_PORT) $(FRONTEND_PORT) $(POSTGRES_PORT); do lsof -ti :$$port | xargs -r kill -9 2>/dev/null || true; done
# Runs docker compose with the make-level POSTGRES_PORT (a shell-env prefix
# beats .env, so `make start POSTGRES_PORT=5433` republishes Postgres too).
COMPOSE = POSTGRES_PORT=$(POSTGRES_PORT) docker compose

help: ## Show available make targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start the full stack with Docker Compose (Postgres, backend, frontend)
	@echo "==> Freeing ports $(POSTGRES_PORT), $(BACKEND_PORT), and $(FRONTEND_PORT)..."
	-@$(FREE_PORTS)
	@echo "==> Starting the stack with Docker Compose (Postgres :$(POSTGRES_PORT), backend :$(BACKEND_PORT), frontend :$(FRONTEND_PORT))..."
	@$(COMPOSE) up -d --build || { echo "ERROR: docker compose up failed."; echo "If host-mode services are holding :$(BACKEND_PORT)/:$(FRONTEND_PORT), run 'make stop' first."; exit 1; }
	@echo "==> Waiting for services to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://127.0.0.1:$(BACKEND_PORT)/healthz > /dev/null 2>&1 && curl -sf http://127.0.0.1:$(FRONTEND_PORT) > /dev/null 2>&1; then \
			echo "==> Stack ready."; \
			echo "    Backend:  http://127.0.0.1:$(BACKEND_PORT) (docs: /docs)"; \
			echo "    Frontend: http://127.0.0.1:$(FRONTEND_PORT)"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then echo "ERROR: services not ready"; docker compose ps; docker compose logs --tail=20 backend frontend; exit 1; fi; \
		sleep 2; \
	done

down: ## Stop the Docker Compose stack (containers and data volumes are kept)
	-@docker compose stop
	@echo "==> Stack stopped. 'make up' brings it back; 'docker compose down' removes it."

start: ## Run backend (:8000) and frontend (:3000) on the host (Postgres via Compose)
	@echo "==> Freeing ports $(POSTGRES_PORT), $(BACKEND_PORT), and $(FRONTEND_PORT)..."
	-@$(FREE_PORTS)
	@echo "==> Ensuring Postgres is up via Docker Compose..."
	-@docker compose stop backend frontend
	@$(COMPOSE) up -d postgres
	@echo "==> Stopping any host-mode services on ports $(BACKEND_PORT) and $(FRONTEND_PORT)..."
	-@$(STOP_SERVICES)
	@echo "==> Starting backend on port $(BACKEND_PORT)..."
	cd $(BACKEND_DIR) && DATABASE_URL=$(DATABASE_URL) skl_DATA_DIR=/tmp/skl-data $(BACKEND_UV_RUN) python -m uvicorn main:app --host 127.0.0.1 --port $(BACKEND_PORT) > $(BACKEND_LOG) 2>&1 & echo $$! > $(BACKEND_PID)
	@echo "==> Starting frontend on port $(FRONTEND_PORT)..."
	cd $(FRONTEND_DIR) && pnpm run dev > $(FRONTEND_LOG) 2>&1 & echo $$! > $(FRONTEND_PID)
	@echo "==> Waiting for services to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://127.0.0.1:$(BACKEND_PORT)/healthz > /dev/null 2>&1 && curl -sf http://127.0.0.1:$(FRONTEND_PORT) > /dev/null 2>&1; then \
			echo "==> Services ready."; \
			echo "    Backend:  http://127.0.0.1:$(BACKEND_PORT)"; \
			echo "    Frontend: http://127.0.0.1:$(FRONTEND_PORT)"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then echo "ERROR: services not ready"; echo "--- $(BACKEND_LOG) ---"; tail -20 $(BACKEND_LOG) 2>/dev/null; echo "--- $(FRONTEND_LOG) ---"; tail -20 $(FRONTEND_LOG) 2>/dev/null; $(STOP_SERVICES); exit 1; fi; \
		sleep 2; \
	done

stop: ## Stop the host-mode backend and frontend
	-@$(STOP_SERVICES)
	@echo "==> Host services stopped."

e2e: ## Run e2e tests only (requires backend on :8000 and frontend on :3000)
	cd $(FRONTEND_DIR) && pnpm exec playwright test

e2e-demo: ## Capture all demo scenarios and save as demo/demo.mp4 and demo/demo.gif
	@echo "==> Starting the stack with Docker Compose..."
	@$(COMPOSE) up -d --build
	@echo "==> Waiting for services to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://127.0.0.1:$(BACKEND_PORT)/healthz > /dev/null 2>&1 && curl -sf http://127.0.0.1:$(FRONTEND_PORT) > /dev/null 2>&1; then \
			echo "==> Services ready."; break; \
		fi; \
		if [ $$i -eq 30 ]; then echo "ERROR: services not ready"; docker compose ps; docker compose logs --tail=20 backend frontend; exit 1; fi; \
		sleep 2; \
	done
	@echo "==> Recording demo video (single worker, so scenes stay in order)..."
	@rm -rf $(FRONTEND_DIR)/test-results
	@cd $(FRONTEND_DIR) && DEMO=1 pnpm exec playwright test e2e/skl.e2e.test.ts --reporter=list --workers=1 || { \
		status=$$?; echo "==> Recording failed"; docker compose stop backend frontend; exit $$status; \
	}
	@echo "==> Saving video to $(DEMO_VIDEO)..."
	@mkdir -p $(DEMO_DIR)
	@VIDEO_FILES=$$(cd $(FRONTEND_DIR) && find test-results -name "video.webm" -type f 2>/dev/null | sort -t- -k4,4n); \
	if [ -n "$$VIDEO_FILES" ]; then \
		echo "$$VIDEO_FILES" | sed 's|^|file '"'"'$(CURDIR)/$(FRONTEND_DIR)/|;s|$$|'"'"'|' > $(DEMO_DIR)/concat.txt; \
		if command -v ffmpeg > /dev/null 2>&1; then \
			ffmpeg -y -f concat -safe 0 -i $(DEMO_DIR)/concat.txt -c:v libx264 -pix_fmt yuv420p -crf 23 -preset medium "$(DEMO_VIDEO)" && rm -f $(DEMO_DIR)/concat.txt; \
			echo "==> Converting to GIF..."; \
			ffmpeg -y -i "$(DEMO_VIDEO)" -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $(DEMO_DIR)/demo.gif || echo "WARNING: GIF conversion failed"; \
		else \
			echo "WARNING: ffmpeg not found, cannot concatenate/convert"; \
		fi; \
		echo "==> Demo saved to $(DEMO_VIDEO) and demo.gif"; \
	else \
		echo "WARNING: no video file found"; \
	fi
	@echo "==> Stopping services..."
	-@docker compose stop backend frontend

reup: down up
