# justfile - Cross-platform task runner
# Install just: https://github.com/casey/just#installation
# Usage: just <command>

set shell := ["bash", "-cu"]
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

python := "python"
ruff := "ruff"
pytest := "pytest"
mypy := "mypy"

source_dir := "src/automas"
tests_dir := "tests"
examples_dir := "examples"
all_dirs := source_dir + " " + tests_dir + " " + examples_dir

searxng_dir := env_var_or_default('SEARXNG_DIR', '~/searxng-docker')
searxng_port := env_var_or_default('SEARXNG_PORT', '8888')

# Show available commands
default:
    @just --list

# Code Quality

# Run Ruff linting
lint:
    @echo "Running Ruff linting..."
    {{ruff}} check {{all_dirs}}

# Fix linting errors
fix:
    @echo "Fixing linting errors..."
    {{ruff}} check --fix {{all_dirs}}

# Format code
format:
    @echo "Formatting code..."
    {{ruff}} format {{all_dirs}}

# Sort imports
sort:
    @echo "Sorting imports..."
    {{ruff}} check --select I --fix {{all_dirs}}

# Format and sort
format-sort: format sort

# Run all quality checks
all: lint fix format sort

# Testing

# Run tests
tests:
    @echo "Running tests..."
    {{python}} -m {{pytest}} {{tests_dir}}

# Run mypy type checking
mypy:
    @echo "Running mypy type check..."
    {{python}} -m {{mypy}} --strict {{source_dir}} --exclude 'site|venv|\.venv'

# SearXNG Management (Unix)

# Install SearXNG (Unix)
[unix]
searxng-install:
    #!/usr/bin/env bash
    set -euo pipefail
    INSTALL_DIR=$(eval echo {{searxng_dir}})
    PORT={{searxng_port}}

    if [ -d "$INSTALL_DIR" ]; then
        echo "Directory already exists: $INSTALL_DIR"
        exit 1
    fi

    echo "Cloning SearXNG..."
    git clone https://github.com/searxng/searxng-docker.git "$INSTALL_DIR"

    echo "Configuring settings..."
    cd "$INSTALL_DIR"

    SECRET_KEY=$(openssl rand -hex 32)

    cat > searxng/settings.yml << EOF
    # see https://docs.searxng.org/admin/settings/settings.html#settings-use-default-settings
    use_default_settings: true
    server:
      # base_url is defined in the SEARXNG_BASE_URL environment variable, see .env and docker-compose.yml
      secret_key: "$SECRET_KEY"
      limiter: false  # enable this when running the instance for a public usage on the internet
      image_proxy: true
    redis:
      url: redis://redis:6379/0
    search:
      formats:
        - html
        - json
        - csv
        - rss
    EOF

    # Update docker-compose.yml to use custom port
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/127.0.0.1:8080:8080/127.0.0.1:$PORT:8080/" docker-compose.yaml
    else
        sed -i "s/127.0.0.1:8080:8080/127.0.0.1:$PORT:8080/" docker-compose.yaml
    fi

    echo "Installed at $INSTALL_DIR"
    echo "SearXNG will run on port $PORT"
    echo "Run: just searxng-start"

# Start SearXNG (Unix)
[unix]
searxng-start:
    @cd $(eval echo {{searxng_dir}}) && docker-compose up -d
    @echo "SearXNG started at http://localhost:{{searxng_port}}"

# Stop SearXNG (Unix)
[unix]
searxng-stop:
    @cd $(eval echo {{searxng_dir}}) && docker-compose down

# Show SearXNG status (Unix)
[unix]
searxng-status:
    @cd $(eval echo {{searxng_dir}}) && docker-compose ps

# SearXNG Management (Windows)

# Install SearXNG (Windows)
[windows]
searxng-install:
    @$dir = "{{searxng_dir}}" -replace "~", $env:USERPROFILE; \
    if (Test-Path $dir) { Write-Host "Directory exists: $dir"; exit 1 }; \
    Write-Host "Cloning SearXNG..."; \
    git clone https://github.com/searxng/searxng-docker.git $dir; \
    cd $dir; \
    Write-Host "Configuring settings..."; \
    $key = -join ((1..64) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) }); \
    $settings = @" \
    # see https://docs.searxng.org/admin/settings/settings.html#settings-use-default-settings`n\
    use_default_settings: true`n\
    server:`n\
      # base_url is defined in the SEARXNG_BASE_URL environment variable, see .env and docker-compose.yml`n\
      secret_key: "$key"`n\
      limiter: false  # enable this when running the instance for a public usage on the internet`n\
      image_proxy: true`n\
    redis:`n\
      url: redis://redis:6379/0`n\
    search:`n\
      formats:`n\
        - html`n\
        - json`n\
        - csv`n\
        - rss`n\
    "@; \
    $settings | Set-Content searxng/settings.yml; \
    Write-Host "Installed at $dir"; \
    Write-Host "Run: just searxng-start"

# Start SearXNG (Windows)
[windows]
searxng-start:
    @$dir = "{{searxng_dir}}" -replace "~", $env:USERPROFILE; \
    cd $dir; \
    docker-compose up -d; \
    Write-Host "SearXNG started at http://localhost:8080"

# Stop SearXNG (Windows)
[windows]
searxng-stop:
    @$dir = "{{searxng_dir}}" -replace "~", $env:USERPROFILE; \
    cd $dir; \
    docker-compose down

# Show SearXNG status (Windows)
[windows]
searxng-status:
    @$dir = "{{searxng_dir}}" -replace "~", $env:USERPROFILE; \
    cd $dir; \
    docker-compose ps

# GAIA Benchmark - Universal Scripts Runner

# Run any GAIA script with custom parameters (e.g., just gaia-run run_gaia --difficulty=1 --split="validation[:5]")
gaia-run script *args:
    #!/usr/bin/env bash
    set -euo pipefail
    SCRIPT_PATH="examples/gaia/{{script}}.py"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: Script not found: $SCRIPT_PATH"
        echo ""
        echo "Available scripts:"
        ls -1 examples/gaia/run_*.py | xargs -n1 basename | sed 's/\.py$//'
        exit 1
    fi

    echo "Running $SCRIPT_PATH with arguments: {{args}}"
    {{python}} "$SCRIPT_PATH" {{args}}

# List all available GAIA runner scripts
gaia-list:
    @echo "Available GAIA runner scripts:"
    @ls -1 examples/gaia/run_*.py | xargs -n1 basename | sed 's/\.py$//' | sed 's/^/  - /'

# Run specific GAIA script with batch support (e.g., just gaia-script-batch run_gaia 0 all)
gaia-script-batch script batch_num difficulty="all" *extra_args:
    #!/usr/bin/env bash
    set -euo pipefail
    BATCH_NUM={{batch_num}}
    BATCH_SIZE=10
    START=$((BATCH_NUM * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    SPLIT="validation[$START:$END]"

    echo "Running {{script}} batch {{batch_num}}: questions $START-$((END-1))"
    echo "Difficulty: {{difficulty}}"
    echo "Split: $SPLIT"

    just gaia-run {{script}} --difficulty {{difficulty}} --split "$SPLIT" {{extra_args}}

# Run specific GAIA script on all batches (e.g., just gaia-script-all run_gaia_evo all)
gaia-script-all script difficulty="all" *extra_args:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running {{script}} on all 17 batches (165 questions total)"
    echo "Difficulty: {{difficulty}}"

    for i in {0..16}; do
        echo ""
        echo "========================================="
        echo "Starting batch $i/16"
        echo "========================================="
        just gaia-script-batch {{script}} $i {{difficulty}} {{extra_args}}

        if [ $i -lt 16 ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "All batches completed!"
    echo "========================================="

# Run specific GAIA script on a range of batches (e.g., just gaia-script-range run_gaia_unified 0 5 all)
gaia-script-range script start_batch end_batch difficulty="all" *extra_args:
    #!/usr/bin/env bash
    set -euo pipefail
    START={{start_batch}}
    END={{end_batch}}

    if [ $START -lt 0 ] || [ $START -gt 16 ]; then
        echo "Error: start_batch must be between 0 and 16"
        exit 1
    fi

    if [ $END -lt $START ] || [ $END -gt 16 ]; then
        echo "Error: end_batch must be between start_batch and 16"
        exit 1
    fi

    echo "Running {{script}} batches $START to $END"
    echo "Difficulty: {{difficulty}}"

    for i in $(seq $START $END); do
        echo ""
        echo "========================================="
        echo "Starting batch $i/$END"
        echo "========================================="
        just gaia-script-batch {{script}} $i {{difficulty}} {{extra_args}}

        if [ $i -lt $END ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "Batches $START-$END completed!"
    echo "========================================="

# GAIA Benchmark - With Judge (Original Commands)

# Run GAIA benchmark on a specific batch (batch_num: 0-16 for 165 questions split by 10)
gaia-batch batch_num difficulty="all" max_iter="3":
    #!/usr/bin/env bash
    set -euo pipefail
    BATCH_NUM={{batch_num}}
    BATCH_SIZE=10
    START=$((BATCH_NUM * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    SPLIT="validation[$START:$END]"

    echo "Running GAIA batch {{batch_num}}: questions $START-$((END-1))"
    echo "Difficulty: {{difficulty}}, Max iterations: {{max_iter}}"
    echo "Split: $SPLIT"

    {{python}} examples/gaia/run_gaia_with_judge.py \
        --difficulty {{difficulty}} \
        --split "$SPLIT" \
        --max-iter {{max_iter}}

# Run all GAIA batches sequentially (0-16, covering all 165 questions)
gaia-all difficulty="all" max_iter="3":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running all 17 batches (165 questions total)"
    echo "Difficulty: {{difficulty}}, Max iterations: {{max_iter}}"

    for i in {0..16}; do
        echo ""
        echo "========================================="
        echo "Starting batch $i/16"
        echo "========================================="
        just gaia-batch $i {{difficulty}} {{max_iter}}

        if [ $i -lt 16 ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "All batches completed!"
    echo "========================================="

# Run a range of GAIA batches (e.g., just gaia-range 0 5)
gaia-range start_batch end_batch difficulty="all" max_iter="3":
    #!/usr/bin/env bash
    set -euo pipefail
    START={{start_batch}}
    END={{end_batch}}

    if [ $START -lt 0 ] || [ $START -gt 16 ]; then
        echo "Error: start_batch must be between 0 and 16"
        exit 1
    fi

    if [ $END -lt $START ] || [ $END -gt 16 ]; then
        echo "Error: end_batch must be between start_batch and 16"
        exit 1
    fi

    echo "Running GAIA batches $START to $END"
    echo "Difficulty: {{difficulty}}, Max iterations: {{max_iter}}"

    for i in $(seq $START $END); do
        echo ""
        echo "========================================="
        echo "Starting batch $i/$END"
        echo "========================================="
        just gaia-batch $i {{difficulty}} {{max_iter}}

        if [ $i -lt $END ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "Batches $START-$END completed!"
    echo "========================================="

# Run remaining batches starting from a specific batch number
gaia-resume from_batch difficulty="all" max_iter="3":
    @just gaia-range {{from_batch}} 16 {{difficulty}} {{max_iter}}

# GAIA Benchmark (Containerized - Cross-platform)

container_cmd := if `command -v podman > /dev/null 2>&1 && echo "1" || echo "0"` == "1" { "podman-compose" } else { "docker-compose" }

# Run any GAIA script in container with custom parameters
gaia-docker-run script *args:
    #!/usr/bin/env bash
    set -euo pipefail
    SCRIPT_PATH="examples/gaia/{{script}}.py"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: Script not found: $SCRIPT_PATH"
        echo ""
        echo "Available scripts:"
        ls -1 examples/gaia/run_*.py | xargs -n1 basename | sed 's/\.py$//'
        exit 1
    fi

    echo "Using container runtime: {{container_cmd}}"
    echo "Running $SCRIPT_PATH in container with arguments: {{args}}"
    echo ""

    mkdir -p examples/gaia/gaia_logs
    mkdir -p automas_logs

    {{container_cmd}} build
    {{container_cmd}} run --rm automas \
        python3 "$SCRIPT_PATH" {{args}}

# Run specific GAIA script in container with batch support
gaia-docker-script-batch script batch_num difficulty="all" *extra_args:
    #!/usr/bin/env bash
    set -euo pipefail
    BATCH_NUM={{batch_num}}
    BATCH_SIZE=10
    START=$((BATCH_NUM * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    SPLIT="validation[$START:$END]"

    echo "Using container runtime: {{container_cmd}}"
    echo "Running {{script}} batch {{batch_num}}: questions $START-$((END-1))"
    echo "Difficulty: {{difficulty}}"
    echo "Split: $SPLIT"
    echo ""

    mkdir -p examples/gaia/gaia_logs
    mkdir -p automas_logs

    {{container_cmd}} build
    {{container_cmd}} run --rm automas \
        python3 examples/gaia/{{script}}.py \
        --difficulty {{difficulty}} \
        --split "$SPLIT" {{extra_args}}

# Run specific GAIA script in container on all batches
gaia-docker-script-all script difficulty="all" *extra_args:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Using container runtime: {{container_cmd}}"
    echo "Running {{script}} on all 17 batches (165 questions total) in container"
    echo "Difficulty: {{difficulty}}"
    echo ""

    mkdir -p examples/gaia/gaia_logs
    mkdir -p automas_logs

    echo "Building container image..."
    {{container_cmd}} build
    echo ""

    for i in {0..16}; do
        BATCH_SIZE=10
        START=$((i * BATCH_SIZE))
        END=$((START + BATCH_SIZE))
        SPLIT="validation[$START:$END]"

        echo ""
        echo "========================================="
        echo "Starting batch $i/16"
        echo "========================================="

        {{container_cmd}} run --rm automas \
            python3 examples/gaia/{{script}}.py \
            --difficulty {{difficulty}} \
            --split "$SPLIT" {{extra_args}}

        if [ $i -lt 16 ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "All batches completed!"
    echo "========================================="
    echo "Results saved to: examples/gaia/gaia_logs/"

# Run specific GAIA script in container on a range of batches
gaia-docker-script-range script start_batch end_batch difficulty="all" *extra_args:
    #!/usr/bin/env bash
    set -euo pipefail
    START={{start_batch}}
    END={{end_batch}}

    if [ $START -lt 0 ] || [ $START -gt 16 ]; then
        echo "Error: start_batch must be between 0 and 16"
        exit 1
    fi

    if [ $END -lt $START ] || [ $END -gt 16 ]; then
        echo "Error: end_batch must be between start_batch and 16"
        exit 1
    fi

    echo "Using container runtime: {{container_cmd}}"
    echo "Running {{script}} batches $START to $END in container"
    echo "Difficulty: {{difficulty}}"
    echo ""

    mkdir -p examples/gaia/gaia_logs
    mkdir -p automas_logs

    echo "Building container image..."
    {{container_cmd}} build
    echo ""

    for i in $(seq $START $END); do
        BATCH_SIZE=10
        BATCH_START=$((i * BATCH_SIZE))
        BATCH_END=$((BATCH_START + BATCH_SIZE))
        SPLIT="validation[$BATCH_START:$BATCH_END]"

        echo ""
        echo "========================================="
        echo "Starting batch $i/$END"
        echo "========================================="

        {{container_cmd}} run --rm automas \
            python3 examples/gaia/{{script}}.py \
            --difficulty {{difficulty}} \
            --split "$SPLIT" {{extra_args}}

        if [ $i -lt $END ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "Batches $START-$END completed!"
    echo "========================================="
    echo "Results saved to: examples/gaia/gaia_logs/"

# GAIA Benchmark (Containerized) - With Judge (Original Commands)

# Run GAIA benchmark in container on a specific batch
gaia-docker-batch batch_num difficulty="all" max_iter="3":
    #!/usr/bin/env bash
    set -euo pipefail
    BATCH_NUM={{batch_num}}
    BATCH_SIZE=10
    START=$((BATCH_NUM * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    SPLIT="validation[$START:$END]"

    echo "Using container runtime: {{container_cmd}}"
    echo "Running GAIA batch {{batch_num}}: questions $START-$((END-1))"
    echo "Difficulty: {{difficulty}}, Max iterations: {{max_iter}}"
    echo "Split: $SPLIT"
    echo ""

    mkdir -p examples/gaia/gaia_logs
    mkdir -p automas_logs

    {{container_cmd}} build
    {{container_cmd}} run --rm automas \
        python3 examples/gaia/run_gaia_with_judge.py \
        --difficulty {{difficulty}} \
        --split "$SPLIT" \
        --max-iter {{max_iter}}

# Run all GAIA batches in container sequentially
gaia-docker-all difficulty="all" max_iter="3":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Using container runtime: {{container_cmd}}"
    echo "Running all 17 batches (165 questions total) in container"
    echo "Difficulty: {{difficulty}}, Max iterations: {{max_iter}}"
    echo ""

    mkdir -p examples/gaia/gaia_logs
    mkdir -p automas_logs

    echo "Building container image..."
    {{container_cmd}} build
    echo ""

    for i in {0..16}; do
        BATCH_SIZE=10
        START=$((i * BATCH_SIZE))
        END=$((START + BATCH_SIZE))
        SPLIT="validation[$START:$END]"

        echo ""
        echo "========================================="
        echo "Starting batch $i/16"
        echo "========================================="

        {{container_cmd}} run --rm automas \
            python3 examples/gaia/run_gaia_with_judge.py \
            --difficulty {{difficulty}} \
            --split "$SPLIT" \
            --max-iter {{max_iter}}

        if [ $i -lt 16 ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "All batches completed!"
    echo "========================================="
    echo "Results saved to: examples/gaia/gaia_logs/"

# Run a range of GAIA batches in container
gaia-docker-range start_batch end_batch difficulty="all" max_iter="3":
    #!/usr/bin/env bash
    set -euo pipefail
    START={{start_batch}}
    END={{end_batch}}

    if [ $START -lt 0 ] || [ $START -gt 16 ]; then
        echo "Error: start_batch must be between 0 and 16"
        exit 1
    fi

    if [ $END -lt $START ] || [ $END -gt 16 ]; then
        echo "Error: end_batch must be between start_batch and 16"
        exit 1
    fi

    echo "Using container runtime: {{container_cmd}}"
    echo "Running GAIA batches $START to $END in container"
    echo "Difficulty: {{difficulty}}, Max iterations: {{max_iter}}"
    echo ""

    mkdir -p examples/gaia/gaia_logs
    mkdir -p automas_logs

    echo "Building container image..."
    {{container_cmd}} build
    echo ""

    for i in $(seq $START $END); do
        BATCH_SIZE=10
        BATCH_START=$((i * BATCH_SIZE))
        BATCH_END=$((BATCH_START + BATCH_SIZE))
        SPLIT="validation[$BATCH_START:$BATCH_END]"

        echo ""
        echo "========================================="
        echo "Starting batch $i/$END"
        echo "========================================="

        {{container_cmd}} run --rm automas \
            python3 examples/gaia/run_gaia_with_judge.py \
            --difficulty {{difficulty}} \
            --split "$SPLIT" \
            --max-iter {{max_iter}}

        if [ $i -lt $END ]; then
            echo ""
            echo "Batch $i completed. Waiting 5 seconds before next batch..."
            sleep 5
        fi
    done

    echo ""
    echo "========================================="
    echo "Batches $START-$END completed!"
    echo "========================================="
    echo "Results saved to: examples/gaia/gaia_logs/"

# Run remaining batches in container starting from a specific batch number
gaia-docker-resume from_batch difficulty="all" max_iter="3":
    @just gaia-docker-range {{from_batch}} 16 {{difficulty}} {{max_iter}}
