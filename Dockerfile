FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_COMPILE_BYTECODE=1

# Copy the project files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (this creates the .venv in the image)
RUN uv sync --frozen --no-install-project --no-dev

# Copy the application and source code
COPY app ./app
COPY src ./src
COPY config ./config

# Install the project
RUN uv sync --frozen --no-dev

# Expose the port
EXPOSE 8000

# Run the application using uv run to ensure the venv is used
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
