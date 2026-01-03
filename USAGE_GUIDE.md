"""Quick Reference: Using New Modules

This guide shows how to use the newly implemented modules for error handling,
async operations, logging, and retry logic.

## 1. Custom Exceptions

### Import
```python
from openai_sdk_helpers import (
    OpenAISDKError,
    ConfigurationError,
    PromptNotFoundError,
    AgentExecutionError,
    VectorStorageError,
    AsyncExecutionError,
    InputValidationError,
)
```

### Raising Exceptions with Context
```python
# Simple error
raise ConfigurationError("Missing API key")

# Error with debugging context
raise VectorStorageError(
    "Failed to upload file",
    context={
        "file_path": "/path/to/file.txt",
        "vector_store_id": "vs_123",
        "attempt": 2,
    }
)

# Error chaining
try:
    openai_client.files.upload(...)
except FileNotFoundError as e:
    raise PromptNotFoundError(
        "Prompt file not found", 
        context={"expected_path": str(path)}
    ) from e
```

### Catching Exceptions
```python
# Catch specific errors
try:
    storage.upload_file("data.txt")
except VectorStorageError as e:
    print(f"Upload failed: {e}")
    print(f"Context: {e.context}")

# Catch any SDK error
try:
    result = agent.run()
except OpenAISDKError as e:
    # Handles all custom exceptions
    log(f"SDK Error: {e}", level=logging.ERROR)
```

---

## 2. Async Utilities

### Import
```python
from openai_sdk_helpers import (
    run_coroutine_thread_safe,
    run_coroutine_with_fallback,
)
```

### Running Async Code from Sync Context
```python
# Automatically detects if event loop is running
async def fetch_data():
    return "data"

# Safe to call from sync code, handles nested event loops
result = run_coroutine_with_fallback(fetch_data())

# Or use thread-safe variant directly
result = run_coroutine_thread_safe(fetch_data(), timeout=30.0)
```

### Exception Handling
```python
try:
    result = run_coroutine_thread_safe(slow_operation(), timeout=5.0)
except AsyncExecutionError as e:
    print(f"Async operation timed out: {e}")
except Exception as e:
    # Original exception from coroutine is re-raised
    print(f"Operation failed: {e}")
```

---

## 3. Logging Configuration

### Import
```python
from openai_sdk_helpers import LoggerFactory
import logging
```

### Configure Once at Startup
```python
# Configure with default settings
LoggerFactory.configure(level=logging.INFO)

# Configure with custom handlers
import logging.handlers

handlers = [
    logging.StreamHandler(),  # Console output
    logging.FileHandler("app.log"),  # File output
]
LoggerFactory.configure(level=logging.DEBUG, handlers=handlers)
```

### Use in Modules
```python
# In any module
logger = LoggerFactory.get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Example with Sentry
```python
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Set up Sentry
sentry_logging = LoggingIntegration(
    level=logging.DEBUG,
    event_level=logging.WARNING
)
sentry_sdk.init(
    "https://...", 
    integrations=[sentry_logging]
)

# Configure SDK logging
LoggerFactory.configure(level=logging.DEBUG)

# Now all logs are sent to both console and Sentry
logger = LoggerFactory.get_logger("app")
logger.error("This will be sent to Sentry")
```

---

## 4. Retry Decorator

### Import
```python
from openai_sdk_helpers import with_exponential_backoff
```

### Basic Usage
```python
@with_exponential_backoff(max_retries=3, base_delay=1.0)
def call_openai_api(query: str) -> str:
    # OpenAI API call
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    ).choices[0].message.content
```

### Async Functions
```python
@with_exponential_backoff(max_retries=3, base_delay=1.0)
async def fetch_embeddings(texts: list[str]) -> list[list[float]]:
    response = await async_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
```

### With Parameters
```python
@with_exponential_backoff(
    max_retries=5,
    base_delay=2.0,  # Start with 2 second delay
    max_delay=30.0   # Cap at 30 seconds
)
def upload_file(file_path: str, store_id: str) -> dict:
    with open(file_path, "rb") as f:
        return openai_client.files.create(
            file=f,
            purpose="assistants"
        )
```

### What It Retries On
```
✅ Retries:
   - RateLimitError (automatic exponential backoff)
   - APIError with status 408 (Request Timeout)
   - APIError with status 429 (Too Many Requests)
   - APIError with status 500+ (Server errors)

❌ Does Not Retry:
   - APIError with status 4xx (except 408, 429)
   - Other exceptions (re-raised immediately)
```

---

## 5. Integration Example: Complete Agent Setup

```python
from openai_sdk_helpers import (
    LoggerFactory,
    ConfigurationError,
    AgentBase,
    with_exponential_backoff,
)
import logging

# Configure logging at startup
LoggerFactory.configure(level=logging.INFO)
logger = LoggerFactory.get_logger("app")

try:
    # Create agent with retry protection
    @with_exponential_backoff(max_retries=3)
    def create_agent(model: str) -> AgentBase:
        if not model:
            raise ConfigurationError("Model is required")
        return AgentBase(
            config=AgentConfig(model=model),
            default_model=model
        )
    
    agent = create_agent("gpt-4")
    logger.info("Agent created successfully")
    
    # Run agent safely
    from openai_sdk_helpers import run_coroutine_with_fallback
    
    async def run_agent():
        return await agent.run_async("Hello!")
    
    result = run_coroutine_with_fallback(run_agent())
    logger.info(f"Agent result: {result}")

except ConfigurationError as e:
    logger.error(f"Configuration problem: {e}")
    logger.error(f"Context: {e.context}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
```

---

## 6. Migration Guide

### Before (Old Pattern)
```python
import asyncio
import threading

# Problematic daemon thread approach
def run_agent_sync(agent, query):
    coro = agent.run_async(query)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    
    if loop.is_running():
        result = None
        def _runner():
            nonlocal result
            result = asyncio.run(coro)  # May raise silently
        
        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if result is None:
            raise RuntimeError("No result")  # Generic error
        return result
    
    return loop.run_until_complete(coro)
```

### After (New Pattern)
```python
from openai_sdk_helpers import run_coroutine_with_fallback

def run_agent_sync(agent, query):
    return run_coroutine_with_fallback(agent.run_async(query))
```

---

## Best Practices

1. **Configure logging once** at application startup, not in modules
2. **Use specific exceptions** - catch what you expect, let others propagate
3. **Add context to exceptions** for easier debugging
4. **Apply retry decorator** to API calls but not to validation/local logic
5. **Use run_coroutine_with_fallback** instead of manual event loop handling
6. **Log with context** using LoggerFactory for consistency
"""
