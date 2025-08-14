```instructions
<SYSTEM>
You are an AI programming assistant that is specialized in applying code changes to an existing document.
Follow Microsoft content policies.
Avoid content that violates copyrights.
If you are asked to generate content that is harmful, hateful, racist, sexist, lewd, violent, or completely irrelevant to software engineering, only respond with "Sorry, I can't assist with that."
Keep your answers short and impersonal.
The user has a code block that represents a suggestion for a code change and a instructions file opened in a code editor.
Rewrite the existing document to fully incorporate the code changes in the provided code block.
For the response, always follow these instructions:
1. Analyse the code block and the existing document to decide if the code block should replace existing code or should be inserted.
2. If necessary, break up the code block in multiple parts and insert each part at the appropriate location.
3. Preserve whitespace and newlines right after the parts of the file that you modify.
4. The final result must be syntactically valid, properly formatted, and correctly indented. It should not contain any ...existing code... comments.
5. Finally, provide the fully rewritten file. You must output the complete file.
</SYSTEM>
```

# 🤖 AI Agent Context

## Project Overview

## 🎯 Core Purpose

## 🏗️ Architecture & Tech Stack

### Core Framework

### Dependencies & Package Management



### Design Principles

## 📁 Project Structure

```

```

## 🔧 Environment Configuration

### Required Environment Variables

```bash
# Set all required environment variables in your shell or .env file before running any scripts.
```

### Configuration Loading

## 🚀 Core Modules

## 🧪 Testing Strategy

### Test Infrastructure



## 🤖 AI Agent Instructions - STRICT COMPLIANCE REQUIRED

**CRITICAL**: All AI agents working on this project MUST follow these instructions precisely. Deviation from these guidelines is not permitted.


### 🧪 TESTING REQUIREMENTS - NO EXCEPTIONS

1. **ALL new features MUST have tests**:

   - Unit tests for all functions
   - Integration tests for API interactions
   - Async test patterns using pytest-asyncio
   - Mock external dependencies appropriately

2. **Test Coverage Standards**:

   - Minimum 90% code coverage
   - 100% coverage for public APIs
   - Edge cases and error conditions MUST be tested

3. **Test Quality Requirements**:
   - Clear test names describing what is being tested
   - Arrange-Act-Assert pattern
   - No test interdependencies
   - Fast execution (no real API calls in unit tests)


### 🔧 CODE QUALITY - STRICT ENFORCEMENT

1. **Linting and Formatting**:

   - MUST run `ruff format .` before committing
   - MUST run `ruff check .` and fix all issues
   - MUST run `mypy line_api/` and resolve all type errors
   - NO disabled linting rules without justification
   - **ACHIEVED**: 100% mypy strict mode compliance across all modules

2. **Code Style Requirements**:

   - NO wildcard imports (`from module import *`)
   - NO unused imports or variables
   - Consistent naming conventions throughout
   - Use modern type annotations (`dict`/`list` not `Dict`/`List`)

3. **Performance Requirements**:
   - Use async patterns for ALL I/O operations
   - Implement proper connection pooling
   - Cache responses when appropriate
   - Monitor memory usage for large operations

### 🛡️ SECURITY REQUIREMENTS - NON-NEGOTIABLE

1. **Credential Management**:
   - NO hardcoded secrets or tokens
   - ALL credentials MUST use environment variables
   - Pydantic SecretStr for sensitive data
   - Secure defaults for all configuration

### 🔄 DEVELOPMENT WORKFLOW - MANDATORY STEPS

#### Before Starting ANY Task:

1. Create feature branch: `git checkout -b feature/description`
2. Read ALL relevant existing code
3. Check current tests: `uv pip run python -m pytest tests/ -v`
4. Understand the current implementation completely

#### During Development:

1. Write tests FIRST (TDD approach preferred)
2. Implement with full type hints
3. Add comprehensive docstrings
4. Run tests frequently: `uv pip run python -m pytest tests/test_specific.py -v`

#### Before Committing:

1. Run ALL tests: `uv pip run python -m pytest tests/ -v`
2. Run type checking: `uv pip run mypy line_api/`
3. Run linting: `uv pip run ruff check . && uv pip run ruff format .`
4. Verify examples still work
5. Update documentation if needed

### ❌ PROHIBITED ACTIONS

1. **NEVER** use bare `except:` clauses
2. **NEVER** ignore type checker warnings without justification
3. **NEVER** hardcode credentials or secrets
4. **NEVER** commit debug print statements
5. **NEVER** break existing public APIs without deprecation
6. **NEVER** add dependencies without updating pyproject.toml
7. **NEVER** commit code that doesn't pass all tests
8. **NEVER** use synchronous I/O for external API calls

### 🏆 QUALITY GATES - ALL MUST PASS

Before any code is considered complete:

- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Type checking passes: `mypy line_api/`
- [ ] Linting passes: `ruff check .`
- [ ] Code is formatted: `ruff format .`
- [ ] Documentation is updated
- [ ] Examples work correctly
- [ ] Performance is acceptable
- [ ] Security review completed

### 🚨 VIOLATION CONSEQUENCES

Failure to follow these guidelines will result in:

1. Immediate rejection of changes
2. Required rework with full compliance
3. Additional review requirements for future changes

**These guidelines are not suggestions - they are requirements for maintaining the quality and reliability of this production-grade LINE API integration library.**

### Development Guidelines

#### Adding New Features

1. **Plan the API**: Design the public interface first
2. **Write Tests**: Start with test cases for the new feature
3. **Implement**: Create the implementation with full type hints
4. **Document**: Add comprehensive docstrings and examples
5. **Integration**: Update the main `LineAPI` class if needed
6. **Validate**: Run all tests and type checking

#### Code Organization Rules

- **Clean Imports**: All imports at the top of files
- **Debug Scripts**: All debug/investigation scripts MUST go in `/debug` folder (gitignored)
- **Tests**: All pytest tests MUST go in `/tests` folder
- **Examples**: Real-world examples in `/examples` folder
- **Documentation**: API docs and guides in `/docs` folder

#### Error Handling Patterns

```python
from line_api.core.exceptions import LineAPIError, LineRateLimitError

# Proper exception handling with retry
async def send_with_retry(client: LineMessagingClient, message: Any) -> bool:
    """Send message with exponential backoff retry."""
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            return await client.push_message("USER_ID", [message])
        except LineRateLimitError as e:
            if attempt == max_retries:
                raise

            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
            continue
        except LineAPIError as e:
            # Log error with structured data
            logger.error(
                "Message send failed",
                extra={"user_id": "USER_ID", "error": str(e)}
            )
            raise
```

### Production Considerations

- **Rate Limiting**: Implement proper rate limiting for all API calls
- **Error Recovery**: Retry mechanisms with exponential backoff
- **Logging**: Structured logging for debugging and monitoring
- **Security**: Secure credential management and validation
- **Performance**: Async operations and connection pooling
- **Monitoring**: Health checks and metrics collection

### Key Files for AI Understanding

- **README.md**: User-facing documentation and usage examples
- **pyproject.toml**: Dependencies and project configuration
- **Module `__init__.py` files**: Public API exports and module structure
- **Test files**: Examples of proper usage and expected behavior
- **Integration guides**: Patterns for using shared tools in services

