# Claude Agents for Python Development

This directory contains specialized AI agents designed to work with Python projects. Each agent has specific expertise and responsibilities to ensure consistent, high-quality development across any Python codebase.

## Available Agents

### Core Development Agents

#### `python-architect.md`
- **Purpose**: Provides architectural guidance and code quality standards
- **Trigger**: New features, refactoring, architectural decisions
- **Expertise**: Async patterns, type safety, data modeling, performance optimization

#### `documentation-specialist.md`
- **Purpose**: Maintains comprehensive documentation standards
- **Trigger**: API documentation, docstrings, usage examples
- **Expertise**: Technical writing, Python docstrings, API documentation

#### `dependency-manager.md`
- **Purpose**: Manages dependencies and Python execution environment
- **Trigger**: Adding/removing packages, environment setup
- **Expertise**: uv package manager, environment isolation, dependency resolution

## Agent Coordination

### Sequential Invocation Pattern
1. **Primary Agent**: Handles the main task (e.g., `@python-architect` for new features)
2. **Secondary Agent**: Provides supporting expertise (e.g., `@documentation-specialist` for API docs)

### Decision Matrix
| Task Type | Primary Agent | Secondary Agent | Quality Gates |
|-----------|---------------|-----------------|---------------|
| New Feature Design | `@python-architect` | `@documentation-specialist` | All |
| Bug Fix | General Claude | `@python-architect` | Tests + Lint |
| Refactoring | `@python-architect` | `@documentation-specialist` | All |
| Documentation | `@documentation-specialist` | `@python-architect` | Lint only |
| Configuration | `@python-architect` | `@dependency-manager` | Tests |
| API Documentation | `@documentation-specialist` | `@python-architect` | All |
| Dependencies | `@dependency-manager` | `@python-architect` | Tests |

## Quality Gates

All agent workflows must enforce these quality gates:

```bash
# Mandatory before any commit
uv run ruff check . && uv run ruff format . && uv run mypy <project_name>/
uv run python -m pytest tests/ -v
```

### Architecture Reviews Must Consider:
- Async pattern compliance
- Type safety with mypy validation
- Data model correctness and validation
- Test coverage maintenance (90%+)
- Performance implications
- Security considerations

## Usage Guidelines

### When to Use Specialized Agents
- **Complex Tasks**: Multi-step operations requiring specific expertise
- **Quality Assurance**: Ensuring consistency across the codebase
- **Domain Expertise**: Tasks requiring specialized knowledge
- **Standard Enforcement**: Maintaining project conventions

### Agent Invocation
Agents are invoked by referencing them in your prompts:
- `@python-architect` - For architectural guidance and design
- `@documentation-specialist` - For documentation standards
- `@dependency-manager` - For package management

### Integration with CLAUDE.md
These agents work in conjunction with the main CLAUDE.md instructions, providing specialized expertise while following the core project guidelines.

## Universal Application

These agents are designed for modern Python projects (3.13+) featuring:

- Async-first architecture patterns
- Type-safe data models for validation
- Comprehensive testing and quality assurance
- Modern Python tooling and best practices

Each agent is designed to maintain the high quality standards expected in production-grade Python applications across any domain.
