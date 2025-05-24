# Testing Documentation

## Prerequisites
Before running tests, make sure you have installed all development dependencies:

```bash
pip install -r requirements.txt
pip install pytest pytest-mock
```

## Running Tests

### Run all tests
To run all tests in the test suite:
```bash
pytest
```

### Run specific test files
To run tests from a specific file:
```bash
pytest test_inputData.py
pytest test_nodes.py
pytest test_graph.py
```

### Run tests with verbosity
For more detailed test output:
```bash
pytest -v
```

### Run tests with print output
To see print statements during test execution:
```bash
pytest -s
```

### Run tests matching specific names
To run tests that match a specific pattern:
```bash
pytest -k "test_persona"  # Runs all tests with "test_persona" in the name
```

### Run Docker integration tests
To run the full Docker stack integration tests:
```bash
# Requires Docker and docker-compose to be installed
pytest tests/test_docker_integration.py

# To skip Docker tests (e.g., in CI without Docker)
SKIP_DOCKER_TESTS=1 pytest
```

## Test Coverage
To generate a test coverage report:

```bash
pip install pytest-cov
pytest --cov=../ --cov-report=term-missing
```

## Test Organization
- `conftest.py`: Contains shared pytest fixtures and configurations
- `test_inputData.py`: Tests for profile and question sampling functionality
- `test_nodes.py`: Tests for cultural expert nodes and routing logic
- `test_graph.py`: Tests for the cultural graph workflow
- `test_api_endpoints.py`: Tests for REST API endpoints (mocked)
- `test_docker_integration.py`: Integration tests for full Docker stack

## Writing New Tests
When adding new tests:
1. Follow the existing test file structure
2. Use descriptive test names that indicate what's being tested
3. Use pytest fixtures for shared setup
4. Include tests for both success and error cases
5. Add appropriate assertions to verify expected behavior

## Debugging Tests
For debugging test failures:
```bash
pytest --pdb  # Drops into debugger on failures
pytest -x     # Stop after first failure
pytest --maxfail=2  # Stop after two failures
```

## Continuous Integration
Tests are run automatically on:
- Every push to main branch
- Every pull request
- Daily scheduled runs

Make sure all tests pass locally before pushing changes.