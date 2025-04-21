#!/bin/bash

# Run script for the AI-Augmented Full-Stack Algorithmic Trading Pipeline tests
# This script runs both unit tests and integration tests for the pipeline

# Set up environment
echo "Setting up test environment..."
mkdir -p /home/ubuntu/trading_pipeline/logs

# Run unit tests
echo "Running unit tests..."
python3 /home/ubuntu/trading_pipeline/tests/unit_tests.py

# Store unit test result
UNIT_TEST_RESULT=$?
echo "Unit tests completed with exit code: $UNIT_TEST_RESULT"

# Run integration tests
echo "Running integration tests..."
python3 /home/ubuntu/trading_pipeline/tests/integration_tests.py

# Store integration test result
INTEGRATION_TEST_RESULT=$?
echo "Integration tests completed with exit code: $INTEGRATION_TEST_RESULT"

# Determine overall test result
if [ $UNIT_TEST_RESULT -eq 0 ] && [ $INTEGRATION_TEST_RESULT -eq 0 ]; then
    echo "All tests passed successfully!"
    exit 0
elif [ $UNIT_TEST_RESULT -eq 0 ] && [ $INTEGRATION_TEST_RESULT -eq 1 ]; then
    echo "Unit tests passed, but some integration tests failed or were skipped."
    exit 1
elif [ $UNIT_TEST_RESULT -eq 1 ] && [ $INTEGRATION_TEST_RESULT -eq 0 ]; then
    echo "Integration tests passed, but some unit tests failed."
    exit 1
else
    echo "Both unit tests and integration tests had failures."
    exit 2
fi
