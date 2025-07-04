import sys
import os

# Add the directory containing refactored_training_loop.py to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from refactored_training_loop import test_refactored_training_loop
    print("Running test_refactored_training_loop...")
    test_refactored_training_loop()
    print("Test completed successfully.")
except Exception as e:
    print(f"An error occurred during testing: {e}")
    import traceback
    traceback.print_exc()
