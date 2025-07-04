import os

files_to_remove = [
    "diff_high_orig_refactored.pt",
    "diff_high_orig_super.pt",
    "diff_low_orig_refactored.pt",
    "diff_low_orig_super.pt",
    "loss_high_orig_per_element.pt",
    "loss_high_refactored_per_element.pt",
    "loss_high_super_per_element.pt",
    "loss_low_orig_per_element.pt",
    "loss_low_refactored_per_element.pt",
    "loss_low_super_per_element.pt",
    "orig_tensors.pt",
    "refactored_tensors.pt",
    "super_tensors.pt",
    "refactored_training_loop_test.log",
    "diff_debugger_output.log",
    "create_dir.py"
]

for file_name in files_to_remove:
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed: {file_name}")
    else:
        print(f"File not found (skipping): {file_name}")