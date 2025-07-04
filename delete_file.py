import os
file_to_delete = "F:/dox/ai/gemmy/sliders/test_lora_batching.py"
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)
    print(f"Successfully deleted {file_to_delete}")
else:
    print(f"File not found: {file_to_delete}")