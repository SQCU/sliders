##
#   ?python lora_inspector.py "path/to/your/good_lora.safetensors" "path/to/your/molKM_IL_imageslider_b.safetensors" 
#
#

import torch
import os
import threading
from flask import Flask, request, jsonify, Response
from safetensors.torch import load_file
from werkzeug.utils import secure_filename
from typing import Dict, Any

# --- Backend Setup ---
import psutil # Add this import
import time


def diagnose_file_lock(filepath: str):
    """
    Checks all running processes to see which one has a lock on the specified file.
    """
    print(f"--- Running Lock Diagnosis for: {filepath} ---")
    found_locking_process = False
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            # open_files() might fail for some system processes, so we use a try-except block
            if proc.info['open_files'] is not None:
                for file in proc.info['open_files']:
                    if file.path == filepath:
                        print(f"  [!!!] Found locking process:")
                        print(f"        PID:  {proc.info['pid']}")
                        print(f"        Name: {proc.info['name']}")
                        found_locking_process = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass # Ignore processes that we can't inspect
    if not found_locking_process:
        print("  [---] No specific locking process found via psutil. The lock may be very brief (e.g., antivirus) or kernel-level.")
    print("--- End of Lock Diagnosis ---")

def safe_makedirs(path: str, mode: int = 0o755) -> None:
    """
    Creates a directory and sets its permissions.
    This is a cross-platform-safe way to ensure a directory is writable.
    """
    # Create the directory if it doesn't exist.
    # The default behavior is usually sufficient, but we'll be explicit.
    os.makedirs(path, exist_ok=True)
    
    try:
        # On POSIX systems (Linux, macOS), explicitly set permissions.
        # This ensures the owner can rwx, and others can rx.
        # On Windows, this function has limited effect but won't harm anything.
        os.chmod(path, mode)
    except OSError as e:
        # This can happen on some file systems or if permissions are restricted.
        # We can print a warning but continue, as the folder likely exists with usable permissions.
        print(f"Warning: Could not set permissions on {path}: {e}")

# --- THE FIX IS HERE ---
# Get the absolute path to the directory where this script is located.
# This ensures that our paths are always correct, regardless of where
# the script is called from.
_basedir = os.path.dirname(os.path.abspath(__file__))

# Define UPLOAD_FOLDER as an absolute path relative to the script's location.
UPLOAD_FOLDER = os.path.join(_basedir, 'uploads')
# --- END OF FIX ---

# 1. Initialize Flask App and configure upload folder
app = Flask(__name__)

# Call our safe directory creation function with the now-absolute path
safe_makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 # 16 GB upload limit

# 2. Network program housekeeping!

import socket

def find_available_port(start_port: int, host: str = '127.0.0.1') -> int:
    """
    Finds an available TCP port by asking the OS to assign one.
    This is a more reliable method than sequentially probing ports.
    """
    # Create a socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to the host and port 0. Port 0 is a special value that
        # tells the OS to pick an available ephemeral port.
        s.bind((host, 0))
        
        # Ask the socket what port the OS actually assigned to it
        port = s.getsockname()[1]
        
        # The 'with' statement will now close the socket, releasing the port.
        # While a tiny race condition still exists, this "ask, don't guess"
        # pattern is far more reliable and avoids looping issues.
        return port

import stat # Import the stat module for permission constants

# --- Core Comparison Logic (adapted for JSON output) ---

from safetensors import safe_open # Make sure this is imported

class LoRAWebComparator:
    def __init__(self, good_lora_path: str, bad_lora_path: str):
        self.good_lora_path = good_lora_path
        self.bad_lora_path = bad_lora_path
        
        # --- THE FIX IS HERE ---
        # Instead of using the high-level load_file, we use a context manager
        # to guarantee the file handle is closed immediately after loading.
        self.good_lora_state = self._load_state_dict_safely(good_lora_path)
        self.bad_lora_state = self._load_state_dict_safely(bad_lora_path)
        # --- END OF FIX ---

        self.good_keys = set(self.good_lora_state.keys())
        self.bad_keys = set(self.bad_lora_state.keys())

    def _load_state_dict_safely(self, path: str) -> Dict[str, torch.Tensor]:
        """Loads a state dict from a safetensors file, ensuring the file is closed."""
        state_dict = {}
        # The 'with' statement guarantees the file handle 'f' is closed upon exiting this block.
        with safe_open(path, framework="pt", device="cpu") as f:
            # Iterate through all the keys and load each tensor into our new dictionary.
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    # ... all other methods (_get_tensor_norm, compare, etc.) remain exactly the same ...
    def _get_tensor_norm(self, tensor: torch.Tensor) -> float:
        """A simple, dedicated helper to calculate the L2 norm."""
        return torch.linalg.norm(tensor.to(torch.float32)).item()

    def _calculate_global_norm_range(self):
        """
        This method now ONLY calculates norms and sets the min/max attributes.
        It no longer has a circular dependency.
        """
        all_norms = [self._get_tensor_norm(t) for state in [self.good_lora_state, self.bad_lora_state] for t in state.values()]
        
        self.min_norm = min(all_norms) if all_norms else 0.0
        self.max_norm = max(all_norms) if all_norms else 1.0
        if self.max_norm == self.min_norm: self.max_norm += 1e-6

    def _get_norm_hex_color(self, norm: float) -> str:
        """This method can now safely access self.max_norm."""
        if not hasattr(self, 'max_norm'):
            return '#808080' 
            
        ratio = (norm - self.min_norm) / (self.max_norm - self.min_norm) if self.max_norm > self.min_norm else 0
        r = int(255 * ratio)
        b = int(255 * (1 - ratio))
        g = 0
        return f'#{r:02x}{g:02x}{b:02x}'

    def _get_tensor_info(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Gathers all properties of a single tensor."""
        norm = self._get_tensor_norm(tensor)
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "norm": norm,
            "color": self._get_norm_hex_color(norm)
        }

    def compare(self) -> Dict[str, Any]:
        """
        The main public method. The order of operations is now correct.
        1. Calculate global ranges.
        2. Use those ranges to gather detailed info for each tensor.
        """
        self._calculate_global_norm_range()
        
        common_keys = self.good_keys.intersection(self.bad_keys)
        only_in_good = sorted(list(self.good_keys - self.bad_keys))
        only_in_bad = sorted(list(self.bad_keys - self.good_keys))

        all_sorted_keys = sorted(list(self.good_keys.union(self.bad_keys)))
        
        side_by_side_data = []
        for key in all_sorted_keys:
            good_info = self._get_tensor_info(self.good_lora_state[key]) if key in self.good_lora_state else None
            bad_info = self._get_tensor_info(self.bad_lora_state[key]) if key in self.bad_lora_state else None
            side_by_side_data.append({"key": key, "good": good_info, "bad": bad_info})

        return {
            "summary": {
                "good_lora_name": os.path.basename(self.good_lora_path),
                "bad_lora_name": os.path.basename(self.bad_lora_path),
                "common_keys": len(common_keys),
                "only_in_good": len(only_in_good),
                "only_in_bad": len(only_in_bad),
            },
            "key_diffs": { "only_in_good": only_in_good, "only_in_bad": only_in_bad },
            "comparison_data": side_by_side_data
        }


      
class TemporaryFileUpload:
    """
    A context manager to handle the lifecycle of a temporary file uploaded via Flask.
    It saves the file on __enter__ and guarantees its deletion on __exit__.
    """
    def __init__(self, flask_file_obj, upload_folder: str):
        self.flask_file = flask_file_obj
        self.upload_folder = upload_folder
        self.temp_path = None

    def __enter__(self) -> str:
        """Saves the file to a temporary path and returns the path."""
        # Create a unique, absolute path for the temporary file
        tid = threading.get_ident()
        filename = secure_filename(self.flask_file.filename)
        self.temp_path = os.path.join(self.upload_folder, f"{tid}_{filename}")
        
        # Save the file
        self.flask_file.save(self.temp_path)
        print(f"[Context Manager] Created temporary file: {self.temp_path}")
        
        # Return the path for use inside the 'with' block
        return self.temp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deletes the temporary file, ensuring cleanup."""
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                # Add a small delay to let any lingering file handles close,
                # which is a pragmatic fix for Windows race conditions.
                time.sleep(0.1)
                os.remove(self.temp_path)
                print(f"[Context Manager] Cleaned up temporary file: {self.temp_path}")
            except Exception as e:
                print(f"[Context Manager] ERROR: Failed to clean up {self.temp_path}. Error: {e}")
        # Returning False (or None) will re-raise any exception that occurred
        # inside the 'with' block, which is the standard, correct behavior.

    

# --- Flask API Endpoints ---

      
@app.route('/compare', methods=['POST'])
def compare_loras():
    if 'good_lora' not in request.files or 'bad_lora' not in request.files:
        return jsonify({"error": "Missing one or both LoRA files"}), 400

    good_lora_upload = request.files['good_lora']
    bad_lora_upload = request.files['bad_lora']

    if good_lora_upload.filename == '' or bad_lora_upload.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Use our context managers to handle the entire lifecycle of the temp files.
        # This is clean, safe, and guarantees cleanup.
        with TemporaryFileUpload(good_lora_upload, app.config['UPLOAD_FOLDER']) as temp_good_path:
            with TemporaryFileUpload(bad_lora_upload, app.config['UPLOAD_FOLDER']) as temp_bad_path:
                
                # The core logic is now cleanly nested inside.
                # The files exist for the duration of this block.
                comparator = LoRAWebComparator(temp_good_path, temp_bad_path)
                results = comparator.compare()
                
                # When the 'with' blocks exit, __exit__ will be called automatically,
                # deleting the files.
                return jsonify(results)

    except Exception as e:
        # If any error occurs (upload, save, compare), it's caught here
        # and the __exit__ methods of any entered contexts are still called.
        print(f"An error occurred during the request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/')
def index():
    # We serve the entire HTML/CSS/JS as a single response.
    return Response(HTML_CONTENT, mimetype='text/html')

# --- Frontend (HTML, CSS, JS) ---

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoRA Inspector</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }
        .container { max-width: 95%; margin: auto; }
        h1, h2 { color: #569cd6; border-bottom: 1px solid #444; padding-bottom: 10px; }
        .upload-form { background-color: #252526; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .file-input { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="file"] { color: #d4d4d4; }
        button { background-color: #0e639c; color: white; padding: 10px 15px; border: none; border-radius: 3px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #1177bb; }
        #status { margin-top: 15px; font-style: italic; color: #ce9178; }
        .results-container { margin-top: 20px; }
        .summary-table, .comparison-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        .summary-table td, .comparison-table th, .comparison-table td { border: 1px solid #444; padding: 8px 12px; text-align: left; }
        .comparison-table th { background-color: #2a2d2e; }
        .key-name { word-break: break-all; font-family: "Courier New", Courier, monospace; }
        .missing { background-color: #412424; }
        .good { background-color: #244124; }
        .mismatch { background-color: #413a24; }
        .norm-value { font-weight: bold; }
        .key-diffs { display: flex; gap: 20px; }
        .key-list { flex: 1; background-color: #252526; padding: 15px; border-radius: 5px; }
        .key-list h3 { margin-top: 0; }
        .key-list ul { padding-left: 20px; max-height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LoRA Inspector</h1>
        <div class="upload-form">
            <form id="upload-form">
                <div class="file-input">
                    <label for="good_lora">Reference LoRA (Known-Good):</label>
                    <input type="file" id="good_lora" name="good_lora" accept=".safetensors" required>
                </div>
                <div class="file-input">
                    <label for="bad_lora">Your LoRA (To Inspect):</label>
                    <input type="file" id="bad_lora" name="bad_lora" accept=".safetensors" required>
                </div>
                <button type="submit">Compare Models</button>
                <div id="status"></div>
            </form>
        </div>
        <div id="results-container"></div>
    </div>

    <script>
        document.getElementById('upload-form').addEventListener('submit', async function(event) {
            event.preventDefault();
            const form = event.target;
            const formData = new FormData(form);
            const statusDiv = document.getElementById('status');
            const resultsContainer = document.getElementById('results-container');

            statusDiv.textContent = 'Uploading and comparing files... This may take a moment for large files.';
            resultsContainer.innerHTML = '';

            try {
                const response = await fetch('/compare', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'An unknown error occurred.');
                }
                
                statusDiv.textContent = 'Comparison complete!';
                renderResults(data);

            } catch (error) {
                statusDiv.textContent = `Error: ${error.message}`;
                console.error('Comparison failed:', error);
            }
        });

        function renderResults(data) {
            const container = document.getElementById('results-container');
            container.innerHTML = `
                <h2>Comparison Summary</h2>
                <table class="summary-table">
                    <tr><td>Reference Model</td><td>${data.summary.good_lora_name}</td></tr>
                    <tr><td>Your Model</td><td>${data.summary.bad_lora_name}</td></tr>
                    <tr><td>Common Keys</td><td>${data.summary.common_keys}</td></tr>
                    <tr><td>Keys Only in Reference</td><td>${data.summary.only_in_good}</td></tr>
                    <tr><td>Keys Only in Your Model</td><td>${data.summary.only_in_bad}</td></tr>
                </table>
                
                <h2>Key Differences</h2>
                <div class="key-diffs">
                    <div class="key-list">
                        <h3>Only in Reference Model</h3>
                        <ul>
                            ${data.key_diffs.only_in_good.map(k => `<li><span class="key-name">${k}</span></li>`).join('') || '<li>None</li>'}
                        </ul>
                    </div>
                    <div class="key-list">
                        <h3>Only in Your Model</h3>
                        <ul>
                            ${data.key_diffs.only_in_bad.map(k => `<li><span class="key-name">${k}</span></li>`).join('') || '<li>None</li>'}
                        </ul>
                    </div>
                </div>

                <h2>Side-by-Side Comparison</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Parameter Key</th>
                            <th>Reference Model Details</th>
                            <th>Your Model Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.comparison_data.map(renderRow).join('')}
                    </tbody>
                </table>
            `;
        }

        function renderRow(rowData) {
            const { key, good, bad } = rowData;
            let goodHtml = '<div class="missing">MISSING</div>';
            let badHtml = '<div class="missing">MISSING</div>';
            let rowClass = '';

            if (good) {
                goodHtml = `
                    <div>Shape: ${JSON.stringify(good.shape)}</div>
                    <div>DType: ${good.dtype}</div>
                    <div>L2 Norm: <span class="norm-value" style="color:${good.color}">${good.norm.toFixed(4)}</span></div>
                `;
            }
            if (bad) {
                badHtml = `
                    <div>Shape: ${JSON.stringify(bad.shape)}</div>
                    <div>DType: ${bad.dtype}</div>
                    <div>L2 Norm: <span class="norm-value" style="color:${bad.color}">${bad.norm.toFixed(4)}</span></div>
                `;
            }
            
            if (good && !bad) rowClass = 'missing';
            if (!good && bad) rowClass = 'good'; // 'good' here means it exists in your model but not reference
            if (good && bad && (JSON.stringify(good.shape) !== JSON.stringify(bad.shape))) {
                rowClass = 'mismatch';
            }


            return `
                <tr class="${rowClass}">
                    <td class="key-name">${key}</td>
                    <td>${goodHtml}</td>
                    <td>${badHtml}</td>
                </tr>
            `;
        }
    </script>
</body>
</html>
"""

# --- Main Execution ---
if __name__ == '__main__':
    # Using '0.0.0.0' makes the server accessible from other devices on your network.
    # The check will be performed against this host.
    HOST = '0.0.0.0' 
    
    # We find an available port using the robust method
    available_port = find_available_port(start_port=HOST)
    
    # The address to display in the browser should be the loopback address
    display_host = '127.0.0.1' 
    
    print("Starting LoRA Inspector web server...")
    print(f"Server is running. Navigate to http://{display_host}:{available_port} in your browser.")
    
    # Use the discovered available port
    app.run(debug=True, use_reloader=False, host=HOST, port=available_port)