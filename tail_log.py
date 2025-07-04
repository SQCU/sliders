import argparse
import os

def tail(filepath, n_lines):
    """
    Reads the last n_lines from a file.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    with open(filepath, 'r') as f:
        # Go to the end of the file
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

        lines = []
        buffer_size = 4096  # Read in chunks
        read_bytes = 0
        
        # Read backwards in chunks until we have enough lines or reach the beginning
        while len(lines) <= n_lines and read_bytes < file_size:
            read_bytes = min(file_size, read_bytes + buffer_size)
            f.seek(file_size - read_bytes)
            chunk = f.read(buffer_size)
            
            # Split by lines and add to the beginning of our list
            # This handles cases where a line ending is split across chunks
            new_lines = chunk.splitlines(keepends=True)
            lines = new_lines + lines
            
            # If we have more than n_lines, trim from the beginning
            if len(lines) > n_lines:
                lines = lines[-n_lines:]
        
        # Ensure we only return n_lines
        if len(lines) > n_lines:
            lines = lines[-n_lines:]

        for line in lines:
            print(line.strip()) # .strip() to remove extra newlines if keepends=True was used

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display the last N lines of a file, similar to 'tail -n'.")
    parser.add_argument("filepath", type=str, help="The path to the log file.")
    parser.add_argument("-n", "--lines", type=int, default=10, help="Number of lines to display from the end of the file.")
    
    args = parser.parse_args()
    tail(args.filepath, args.lines)
