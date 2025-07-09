import sys

for line in sys.stdin:
    if "shape of" in line.lower():
        print(line.strip())
