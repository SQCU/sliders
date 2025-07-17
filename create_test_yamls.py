
import yaml
from pathlib import Path
import os

def create_yaml_file(filepath: Path, content: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(content, f)
    print(f"Created: {filepath}")

if __name__ == "__main__":
    yamlzoo_dir = Path("F:/dox/ai/gemmy/sliders/run_artifacts/yamlzoo")

    # Case 1: Iguana is awake
    create_yaml_file(yamlzoo_dir / "iguana_awake.yaml", {
        "animal": "iguana",
        "status": "awake",
        "mood": "happy",
        "message": "Enjoying the sun!"
    })

    # Case 2: Iguana is sleeping (triggers early exit)
    create_yaml_file(yamlzoo_dir / "iguana_sleeping.yaml", {
        "animal": "iguana",
        "status": "sleeping",
        "mood": "peaceful",
        "message": "Shhh, don't wake me."
    })

    # Case 3: Another awake iguana (will only be seen if sleeping one is not encountered first)
    create_yaml_file(yamlzoo_dir / "another_awake.yaml", {
        "animal": "iguana",
        "status": "awake",
        "mood": "playful",
        "message": "Ready for some fun!"
    })
