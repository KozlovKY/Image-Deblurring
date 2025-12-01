import subprocess
from pathlib import Path


def ensure_data(stage: str | None = None) -> None:
    """
    Run pipeline stages (`dvc repro <stage>`)
    """

    project_root = Path(__file__).resolve().parents[2]
    cmd = ["dvc", "repro"]
    if stage:
        cmd.append(stage)

    subprocess.run(cmd, cwd=project_root, check=False)
