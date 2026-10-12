"""Download the pinned OpenCV pedestrian clip used for offline verification."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/opencv/opencv/4.12.0/samples/data/vtest.avi"
SHA256 = "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf"


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "vtest.avi"
    if output.exists():
        with output.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != SHA256:
            raise ValueError("Existing demo video has an unexpected checksum; it was not replaced.")
    else:
        with urlopen(URL, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Downloaded video has an unexpected checksum.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    print(f"Verified demo video: {output}")


if __name__ == "__main__":
    main()
