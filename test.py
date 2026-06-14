import os
import time
import traceback

import kagglehub
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout

os.environ["KAGGLEHUB_CACHE"] = "/data/yuhaowang/kagglehub_cache"

OUT_DIR = "/data/yuhaowang/data/pathology/UBC-OCEAN"

for i in range(200):
    try:
        print(f"[try {i + 1}] start / resume downloading UBC-OCEAN ...")

        path = kagglehub.competition_download(
            "UBC-OCEAN",
            output_dir=OUT_DIR,
        )

        print("Download finished.")
        print("Path:", path)
        break

    except (ChunkedEncodingError, ConnectionError, ReadTimeout) as e:
        print(f"[try {i + 1}] network interrupted, will resume later.")
        print(repr(e))
        time.sleep(min(300, 30 * (i + 1)))

    except Exception:
        print("[fatal error]")
        traceback.print_exc()
        raise