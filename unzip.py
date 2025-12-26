import os
import gzip
import shutil

folder = "test_set/instances"

for filename in os.listdir(folder):
    if filename.endswith(".gz"):
        gz_path = os.path.join(folder, filename)
        out_path = os.path.join(folder, filename[:-3])  # remove .gz

        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"Decompressed: {gz_path} -> {out_path}")

print("Done.")
