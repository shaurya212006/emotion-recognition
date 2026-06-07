import os, shutil

SRC_TRAIN = "data/train"
SRC_TEST  = "data/test"
OUT_DIR   = "data/processed"

EMOTION_MAP = {"happy": "happy", "sad": "sad"}

def prepare():
    for split, src in [("train", SRC_TRAIN), ("val", SRC_TEST), ("test", SRC_TEST)]:
        for emotion, cls in EMOTION_MAP.items():
            src_path = os.path.join(src, emotion)
            dst_path = os.path.join(OUT_DIR, split, cls)
            os.makedirs(dst_path, exist_ok=True)
            if not os.path.exists(src_path):
                print(f"WARNING: {src_path} not found!")
                continue
            files = [f for f in os.listdir(src_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if split == "val":
                files = files[:len(files)//2]
            elif split == "test":
                files = files[len(files)//2:]
            for i, fname in enumerate(files):
                ext = os.path.splitext(fname)[1]
                shutil.copy(
                    os.path.join(src_path, fname),
                    os.path.join(dst_path, f"{cls}_{i:05d}{ext}")
                )
            print(f"  {split}/{cls}: {len(files)} images copied")

    print("\nDone! Dataset prepared in data/processed/")

if __name__ == "__main__":
    prepare()