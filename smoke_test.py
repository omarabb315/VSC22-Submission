"""
Minimal smoke test to verify all pipeline imports and code paths work
without requiring real checkpoints or VCDB data.
"""

import os
import sys
import subprocess

# Must run before numpy/torch/sklearn/faiss (avoids OpenMP pthread init crash on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def test_imports():
    """Test all critical imports from both tracks."""
    print("=== Test 1: Critical imports ===")

    # Core
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    import sklearn
    import timm
    import PIL
    import faiss
    print(f"  torch={torch.__version__}, numpy={np.__version__}, faiss OK")

    # Descriptor track vsc
    sys.path.insert(0, os.path.join(REPO_ROOT, "VSC22-Descriptor-Track-1st", "infer"))
    from vsc.storage import store_features, load_features
    from vsc.index import VideoFeature
    from vsc.baseline.score_normalization import query_score_normalize, ref_score_normalize
    from vsc.metrics import Dataset
    from src.dataset import VideoDataset, D_vsc
    from src.image_preprocess import image_process
    from src.transform import sscd_transform, eff_transform, vit_transform
    from src.extractor import extract_vsc_feat
    from src.utils import calclualte_low_var_dim
    print("  Descriptor track imports: OK")

    # Matching track -- use importlib to load from separate directory
    import importlib.util
    match_infer = os.path.join(REPO_ROOT, "VSC22-Matching-Track-1st", "infer")
    for mod_name, rel_path in [
        ("match_dataset", "src/dataset.py"),
        ("match_utils", "src/utils.py"),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, os.path.join(match_infer, rel_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    print("  Matching track imports: OK")

    return True


def test_device_detection():
    """Test device detection logic."""
    print("\n=== Test 2: Device detection ===")
    import torch

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  Selected device: {DEVICE}")

    # Verify JIT fuser guards don't crash
    try:
        torch.jit.fuser('off')
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        print("  JIT fuser guards: applied (older PyTorch)")
    except (AttributeError, RuntimeError):
        print("  JIT fuser guards: skipped (PyTorch 2.x, expected)")

    return True


def test_faiss():
    """Test FAISS works."""
    print("\n=== Test 3: FAISS ===")
    import faiss
    import numpy as np

    d = 512
    index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    data = np.random.randn(10, d).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    index.add(data)
    D, I = index.search(data[:2], 5)
    print(f"  FAISS search: OK (top-5 for 2 queries against 10 refs)")
    print(f"  GPU count: {faiss.get_num_gpus()}")
    return True


def test_storage():
    """Test VSC storage read/write."""
    print("\n=== Test 4: VSC storage ===")
    import numpy as np
    sys.path.insert(0, os.path.join(REPO_ROOT, "VSC22-Descriptor-Track-1st", "infer"))
    from vsc.storage import store_features, load_features
    from vsc.index import VideoFeature

    features = [
        VideoFeature(video_id="R000001", timestamps=np.array([0, 1, 2]),
                     feature=np.random.randn(3, 512).astype(np.float32)),
        VideoFeature(video_id="R000002", timestamps=np.array([0, 1]),
                     feature=np.random.randn(2, 512).astype(np.float32)),
    ]

    test_path = os.path.join(REPO_ROOT, "_smoke_test_features.npz")
    store_features(test_path, features)
    loaded = load_features(test_path)
    os.remove(test_path)
    print(f"  Store/load: OK ({len(loaded)} features round-tripped)")
    return True


def test_transforms():
    """Test image transforms."""
    print("\n=== Test 5: Image transforms ===")
    sys.path.insert(0, os.path.join(REPO_ROOT, "VSC22-Descriptor-Track-1st", "infer"))
    from src.transform import sscd_transform, vit_transform
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    t256 = sscd_transform(256, 256)
    t384 = vit_transform(384, 384)
    out1 = t256(img)
    out2 = t384(img)
    print(f"  sscd_transform(256): {out1.shape}")
    print(f"  vit_transform(384):  {out2.shape}")
    return True


def test_infer_ref_sh_syntax():
    """Verify infer_ref.sh has valid bash syntax."""
    print("\n=== Test 6: Shell script syntax ===")
    sh_path = os.path.join(REPO_ROOT, "VSC22-Descriptor-Track-1st", "infer", "infer_ref.sh")
    result = subprocess.run(["bash", "-n", sh_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  infer_ref.sh: valid syntax")
    else:
        print(f"  infer_ref.sh: SYNTAX ERROR: {result.stderr}")
        return False
    return True


def main():
    print("VSC22 Pipeline Smoke Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_device_detection,
        test_faiss,
        test_storage,
        test_transforms,
        test_infer_ref_sh_syntax,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
