import torch
import os
import sys


def check_cuda():
    print("=" * 60)
    print("CHECKING CUDA SETUP")
    print("=" * 60)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

        # Test GPU
        print("\nTesting GPU...")
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("✓ GPU test successful!")
        except Exception as e:
            print(f"✗ GPU test failed: {e}")
            return False
    else:
        print("✗ CUDA not available!")
        return False

    return True


def check_files():
    print("\n" + "=" * 60)
    print("CHECKING DATA FILES")
    print("=" * 60)

    required_files = [
        "data/combo_training_data.json",
        "data/pauper_cards_detailed.json",
        "data/known_combos.json",
    ]

    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"{status} {file}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n⚠ Missing files! Please run: python collect_combo_data.py")

    return all_exist


def check_imports():
    print("\n" + "=" * 60)
    print("CHECKING PYTHON PACKAGES")
    print("=" * 60)

    packages = [
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "accelerate",
    ]

    all_installed = True
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False

    if not all_installed:
        print("\n⚠ Missing packages! Install with:")
        print("pip install transformers datasets accelerate bitsandbytes peft trl")

    return all_installed


def main():
    print("\n🔍 PRE-TRAINING DIAGNOSTICS\n")

    cuda_ok = check_cuda()
    imports_ok = check_imports()
    files_ok = check_files()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CUDA Setup: {'✓ OK' if cuda_ok else '✗ FAILED'}")
    print(f"Packages: {'✓ OK' if imports_ok else '✗ FAILED'}")
    print(f"Data Files: {'✓ OK' if files_ok else '✗ FAILED'}")

    if cuda_ok and imports_ok and files_ok:
        print("\n✓ Everything looks good! You can run: python train_gemma.py")
        return 0
    else:
        print("\n✗ Please fix the issues above before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
