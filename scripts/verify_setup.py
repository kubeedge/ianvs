"""
Verify that the correct KubeEdge Sedna package is installed.

WARNING: Running 'pip install sedna' installs a completely wrong
PyPI package (a unit conversion library). The correct KubeEdge
Sedna must be installed from the bundled wheel file:

    pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl
"""
import sys


def check_sedna():
    """Check if the correct KubeEdge Sedna is installed."""
    print("Checking Sedna installation...")
    try:
        # pyrefly: ignore [missing-import]
        import sedna  # noqa: F401
    except ImportError:
        print("\nSedna is not installed.")
        print("\nCorrect installation:")
        print("  pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl")
        return False

    # check if it is the RIGHT sedna by verifying KubeEdge-specific module
    try:
        # pyrefly: ignore [missing-import]
        import sedna.datasources  # noqa: F401
        print("KubeEdge Sedna is installed correctly.")
        return True
    except ImportError:
        print("\nWRONG sedna package detected!")
        print("\nYou have the unrelated 'sedna' unit-conversion package from PyPI.")
        print("\nTo fix:")
        print("  1. Uninstall the wrong package:")
        print("     pip uninstall sedna -y")
        print("  2. Install the correct KubeEdge Sedna:")
        print("     pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl")
        print("\nNever run 'pip install sedna' for Ianvs.")
        return False


def check_ianvs():
    print("\nChecking Ianvs installation...")
    try:
        import core
        print("Ianvs is installed correctly.")
        return True
    except ImportError:
        print("Ianvs is not installed.")
        print("  Run: python setup.py install")
        return False


if __name__ == "__main__":
    sedna_ok = check_sedna()
    ianvs_ok = check_ianvs()

    if sedna_ok and ianvs_ok:
        print("\nEnvironment setup complete. Ready to run benchmarks.")
        sys.exit(0)
    else:
        print("\nEnvironment setup incomplete. Fix the issues above first.")
        sys.exit(1)