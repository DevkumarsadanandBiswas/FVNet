import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run without installing
from fvnet.cli import preprocess_main

if __name__ == "__main__":
    preprocess_main()
