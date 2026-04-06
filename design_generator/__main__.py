"""Entry point for python -m design_generator.

Dispatches to generate.main() to handle CLI invocation.
"""
import sys
from design_generator.generate import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

