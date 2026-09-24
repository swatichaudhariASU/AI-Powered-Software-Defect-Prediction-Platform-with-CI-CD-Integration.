"""Verify the Month 1 Python environment."""

def main():
    import pandas
    import numpy
    import sklearn
    import git
    import radon
    import lizard
    import fastapi

    print("Environment check passed.")
    print(f"pandas={pandas.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"scikit-learn={sklearn.__version__}")
    print(f"GitPython={git.__version__}")
    print(f"FastAPI={fastapi.__version__}")


if __name__ == "__main__":
    main()
