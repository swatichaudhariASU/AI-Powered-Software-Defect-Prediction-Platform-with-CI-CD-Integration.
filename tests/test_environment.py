def test_project_environment_imports():
    import pandas
    import numpy
    import sklearn
    import git
    import radon
    import fastapi

    assert pandas.__version__
    assert numpy.__version__
    assert sklearn.__version__
    assert git.__version__
    assert radon.__version__
    assert fastapi.__version__
