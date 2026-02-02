def print_environment():
    import os, sys, platform
    import numpy as np

    print("=== BOREAS environment ===")
    print("Python:", sys.version.replace("\n"," "))
    print("Platform:", platform.platform())
    print("Executable:", sys.executable)
    print("CWD:", os.getcwd())

    try:
        import scipy
        print("numpy:", np.__version__)
        print("scipy:", scipy.__version__)
    except Exception as e:
        print("Could not import scipy:", e)

    # show where boreas is imported from (helps catch shadowing)
    try:
        import boreas
        print("boreas imported from:", boreas.__file__)
    except Exception as e:
        print("Could not import boreas:", e)

    print("==========================")