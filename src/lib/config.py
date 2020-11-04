class Config():
    """ 
    Config class passed to the dataset class.
    """

    # Data
    sampling_rate = 100
    x_interval_mins = 30
    x_interval_seconds = x_interval_mins * sampling_rate
    
    # Matplotlib
    default_fig_size = (10,5)

    # Label values
    AWAKE = 0 # Awake
    N1 = 1 # N1
    N2 = 2 # N2
    N3 = 3 # N3
    REM = 4 # REM
    UNKNOWN = 5 # UNKNOWN
    
    label_dict = {
        AWAKE : "W",
        N1 : "A",
        N2 : "B",
        N3 : "C",
        REM : "R",
        UNKNOWN : "U",

        "W" : AWAKE,
        "A" : N1,
        "B" : N2,
        "C" : N3,
        "R" : REM,
        "U" : UNKNOWN,
    }

    stage_name = {
        AWAKE : "AWAKE",
        N1 : "N1",
        N2 : "N2",
        N3 : "N3",
        REM : "REM",
        UNKNOWN : "N5"
    }

    ann2label = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
        "Sleep stage ?": 5,
        "Movement time": 5
    }
