import numpy as np
from datetime import datetime
import pandas as pd
from .Generate_TAXI_PHV_hotmap import phv_time_stamp_generate

def string2timestamp(strings, T=144):
    """
    :param strings:
    :param T:
    :return:
    example:
    str = [b'2013070101', b'2013070102']
    print(string2timestamp(str))
    [Timestamp('2013-07-01 00:00:00'), Timestamp('2013-07-01 00:30:00')]
    """
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day= int(t[:4]), int(t[4:6]), int(t[6:8])
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(t[8:10]),
                                                minute=int(t[10:12]))))

    return timestamps


strings  = phv_time_stamp_generate(min=30)
phv_timestamp = string2timestamp(strings)
np.save("NYC_timestamp.npy",phv_timestamp)