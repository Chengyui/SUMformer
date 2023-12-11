import datetime
import time
def phv_time_stamp_generate(min=30,start_time_str = '2023-02-26 00:00:00',end_time_str = '2023-07-23 00:00:00'):

    start_time_str = '2015-01-01 00:00:00'
    end_time_str = '2015-03-01 23:30:00'
    start_dt = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

    # interval 30min
    delta = datetime.timedelta(minutes=min)

    # 存储时间戳的列表
    timestamps = []

    dt = start_dt
    while dt <= end_dt:
        timestamps.append(dt.strftime('%Y%m%d%H%M%S'))
        dt += delta

    return timestamps


timestamps = phv_time_stamp_generate()

