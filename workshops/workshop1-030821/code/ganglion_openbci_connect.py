import argparse
import time
import datetime
import brainflow
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *


def ganglion_connect(secondes):
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()
    
    params = BrainFlowInputParams()
    params.serial_port = 'COM5'
    params.mac_address = 'f8:01:dc:1b:8a:e5'
    # params.ip_address = ''

    board = BoardShim(BoardIds.GANGLION_BOARD, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(secondes)  
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    return board, data


def write_to_log(file_name, data, mode = 'w'):
    path = "dataset\\%s_%s.csv" %(file_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(path, mode) as f:
        np.savetxt(f, data, fmt = '%.4f', delimiter = ',')


if __name__ == "__main__":
    # if True then write to log
    logging = True

    # sample duration in seconds
    sample_duration = 5

    sampling_freq = 200

    # which eeg chancles to record
    eeg_channels = [2, 3]
    
    board, data = ganglion_connect(sample_duration)
    if logging:
        write_to_log('eeg_data', [data[i] for i in eeg_channels])