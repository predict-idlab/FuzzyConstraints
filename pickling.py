import os
import pickle

def dump_to_pickle(data, setup, file_path):
    """Write a binary dump to a pickle file of arbitrary size."""
    """misc: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb"""
    bytes_out = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    with open("pickles/pickle_" + setup + "/" + file_path, 'wb') as f_out:
        # 2**31 - 1 is the max nr. of bytes pickle can dump at a time
        for idx in range(0, len(bytes_out), 2 ** 31 - 1):
            f_out.write(bytes_out[idx:idx + 2 ** 31 - 1])
    return


def load_from_pickle(setup, file_path):
    """Read from a pickle file of arbitrary size."""
    """misc: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb"""
    bytes_in = bytearray(0)
    input_size = os.path.getsize("pickles/pickle_" + setup + "/" + file_path)
    with open("pickles/pickle_" + setup + "/" + file_path, 'rb') as f_in:
        for _ in range(0, input_size, 2 ** 31 - 1):
            bytes_in += f_in.read(2 ** 31 - 1)
    return pickle.loads(bytes_in)