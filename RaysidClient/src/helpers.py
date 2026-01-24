from numba import njit, types
import numpy as np

# @njit("int16(uint8, uint8)", cache=True)
def two_bytes_to_int(b1: np.uint8, b2: np.uint8) -> np.int16:
    return np.uint16(b2 & 0xFF) * 256 + (b1 & 0xFF)


# @njit("int32(uint8, uint8, uint8)", cache=True)
def three_bytes_to_int(b3: np.uint8, b2: np.uint8, b1: np.uint8) -> np.int32:
    return np.uint32(np.uint32(b3) | (np.uint32(b2) << 8) | (np.uint32(b1) << 16))

# @njit(inline='always')
def unpack_value(v):
    mult10 = v // 6000
    res = v % 6000
    return res * (10 ** mult10)

def long_to_bytes(l):
    return bytes([(l>>24)&0xFF, (l>>16)&0xFF, (l>>8)&0xFF, l&0xFF])