import sys
import matplotlib.pyplot as plt
import numpy as np
import ctypes

class DoubleVector(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)),
                ("size", ctypes.c_size_t)]

class DoubleVectorVector(ctypes.Structure):
    _fields_ = [("data",ctypes.POINTER(DoubleVector)),
                ("size",ctypes.c_size_t)]
    

def py_list_to_double_vector(py_list):
    data = (ctypes.c_double * len(py_list))(*py_list)
    return DoubleVector(data, len(py_list))

def py_list_to_double_vector_vector(py_list):
    data = (DoubleVector * len(py_list))()
    for i, sublist in enumerate(py_list):
        data[i] = py_list_to_double_vector(sublist)
    return DoubleVectorVector(ctypes.cast(data, ctypes.POINTER(DoubleVector)), len(py_list))


def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

def read_file(filename):
    try:
        return np.genfromtxt(filename, delimiter=',')
    except FileNotFoundError:
        print("Error: file {filename} not found.")

def visualize(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

def py_str_to_c_str(py_str):
    return ctypes.c_char_p(py_str.encode())

def main():
    if len(sys.argv) != 3:
        print("Usage: python tsne.py <filename> <perplexity>")
        return
    
    filename = sys.argv[1].strip()
    perplexity = float(sys.argv[2])
    lib = ctypes.CDLL('libtsne.so')

    lib.run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
    
    lib.run(py_str_to_c_str(filename), 2, perplexity)

    output = read_data("/Users/Akash/Documents/Projects/CS/tsne-cpp/tmp/data")
    visualize(output)

if __name__ == "__main__":
    main()
