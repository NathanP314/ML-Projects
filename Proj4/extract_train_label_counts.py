import numpy as np
import struct
filepath = "C:\\Users\\natep\\Downloads\\MNIST\\MNIST\\train-labels.idx1-ubyte"
with open(filepath, "rb") as f:
    magic, num = struct.unpack(">II", f.read(8))
    if magic != 2049:
        raise ValueError(f"Invalid magic number in label file: {magic}")
    labels = np.frombuffer(f.read(), dtype=np.uint8)
counts = np.bincount(labels, minlength=10)
for digit, count in enumerate(counts):
    print(f"Label {digit}: {count}")
print(f"Total count: {np.sum(counts)}")