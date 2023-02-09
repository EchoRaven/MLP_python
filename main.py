import numpy as np

from Network import Network

if __name__ == "__main__":
    net = Network(hiddenStructure=[15, 20, 20, 10, 7, 3], randomInit=True, outputSize=10, inputSize=11, turn=10000, step=0.001)
    net.Train("./data.csv")