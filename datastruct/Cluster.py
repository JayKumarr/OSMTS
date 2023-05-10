from datastruct import MCluster as con

class Cluster:
    def __init__(self, CF):
        self.widFreq = CF[con.Z_w]  # maintaining wordId and the occurance
        self.widToWidFreq = CF[con.Z_v]
