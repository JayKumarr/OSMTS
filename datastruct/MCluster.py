# Z_n = 'Z_n'     # docs()  total number of documents
# Z_w = 'Z_w'     # cluster word frequency/probability
# Z_tw = 'Z_tw'    # word arrival time stamps
# Z_v = 'Z_v'    # cluster word to word co-occurence
# Z_tv = 'Z_tv'    # word arrival time stamps
# Z_l = 'Z_l'     # total number of words in cluster
# Z_c = 'Z_c'     # assigned class, None if no class
# Z_f = 'Z_f'     # decay weight
# Z_u = 'Z_u'     # last updated time stamp
# Z_tg = 'Z_tg'    # cluster time-stamp of created
# Z_pd = 'Z_pd'    # previously deleted, extension if the cluster updated after first time deletion


Z_n = 0     # total number of documents
Z_w = 1     # cluster word frequency/probability
Z_tw = 2    # word arrival time stamps
Z_v = 3    # cluster word to word co-occurence
Z_tv = 4    # word arrival time stamps
Z_l = 5     # total number of words in cluster
Z_c = 6     # assigned class, None if no class or None
Z_f = 7     # decay weight
Z_u = 8     # last updated time stamp
Z_tg = 9    # cluster time-stamp of created
Z_pd = 10    # previously deleted, extension if the cluster updated after first time deletion


class MCluster:
    def __init__(self,  model, mature_threshold, label_id):
        self.CF = {}
        self.model = model
        self.cluster_id = model.get_new_cluster_id()

        self.MATURE_THRESHOLD = mature_threshold

        # initialization cluster features
        self.CF[Z_n] = set()  # docs
        self.CF[Z_w] = {}  # word frequency
        self.CF[Z_v] = {}  # word2word occurance
        self.CF[Z_f] = 1.0  # decay weight
        self.CF[Z_l] = 0    # total words
        self.CF[Z_tw] = {}  # words arrival time sequence
        self.CF[Z_tv] = {}  # words co-occurence arrival time sequence
        self.CF[Z_tg] = self.model.currentTimestamp
        self.CF[Z_c] = label_id
        # print("cluster created Label: {0}".format(label_id))


    def add_document(self, document):
        self.CF[Z_u] = self.model.currentTimestamp
        self.CF[Z_f] = 1.0
        self.CF[Z_n].add(document.doc_id)
        # update feature of cluster
        for w, w_freq in document.CF[Z_w].items():

            # helps to calculate ICF, if this word is not contained by widClusMap then add it
            if w not in self.model.widClusid:  # updating widClusid
                self.model.widClusid[w] = set()
                self.model.widClusid[w].add(self.cluster_id)
            else:
                self.model.widClusid[w].add(self.cluster_id)



            # ------single term adding into cluster feature--------
            if w not in self.CF[Z_w]:
                self.CF[Z_w][w] = 0
                self.CF[Z_tw][w] = []
                self.CF[Z_v][w] = {}
                self.CF[Z_tv][w] = {}
            elif w not in self.CF[Z_v]:
                self.CF[Z_v][w] = {}
                self.CF[Z_tv][w] = {}

            self.CF[Z_w][w] = self.CF[Z_w][w] + w_freq  # update word frequency in cluster
            self.CF[Z_l] = self.CF[Z_l] + w_freq  # increasing number of words in cluster

            # ------co-occurence term adding in cluster feature---------
            for w2 in document.CF[Z_w]:  # updating CF[cww] word to word frequency
                if w != w2:
                    if w2 not in self.CF[Z_v][w]:
                        self.CF[Z_v][w][w2] = document.CF[Z_v][w][w2]
                        self.CF[Z_tv][w][w2] = []
                    else:
                        self.CF[Z_v][w][w2] = self.CF[Z_v][w][w2] + document.CF[Z_v][w][w2]

            #----------if feature decay Enable, then maintain the time arrival of single words, and word co-occurence
            if (self.model.config.applyFeatureReduction):  # if true then maintain term arrival time
                # update arrival time of wid
                self.CF[Z_tw][w].append(self.CF[Z_n].__len__())

                for w2 in document.CF[Z_w]:  # updating CF[cww] word to word frequency
                    if w != w2:
                        self.CF[Z_tv][w][w2].append(self.CF[Z_n].__len__())

        # assert (self.CF[Z_w].__len__() == self.CF[Z_tw].__len__()), "The Z_w is not equal to Z_tw of cluster {}".format(self.cluster_id)
        # assert (self.CF[Z_v].__len__() == self.CF[Z_tv].__len__()), "The Z_v is not equal to Z_tv of cluster {}".format(self.cluster_id)


    def is_mature(self):
        if (self.CF[Z_n]) >= self.MATURE_THRESHOLD:
            return True
        else:
            return False


    def add_micro_cluster(self, mccluster_to_be_merged):
        print("cluster merged",self.cluster_id," <- ", mccluster_to_be_merged.cluster_id)
        # adding documents
        for docId in mccluster_to_be_merged.CF[Z_n]:
            self.CF[Z_n].add(docId)
            self.model.active_documents[docId].remove(mccluster_to_be_merged.cluster_id)
            self.model.active_documents[docId].add(self.cluster_id)

        for wid, w_freq in mccluster_to_be_merged.CF[Z_w].items():
            # helps to calculate ICF, if this word is not contained by widClusMap then add it
            self.model.widClusid[wid].remove(mccluster_to_be_merged.cluster_id)
            if self.cluster_id not in self.model.widClusid[wid]:
                self.model.widClusid[wid].add(self.cluster_id)

            if wid not in self.CF[Z_w]:
                self.CF[Z_w][wid] = 0
                self.CF[Z_v][wid] = {}
                self.CF[Z_tw][wid] = []
                self.CF[Z_tv][wid] = {}
            elif wid not in self.CF[Z_v]:
                self.CF[Z_v][wid] = {}
                self.CF[Z_tv][wid] = {}
            self.CF[Z_w][wid] = self.CF[Z_w][wid] + w_freq  # update word frequency in cluster
            self.CF[Z_l] = self.CF[Z_l] + w_freq  # increasing number of words in cluster

            for linked_w2, w2_weight in mccluster_to_be_merged.CF[Z_v][wid].items():  # updating CF[cww] word to word frequency
                if linked_w2 not in self.CF[Z_v][wid]:
                    self.CF[Z_v][wid][linked_w2] = w2_weight
                    self.CF[Z_tv][wid][linked_w2] = []
                else:
                    self.CF[Z_v][wid][linked_w2] = self.CF[Z_v][wid][linked_w2] + w2_weight

            if (self.model.config.applyFeatureReduction):
                self.CF[Z_tw][wid].extend(mccluster_to_be_merged.CF[Z_tw][wid])
                for linked_w2, time_list in mccluster_to_be_merged.CF[Z_tv][wid].items():
                    self.CF[Z_tv][wid][linked_w2].extend(time_list)

        # assert (self.CF[Z_w].__len__() == self.CF[Z_tw].__len__()), "The Z_w is not equal to Z_tw of cluster {}".format(self.cluster_id)
        # assert (self.CF[Z_v].__len__() == self.CF[Z_tv].__len__()), "The Z_v is not equal to Z_tv of cluster {}".format(self.cluster_id)



    def is_old(self, threshold, LAMDA=None):
        if LAMDA is None:
            LAMDA = self.model.lamda

        lastupdated = self.CF[Z_u]
        power = -LAMDA * (self.model.currentTimestamp - lastupdated)
        decay = pow(2, power)
        self.CF[Z_f] = self.CF[Z_f] * decay

        if self.CF[Z_f] < threshold:
            return True
        else:
            return False
