import collections
import math
from datastruct.Words import Words
from datastruct import MCluster
from ssclassifier import ccutils as ut
import numpy as np

K_N_C = 10

class Model:
    """
    ALPHA  {float}
    BETA   {float}
    LAMDA  {float}
    applyDecay  {boolean}   apply the fading/decay (LAMDA) factor with every micro-cluster, if fading weight reaches to zero, it will delete the micro-cluster
    applyICF    {boolean}   apply inverse cluster frequency while calculating the term similarity score
    applyCWW    {boolean}   apply co-occurence term while calculating document micro-cluster similarity
    FR_THRESHOLD  {float}(0,1] apply feature reduction using triangular time decay function, if value = -1 then it will not apply feature reduction   
    local_vocabulary_beta {boolean}   while calculating Vocabulary*Beta [the denominator] ,  if False then it will count vocabulary from all cluster, if True then it will count vocabulary of one cluster 
    merge_old_cluster {boolean}   it will merge old cluster according to given condition of probability
    new_vocabulary_for_beta {boolean}   it will merge cluster/model vocabulary with document vocabulary, then used to calculate Vocabulary*Beta
    self.wid_docId = {}      # wordID, documentId:    updated by Document
    """
    def __init__(self, configuration):
        self.config = configuration

        self.words = Words()
        # self.wid_docId = wid_docId
        self.active_clusters = {}   #   clusterID -> [ MCluster(cn, cw, cwf, cww, cd, csw), ... ]
        self.active_documents = {}   # [documentId1, ...]
        self.widClusid = {}   # {wordID ,clusterID }: to know that how many cluster this word has occurred

        self.deletedDocIdClusId = {} # those documents which are deleted while deleting the cluster, #this DS will be utilized to print output

        self.word_counter = {0:0}   # used for generating new index for each word
        self.cluster_counter = {0:0}  # used for generating new index for each cluster

        self.currentTimestamp = 1

        self.last_index_traversed = 0

        self.label_clusters = {}  # {label_id:[clus_id1, clus_id2,...], }

        self.plots_folder = None

        self.number_of_skipped_documents = 0   # if we skip instances either the length of certain documents is lesser than some value
        self.skipped_documents_notify_text = ""
        self.config.mlog(self.config.output_file.__str__())




    def processDocument(self, document, labels = None):
        self.active_documents[document.doc_id] = set()

        self.sampleCluster(document,labels = labels)
        if (self.config.applyDecay == True):
            if self.active_clusters.__len__() > self.config.min_active_clusters:
                self.checkOldClusters(self.config.lamda)
        if (self.config.applyFeatureReduction):
            for clus_id, doc in self.active_clusters.items():
                self.check_cluster_outdated_features(doc, self.config.feature_threshold)  # UPDATED LINE

        self.currentTimestamp += 1

    def sampleCluster(self, document, labels = None):
        clusIdOfMaxProb = -1
        clusMaxProb = np.longdouble(0.0)
        probabilities_with_cluster_ids = {}

        #to plot the probability distribution towards existing cluster
        # probabilities_with_labels = {}

        N = self.active_documents.__len__()  # number of maintained documents, some documents might be deleted from cluster
        VintoBETA = self.get_vocabulary_into_Beta()
        beta_sum = 0.0
        count_related_clusters = 0

        # need to calculate probablity of existing clusters, if no existing cluster this loop will be skipped
        for clusId, cluster in self.active_clusters.items():


            if (self.config.atleast_one_term_matched_for_clustering):
                doc_wids = document.CF[MCluster.Z_w].keys()
                common_wids = ut.intersection(cluster.CF[MCluster.Z_w].keys(), doc_wids)
                if common_wids.__len__() < 1:
                    continue

            not_matched_terms = document.CF[MCluster.Z_w].__len__() - common_wids.__len__()

            # --- updation for beta calculation
            if (self.config.local_cluster_vocabulary_beta):
                v_size = float(cluster.CF[MCluster.Z_w].__len__())
                if (self.config.new_vocabulary_for_beta):
                    v_size = v_size + (doc_wids.__len__() - common_wids.__len__())
                VintoBETA = float(self.config.beta)*v_size
                beta_sum+=VintoBETA
                count_related_clusters+=1

            m_z = cluster.CF[MCluster.Z_n].__len__()
            eqPart1 = float(m_z) / float(( N-1 + self.config.alpha*N))
            eqPart2Nominator = 1.0
            eqPart2Denominator = 1.0

            i = 0  # represent word count in document

            ##-----------homogeneity updataed --------
            for w in common_wids:
                widFreqInClus = cluster.CF[MCluster.Z_w][w]
                icf = 1.0
                if (self.config.applyICF):  # This condition is used to control parameters by main method
                    icf = self.ICF(w)
                term_weight = widFreqInClus * icf
                freq = document.CF[MCluster.Z_w][w]
                for j in range(freq):
                    eqPart2Nominator *= (term_weight + self.config.beta + j)
            if not_matched_terms > 0:
                # penalty_score = (not_matched_terms*( abs(not_matched_terms-common_wids.__len__())+1 ))* self.config.beta
                penalty_score = not_matched_terms/document.CF[MCluster.Z_w].__len__()
                eqPart2Nominator *= penalty_score
            ##---------------

            # outlier_terms_count = cluster.CF[MCluster.Z_w].__len__()-common_wids.__len__()

            for w, cw_freq in cluster.CF[MCluster.Z_w].items(): # penalty for large size of cluster dimension
                if w in common_wids:
                    continue
                # widFreqInClus = 0
                # if w in common_wids: #if the word of the document exists in cluster
                #     widFreqInClus = CF[MCluster.Z_w][w]
                #
                # icf = 1.0
                # if (self.config.applyICF):  # This condition is used to control parameters by main method
                #     icf = self.ICF(w)
                # term_weight = widFreqInClus*icf
                # for j in range(freq):
                i += 1
                # eqPart2Denominator *=  ( (  (cluster.CF[MCluster.Z_w].__len__())+(VintoBETA) ) + i)   # this creates problem with very high dimension
                eqPart2Denominator += (((cw_freq)*(VintoBETA)+i)  )



            eqPart2 = eqPart2Nominator / eqPart2Denominator
            if (self.config.applyCWW == True): # to control applying CWW from main method
                eqPart2 = eqPart2 * self.addingWidToWidWeightInEqPart2(document,cluster.CF,eqPart2)

            clusProb = eqPart1 * eqPart2
            # to plot the probability distribution towards existing cluster
            # if cluster.CF[MCluster.Z_c] == None:
            #     prob_label = (-1 * clusId)
            # else:
            #     prob_label = cluster.CF[MCluster.Z_c]
            #
            # if prob_label not in probabilities_with_labels:
            #     probabilities_with_labels[prob_label] = 0.0
            # if probabilities_with_labels[prob_label] < clusProb:
            #     probabilities_with_labels[prob_label]= clusProb

            probabilities_with_cluster_ids[clusId] = clusProb
            if clusProb > clusMaxProb:
                clusMaxProb = clusProb
                clusIdOfMaxProb = clusId
        # end for , all probablities of existing clusters have been calculated

        predicted_labels = None

        if (self.config.local_cluster_vocabulary_beta) and (count_related_clusters > 0):
            VintoBETA = float(beta_sum)/float(count_related_clusters)

        # need to calculate probablity of creating a new cluster
        eqPart1 = (self.config.alpha * N) / (N - 1 + self.config.alpha * N)
        eqPart2Nominator = 1.0
        eqPart2Denominator = 1.0
        i = 0 # represent word count in document
        for w,freq in document.CF[MCluster.Z_w].items():

            for j in range(freq):
                i += 1
                eqPart2Nominator*= (self.config.beta+j)
                eqPart2Denominator*= (VintoBETA+i)
        probNewCluster = eqPart1*(eqPart2Nominator/eqPart2Denominator)

        if probabilities_with_cluster_ids.__len__()>0:
            # sort clusters by probability score descending
            sort_dict_clusif_prob = sorted(probabilities_with_cluster_ids.items(), key=lambda x: x[1], reverse=True)
            nearest_clusters = {}
            strr = "There is a problem with code - timestampe: {0}".format(self.currentTimestamp) + " "+str(sort_dict_clusif_prob)
            assert sort_dict_clusif_prob[0][0] == clusIdOfMaxProb, strr
            # x = []
            y = []
            # annotations = []
            idx = 1
            # select atmost K clusters
            for c_id_prob_tuple in sort_dict_clusif_prob:
                if (idx > K_N_C) or c_id_prob_tuple[1] == 0.0:
                    break

                # count the number of clusters of each unique label
                temp_label = self.active_clusters[c_id_prob_tuple[0]].CF[MCluster.Z_c] # extract the label of nearest cluster
                if temp_label not in nearest_clusters:
                    nearest_clusters[temp_label] = []
                nearest_clusters[temp_label].append(c_id_prob_tuple)
                # x.append(idx)
                y.append(c_id_prob_tuple[1])
                # if temp_label == None:
                #     annotations.append( (-1*c_id_prob_tuple[0]) )
                # else:
                #     annotations.append(temp_label)
                idx+=1

            normalized_values = ut.normalize_dict_by_values(nearest_clusters, y)  # need to normalize the nearest probabilities
            label_cluster_count = {} # count the clusters for each label
            for label_id, list_of_tup_clusid_prob in nearest_clusters.items():
                total_number_of_cluster_belong_to_label = list_of_tup_clusid_prob.__len__()
                if total_number_of_cluster_belong_to_label not in label_cluster_count:
                    label_cluster_count[total_number_of_cluster_belong_to_label] = []
                label_cluster_count[total_number_of_cluster_belong_to_label].append(label_id)

            values_above_mean = ut.count_values_above_mean(probabilities_with_cluster_ids.values())
            values_above_mean_knn = ut.count_values_above_mean(y)

            if label_cluster_count.__len__() == 0:
                self.config.mlog("label_cluster_count.__len__() == 0")
                exit(-1)

            predicted_labels = self.extacted_predicted_label(label_cluster_count, nearest_clusters, values_above_mean_knn, normalized_values)

            # ut.export_scatter_plot(self.get_folder_for_plots()+"/s-{0}".format(self.currentTimestamp), x,y,annotations, idx=labels, values_above_mean=values_above_mean, predicted_labels=str(predicted_labels))

            if probNewCluster < clusMaxProb:
                for p_lab in predicted_labels:
                    if p_lab == None:
                        self.config.mlog("Label with none found - timestamp:{0}".format(self.currentTimestamp))
                        self.config.mlog(predicted_labels.__str__())
                        exit(-1)
                    clus_tup = nearest_clusters[p_lab][0]
                    self.active_clusters[clus_tup[0]].add_document(document)
                    self.active_documents[document.doc_id].add(clus_tup[0])
            else:
                for p_lab in predicted_labels:
                    self.createNewCluster(document, p_lab)

        else:
            # self.createNewCluster(document, p_lab)
            self.skipped_documents_notify_text = (self.skipped_documents_notify_text +("-{0}  - \t -{1} - \t -{2}".format(document.doc_id, document.CF[MCluster.Z_w].__len__(), self.active_clusters.__len__()))+"\n" )
            self.number_of_skipped_documents = self.number_of_skipped_documents+1




    def extacted_predicted_label(self,sorted_number_of_KNN_cluster_by_label, nearest_clusters, values_above_mean, normalized_values):
        maximum_count = max(sorted_number_of_KNN_cluster_by_label.keys())
        labels = sorted_number_of_KNN_cluster_by_label[maximum_count]
        if values_above_mean == 1:
            if labels.__len__() == 1:
                return labels
            else:
                max_prob_label = None
                max_prob = 0.0
                for lab in labels:
                    if ut.prob_sum(lab, nearest_clusters.keys(),self.config.label_cooccurence_probability) > max_prob:
                        max_prob_label = lab
                if max_prob_label == None:
                    return labels
                else:
                    return [max_prob_label]
        else:
            if labels.__len__() == 1:
                return labels
            else:
                temp_d = {}
                for lab in labels:
                    prob =ut.prob_sum(lab, sorted_number_of_KNN_cluster_by_label.keys(), self.config.label_cooccurence_probability) + ut.sum_of_tup_values(normalized_values[lab])
                    temp_d[lab] = prob
                sort_dict = sorted(temp_d.items(), key=lambda x: x[1], reverse=True)
                r_list = []
                for v in sort_dict:
                    r_list.append(v[0])
                    if r_list.__len__() == values_above_mean:
                        break
                return r_list


    def get_vocabulary_into_Beta(self, custom_beta= None):
        if (custom_beta is None):
            custom_beta = self.config.beta
        temp = float(custom_beta)*float(self.widClusid.__len__())
        return temp

    def get_new_cluster_id(self):
        self.cluster_counter[0] = self.cluster_counter[0] + 1
        newIndexOfClus = self.cluster_counter[0] # = {}   clusterID -> [ cn, cw, cwf, cww, cd, csw]
        return newIndexOfClus


    # called by classifier, outside of this model class
    def add_cluster(self, mccluster):
        self.active_clusters[mccluster.cluster_id] = mccluster
        for doc_id in mccluster.CF[MCluster.Z_n]:
            if doc_id not in self.active_documents:
                self.active_documents[doc_id] = set()
            self.active_documents[doc_id].add(mccluster.cluster_id)

        if mccluster.CF[MCluster.Z_c] not in self.label_clusters:
            self.label_clusters[mccluster.CF[MCluster.Z_c]] = []

        self.label_clusters[mccluster.CF[MCluster.Z_c]].append(mccluster.cluster_id)  # used to check the related cluster of each label, so that each label should have atleast one or more cluster. we need to keep cluster for each label


    def createNewCluster(self,document, doc_label_id = None):

        new_cluster = MCluster.MCluster(self, self.config.mature_threshold,doc_label_id)
        new_cluster.add_document(document)
        self.active_clusters[new_cluster.cluster_id] = new_cluster
        self.active_documents[document.doc_id].add(new_cluster.cluster_id)

        self.label_clusters[new_cluster.CF[MCluster.Z_c]].append(new_cluster.cluster_id) # used to check the related cluster of each label, so that each label should have atleast one or more cluster. we need to keep cluster for each label



    def addingWidToWidWeightInEqPart2(self,document, CF, eqPart2):
        product = 1.0
        traversed = []
        for wid in document.CF[MCluster.Z_v]:
            if wid not in CF[MCluster.Z_v]:  # if this word not exist in the cluster
                continue
            sumOfProbablitiesOfWid = 0.0
            for wid2 in document.CF[MCluster.Z_v][wid]:
                sumOfProbablitiesOfWid = sumOfProbablitiesOfWid+document.CF[MCluster.Z_v][wid][wid2]
            for wid2 in document.CF[MCluster.Z_v][wid]:
                if wid2 in CF[MCluster.Z_v][wid]:
                    if wid2 not in traversed:
                        weight = CF[MCluster.Z_v][wid][wid2] / sumOfProbablitiesOfWid
                        product = product+weight
            traversed.append(wid)
        return product


    def checkOldClusters(self, LAMDA):
        threshold = 0.00001
        clustersToDelete = set()
        for clusterID, mccluster in self.active_clusters.items():
            if mccluster.is_old(threshold, LAMDA=LAMDA):
                if self.label_clusters[mccluster.CF[MCluster.Z_c]].__len__() > self.config.max_clusters_for_each_label:
                    clustersToDelete.add(mccluster)

        for mc in clustersToDelete:
            if (self.config.merge_old_cluster): #merge_old_cluster
                id = self.check_cluster_to_merge(mc)

                if id != mc.cluster_id:
                    # merge clusters
                    temp = (mc.cluster_id,id)
                    # self.history.merged_clusters.append(temp)
                    self.active_clusters[id].add_micro_cluster(mc)  # merge two cluster
                else:
                    self.deleteOldCluster(mc)
                    self.config.mlog("cluster deleted {0}".format(mc.cluster_id))
            else:
                self.deleteOldCluster(mc)
                self.config.mlog("cluster deleted {0}".format(mc.cluster_id))

            self.active_clusters.pop(mc.cluster_id)  # delete from active cluster


    def deleteOldCluster(self, mcluster):
        # temp = (mcluster.cluster_id, set(mcluster.CF[MCluster.Z_w]))
        # self.history.removed_wids_from_deleted_clusters.append(temp)
        for wid in mcluster.CF[MCluster.Z_w]:  # remove words from self.widClusid

            self.widClusid[wid].remove(mcluster.cluster_id)
            if self.widClusid[wid].__len__() == 0:
                self.widClusid.pop(wid)
                self.words.remove_wid(wid)

        for docId in mcluster.CF[MCluster.Z_n]: # remove documents from self.active_documents,
            try:
                self.active_documents[docId].remove(mcluster.cluster_id)
                if self.active_documents[docId].__len__() == 0:
                    self.active_documents.pop(docId)
                self.deletedDocIdClusId[docId] = mcluster.cluster_id #this DS will be utilized to print output
            except:
                self.config.mlog("current_timestamp : "+self.currentTimestamp)
                exit(-1)




    def ICF(self,wid):
        icf = 1.0
        if self.active_clusters.__len__() < 5:
            icf = 1.0
        else:
            if wid in self.widClusid:
                icf = math.log2( self.active_clusters.__len__()/self.widClusid[wid].__len__())
        return icf



    # this function does not need changing
    def calculate_triangular_time(self, timestamp):
        return (( (timestamp*timestamp) + timestamp )/2)

    # this function does not need changing, calculate recency of terms according to cluster documents
    def check_cluster_outdated_features(self, mccluster, FEATURE_RECENCY_THRESHOLD):

        CF = mccluster.CF
        if CF[MCluster.Z_c] == None:
            assert CF[MCluster.Z_n] < mccluster.MATURE_THRESHOLD, "Cluster is mature and there exist no label till now, need to check is everything okay"
            return False

        wid_to_be_removed = set()
        word_coocurrence_to_be_removed = {}

        cluster_triangular_time = self.calculate_triangular_time(1)
        current_cluster_triangular_time = self.calculate_triangular_time(CF[MCluster.Z_n].__len__())
        real_triangular_number =current_cluster_triangular_time - cluster_triangular_time   + 1
        for w_id in CF[MCluster.Z_w].keys():
            word_coocurrence_to_be_removed[w_id] = []
            list_of_time_stamps = CF[MCluster.Z_tw][w_id]  # sequential number of document in cluster [1,2,3,4,5,6,7.....]
            word_actual_time_values = sum(list_of_time_stamps)
            recency = ((word_actual_time_values*100)/real_triangular_number)
            if recency < FEATURE_RECENCY_THRESHOLD:
                wid_to_be_removed.add(w_id)


            if (self.config.applyCWW) and (w_id in CF[MCluster.Z_tv]):
                # for word-cooccurence recency check
                for linked_w2,ist_of_time_stamps_cww in CF[MCluster.Z_tv][w_id].items():
                    word_actual_time_values = sum(ist_of_time_stamps_cww)
                    recency = ((word_actual_time_values * 100) / real_triangular_number)
                    if recency < FEATURE_RECENCY_THRESHOLD:
                        word_coocurrence_to_be_removed[w_id].append(linked_w2)

            if word_coocurrence_to_be_removed[w_id].__len__() == 0:
                word_coocurrence_to_be_removed.pop(w_id)


        # if wid_to_be_removed.__len__() > 0:  # update co-occurance of related wid
        #     update_coorrence_mtrix_according_to_cluster_features(CF,wid_to_be_removed)

        # deleting single outdated terms
        for wid in wid_to_be_removed:
            self.widClusid[wid].remove(mccluster.cluster_id)
            if self.widClusid[wid].__len__() == 0: #if a word is not used by any cluster then delete it
                self.widClusid.pop(wid)
                self.words.remove_wid(wid)

            CF[MCluster.Z_w].pop(wid)  # deleting from cluster
            CF[MCluster.Z_tw].pop(wid)  # deleting from cluster

            #  -------- ------- ------- -----
        # deleting outdated co-occured terms
        for wid, coocu_wids in word_coocurrence_to_be_removed.items():
            for wid2 in coocu_wids:
                CF[MCluster.Z_v][wid].pop(wid2)
                CF[MCluster.Z_tv][wid].pop(wid2)

            if (CF[MCluster.Z_tv][wid].__len__() == 0) and (wid in wid_to_be_removed):
                CF[MCluster.Z_v].pop(wid)
                CF[MCluster.Z_tv].pop(wid)


    def check_cluster_to_merge(self, mccluster_as_document):

        clusIdOfMaxProb = -1
        clusMaxProb = 0.0

        N = self.active_documents.__len__()  # number of maintained documents, some documents might be deleted from cluster
        v_size = self.widClusid.__len__()
        VintoBETA = self.get_vocabulary_into_Beta(self.config.beta*self.config.mclus_beta_multi)
        beta_sum = 0.0
        count_related_clusters = 0

        # need to calculate probablity of existing clusters, if no existing cluster this loop will be skipped
        for clusId, mc in self.active_clusters.items():
            if (clusId == mccluster_as_document.cluster_id):
                continue

            cluster_wids = mc.CF[MCluster.Z_w].keys()
            doc_wids = mccluster_as_document.CF[MCluster.Z_w].keys()
            common_wids = ut.intersection(cluster_wids, doc_wids)
            if common_wids.__len__() < 1:
                continue

            # --- updation for beta calculation
            if (self.config.local_cluster_vocabulary_beta):
                # VintoBETA = float(self.config.beta) * float( self.union(CF[MCluster.Z_w].keys(), document.widFreq.keys()).__len__() ) # combine vocabulary of both cluster and document to calculate local beta
                v_size = float(mc.CF[MCluster.Z_w].__len__())
                if (self.config.new_vocabulary_for_beta):
                    v_size = v_size + (doc_wids.__len__() - common_wids.__len__() )
                VintoBETA = (float(self.config.beta)*self.config.mclus_beta_multi) * v_size # consider cluster vocabulary to compute beta
                beta_sum += VintoBETA
                count_related_clusters += 1

            numOfDocInClus = mc.CF[MCluster.Z_n].__len__()
            eqPart1 = float(numOfDocInClus) / float((N - 1 + self.config.alpha * N))
            eqPart2Nominator = 1.0
            eqPart2Denominator = 1.0
            numOfWordsInClus = mc.CF[MCluster.Z_l]
            i = 0  # represent word count in document
            for w,freq in mccluster_as_document.CF[MCluster.Z_w].items():
                widFreqInClus = 0
                if w in mc.CF[MCluster.Z_w]:  # if the word of the document exists in cluster
                    widFreqInClus = mc.CF[MCluster.Z_w][w]

                icf = 1.0
                if (self.config.applyICF == True):  # This condition is used to control parameters by main method
                    icf = self.ICF(w)


                for j in range(freq):
                    i += 1
                    eqPart2Nominator *= (widFreqInClus * icf + self.config.beta + j)
                    eqPart2Denominator *= (numOfWordsInClus * VintoBETA + i)

            eqPart2 = eqPart2Nominator / eqPart2Denominator
            if (self.config.applyCWW == True):  # to control applying CWW from main method
                eqPart2 = eqPart2 * self.addingWidToWidWeightInEqPart2(mccluster_as_document, mc.CF, eqPart2)

            clusProb = eqPart1 * eqPart2
            if clusProb > clusMaxProb:
                clusMaxProb = clusProb
                clusIdOfMaxProb = clusId
        # end for , all probablities of existing clusters have been calculated

        probNewCluster = 0.0
        if (mccluster_as_document.CF[MCluster.Z_n].__len__() >  1):
            if (self.config.local_cluster_vocabulary_beta) and (count_related_clusters > 0):
                VintoBETA = float(beta_sum)/float(count_related_clusters)

            # need to calculate probablity of creating a new cluster
            eqPart1 = (self.config.alpha * N) / (N - 1 + self.config.alpha * N)
            eqPart2Nominator = 1.0
            eqPart2Denominator = 1.0
            i = 0 # represent word count in document
            for w, freq in mccluster_as_document.CF[MCluster.Z_w].items():

                for j in range(freq):
                    i += 1
                    eqPart2Nominator*= (self.config.beta+j)
                    eqPart2Denominator*= (VintoBETA+i)
            probNewCluster = eqPart1*(eqPart2Nominator/eqPart2Denominator)
        if probNewCluster < clusMaxProb:
            return clusIdOfMaxProb
        else:
            return mccluster_as_document.cluster_id


    def get_folder_for_plots(self):
        # return "plots"
        if self.plots_folder == None:
            import os
            try:
                self.plots_folder = 'plots/'+self.get_model_configuration(dataset_name=self.config.dataset_name) +"-plot"
                os.makedirs(self.plots_folder)
            except OSError as e:
                do_nothing = 0
        return self.plots_folder

    def close(self):
        self.config.mlog("model closed")
        print(self.config.model_log)

def update_coorrence_mtrix_according_to_cluster_features(CF, removed_features_wids=[]): # this function will delete terms from co-occurence matrix [C_WW] which are not found in [C_WORD_FREQ]
    feature_set = CF[MCluster.Z_w].keys()
    features_coorrences = CF[MCluster.Z_v].keys()
    if removed_features_wids.__len__() == 0:  # if user does not pass removed feature, then we have to create list of feature for deletion by looking at both matrix
        common_terms=ut.intersection(feature_set,features_coorrences) # find active terms, not to be deleted
        if common_terms.__len__() == features_coorrences.__len__():   # if no wid for deletion
            return

        for wid_ww in features_coorrences: # traverse cooccurence matrix to find those terms which have to be deleted
            if wid_ww not in common_terms:
                removed_features_wids.append(wid_ww)
    for expired_wid in removed_features_wids:
        list_of_terms_coccured = CF[MCluster.Z_v][expired_wid] # we have to remove expired term from other linked terms as well
        for linked_wid in list_of_terms_coccured:
            del[CF[MCluster.Z_v][linked_wid][expired_wid]]
        del[CF[MCluster.Z_v][expired_wid]]   # deleting expired term from C_WW


