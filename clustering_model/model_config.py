from ssclassifier import ccutils as ut

class model_config:

    def __init__(self, initial_sample_index, ALPHA, BETA, LAMDA, applyDecay=True, applyICF = True, applyCWW = True, single_term_clustering = True, FR_THRESHOLD=-1, local_vocabulary_beta = False, merge_old_cluster = False,
                 mclus_beta_multi = 1, new_vocabulary_for_beta = False, mature_threshold=2, dataset_name="sample_dataset", label_count=0, max_cluster_for_each_label = 1, labelled_data_percentage=0, total_instances = 0, initial_cardinality = 1, label_coccurence_prob = {}):
        self.alpha = ALPHA
        self.beta = BETA
        self.applyDecay = applyDecay
        self.applyICF = applyICF
        self.applyCWW = applyCWW
        self.atleast_one_term_matched_for_clustering = single_term_clustering
        self.applyFeatureReduction = False
        if (FR_THRESHOLD > 0):
            self.applyFeatureReduction = True
        self.feature_threshold = FR_THRESHOLD
        self.local_cluster_vocabulary_beta = local_vocabulary_beta  # if we want to calculate beta according to related cluster vocabulary. if not it will calculate the model active vocabulary beta
        self.merge_old_cluster = merge_old_cluster
        self.mclus_beta_multi = mclus_beta_multi
        self.new_vocabulary_for_beta = new_vocabulary_for_beta

        self.lamda = LAMDA

        self.mature_threshold = mature_threshold

        self.output_file = None
        self.last_index_traversed = 0
        self.dataset_name = dataset_name

        self.total_label = label_count
        self.max_clusters_for_each_label = max_cluster_for_each_label
        self.min_active_clusters = self.total_label*max_cluster_for_each_label

        self.LABELED_INSTANCE_PERCENTAGE = labelled_data_percentage
        self.total_instances = total_instances
        self.initial_sample_indexes = initial_sample_index

        self.cardinality = initial_cardinality
        self.label_cooccurence_probability = label_coccurence_prob

        self.labeled_sample_indexes = ut.generate_random_index(0, self.total_instances - 1, self.LABELED_INSTANCE_PERCENTAGE, excluded_indexes=self.initial_sample_indexes)

        self.output_file = self.set_output_file()

        self.model_log = ""


    def redefine(self, ALPHA, BETA, LAMDA, decay, applyICF, applyCWW, single_term_clustering, feature_threshold,local_cluster_vocabulary_beta,merge_old_cluster,mclus_beta_multi,include_new_vocabulary):
        self.alpha = ALPHA
        self.beta = BETA
        self.applyDecay = decay
        self.applyICF = applyICF
        self.applyCWW = applyCWW
        self.atleast_one_term_matched_for_clustering = single_term_clustering
        self.applyFeatureReduction = False
        if (feature_threshold > 0):
            self.applyFeatureReduction = True
        self.feature_threshold = feature_threshold
        self.local_cluster_vocabulary_beta = local_cluster_vocabulary_beta  # if we want to calculate beta according to related cluster vocabulary. if not it will calculate the model active vocabulary beta
        self.merge_old_cluster = merge_old_cluster
        self.mclus_beta_multi = mclus_beta_multi
        self.new_vocabulary_for_beta = include_new_vocabulary
        self.lamda = LAMDA
        # print("ALPHA-",ALPHA, ";BETA-", BETA, ";LAMDA-", LAMDA, ";decay-", decay, ";applyICF-",applyICF, ";applyCWW-",applyCWW, ";single_term_clustering-" , single_term_clustering, ";feature_threshold-" , feature_threshold, ";local_cluster_vocabulary_beta-" , local_cluster_vocabulary_beta, ";merge_old_cluster-" , merge_old_cluster, ";mclus_beta_multi-" , mclus_beta_multi, ";include_new_vocabulary-" , include_new_vocabulary)
        self.output_file = self.set_output_file()

        self.model_log = ""


    def set_output_file(self):
        file_name=("results/"+self.get_model_configuration()+".csv")
        # self.output_file = open(file_name, "w")
        self.output_file = dummy_file(file_name)
        return self.output_file

    def mlog(self, str):
        self.model_log = (self.model_log+"\n"+str)

    def get_model_configuration(self):
        rt_str=  "results-"+self.dataset_name
        if self.applyICF:
            rt_str = (rt_str+"_ICF")
        if self.applyCWW:
            rt_str = (rt_str+"_CWW")
        if self.atleast_one_term_matched_for_clustering:
            rt_str = (rt_str + "_STC")
        if self.local_cluster_vocabulary_beta:
            rt_str = (rt_str+"_LCB")

        string = "_ALPHA{0}_BETA{1}_LAMBDA{2}_GAMMA{3}_MXMC{4}".format(self.alpha, self.beta, self.lamda, self.feature_threshold, self.max_clusters_for_each_label)
        rt_str = (rt_str + string)
        return rt_str

class dummy_file:  # when we don't want a file to be output
    def __init__(self, f_path):
        self.path = f_path
        do_nothing =10
    def write(self, *arg):
        do_nothing = 10

    def close(self):
        do_nothing = 10
    def __str__(self):
        return "<io.file_"+self.path+">"