from datastruct.Document import Document
from datastruct.dataset import Dataset
from ssclassifier import ccutils as ut
from datastruct.MCluster import MCluster
from clustering_model import OSGM, model_config as mconf
import pickle
import ntpath
import datastruct

def init_m_l_classification(file_name,D_INIT,MAX_MC_FOR_LABEL, MC_MATURE_THRESHOLD, LABELED_INSTANCE_PERCENTAGE, MINIMUM_DIMENSION_DOC, ALPHA=0.00001, BETA=0.00001, LAMDA=0.000001, decay=False, applyICF=False, applyCWW=False, stc=False, feature_threshold=0,local_cluster_vocabulary_beta=False,merge_old_cluster=False,mclus_beta_multi=False,include_new_vocabulary=False):
    X, y, label_count = ut.read_ARFF(file_name)

    total_instances = X._shape[0]
    total_instances_after_filter = 0
    dataset = Dataset(total_instances=total_instances, minimum_dimension_for_doc=MINIMUM_DIMENSION_DOC)
    for idxx in range(0, total_instances):
        total_instances_after_filter = dataset.add_instance( X.rows[idxx], y.rows[idxx])
    X = 0
    y = 0
    print("Dataset Statistics: total/after filter-{0}/{3}, lowest_dimension-{1}, highest_dimension-{2}".format(total_instances, dataset.stat_lowest_dimension_value, dataset.stat_highest_dimension_value, total_instances_after_filter))
    total_instances =total_instances_after_filter
    print("Processing LDA")
    contain_all_label, initial_cardinality, label_cooccurence_prob, label_indexes, sample_indexes=ut.check_all_labels_existence(D_INIT,label_count,dataset.Y)
    if contain_all_label == False:
        print("More indexes has been choosen to collected samples of all labels")

    # the most default parameters are temporary including alpha beta gamma lambda and others, they will be changed by process_m_l_classification()
    model_configuration = mconf.model_config(initial_sample_index=sample_indexes, ALPHA=ALPHA, BETA=BETA, LAMDA=LAMDA, applyDecay=decay, applyICF = applyICF, applyCWW = applyCWW, single_term_clustering = stc, FR_THRESHOLD=feature_threshold,
                       local_vocabulary_beta = local_cluster_vocabulary_beta, merge_old_cluster = merge_old_cluster,
                 mclus_beta_multi = mclus_beta_multi, new_vocabulary_for_beta = include_new_vocabulary, mature_threshold=MC_MATURE_THRESHOLD,
                                             dataset_name=ntpath.basename(file_name), label_count=label_count, max_cluster_for_each_label=MAX_MC_FOR_LABEL, labelled_data_percentage=LABELED_INSTANCE_PERCENTAGE, total_instances=total_instances, initial_cardinality = initial_cardinality,   label_coccurence_prob = label_cooccurence_prob)

    tfidfmodel, dictionary = ut.get_tfidf_model(dataset.X[0:D_INIT - 1])
    labelID_LDA_clusters = {}
    for label_id, sample_index_list in label_indexes.items():     # traverse documents for each label
        class_instances = []        # raw instances of that particular label
        for idx in sample_index_list:
            sample = dataset.X[idx]
            class_instances.append(sample)

        topic_class_doc_map = ut.apply_LDA(tfidfmodel,dictionary,MAX_MC_FOR_LABEL,class_instances)   # split the document of same label into multiple topics using LDA
        labelID_LDA_clusters[label_id] = (topic_class_doc_map, sample_index_list)
    print("--------------Initialization Completed----------")
    return dataset, model_configuration, labelID_LDA_clusters


def process_m_l_classification(dataset, redefined_model_configuration, labelID_LDA_clusters, save_point_interval):

    osgm_model = OSGM.Model(configuration=redefined_model_configuration)
    for label_id, tc_sl in labelID_LDA_clusters.items():
        topic_class_doc_map = tc_sl[0]
        sample_index_list = tc_sl[1]
        class_instances_document = []  # instances of that particular label in form of Document object
        for idx in sample_index_list:
            sample = dataset.X[idx]
            doc_instance = Document(sample, idx, osgm_model)
            class_instances_document.append(doc_instance)

        for topic_id, list_of_doc_index  in topic_class_doc_map.items():   # after identifying the topics, use each topic to construct one micro_cluster
            docs_topic_class = [class_instances_document[d_idx] for d_idx in list_of_doc_index]
            MC_instance = MCluster(osgm_model,redefined_model_configuration.mature_threshold, label_id)
            for doc in docs_topic_class:
                MC_instance.add_document(doc)

            osgm_model.add_cluster(MC_instance)
            assert MC_instance.CF[datastruct.MCluster.Z_w].__len__() == MC_instance.CF[datastruct.MCluster.Z_v].__len__(), "The created cluster have problem"


    start_streaming(osgm_model, dataset.X, dataset.Y, file_name=osgm_model.config.dataset_name, save_interval=save_point_interval)
    osgm_model.close()



def multi_label_classification(file_name,D_INIT,MAX_MC_FOR_LABEL, MC_MATURE_THRESHOLD, LABELED_INSTANCE_PERCENTAGE, MINIMUM_DIMENSION_DOC, ALPHA, BETA, LAMDA, decay, applyICF, applyCWW, stc, feature_threshold,local_cluster_vocabulary_beta,merge_old_cluster,mclus_beta_multi,include_new_vocabulary,  save_point_interval = 1000, load_file = None):
    X, y, label_count = ut.read_ARFF(file_name)

    total_instances = X._shape[0]
    dataset = Dataset(total_instances=total_instances, minimum_dimension_for_doc=MINIMUM_DIMENSION_DOC)

    total_instances_after_filter = 0
    for idxx in range(0, total_instances):
        total_instances_after_filter = dataset.add_instance( X.rows[idxx], y.rows[idxx])

    total_instances = total_instances_after_filter
    X = 0
    y = 0

    if load_file != None:
        print("loading: ", load_file)
        fileObj = open(load_file, 'rb')
        osgm_model = pickle.load(fileObj)
        fileObj.close()
        output_string = ut.evalute_model(osgm_model, dataset.Y)
        osgm_model.config.output_file.write(output_string)
        start_streaming(osgm_model, X=dataset.X, total_instances=total_instances, y=dataset.Y, file_name=file_name, save_interval=save_point_interval)
        print("loaded from file")
        exit(-1)

    print("Processing LDA")
    # missing_attribute = False
    # for idx in range(0, total_instances):
    #     sample = X.rows[idx]
    #     if len(sample) < 1:
    #         missing_attribute = True
    #         print('There is no word in the document {}'.format(idx))
    #     # ------------------------------------------------------------------
    # if missing_attribute:
    #     print("exiting")
    #     exit(0)


    # check if selected samples contain samples of all the labels
    contain_all_label, initial_cardinality, label_cooccurence_prob, label_indexes, sample_indexes=ut.check_all_labels_existence(D_INIT,label_count,dataset.Y)


    if contain_all_label == False:
        print("More indexes has been choosen to collected samples of all labels")

    model_configuration = mconf.model_config(initial_sample_index=sample_indexes, ALPHA=ALPHA, BETA=BETA, LAMDA=LAMDA, applyDecay=decay, applyICF = applyICF, applyCWW = applyCWW, single_term_clustering = stc, FR_THRESHOLD=feature_threshold,
                       local_vocabulary_beta = local_cluster_vocabulary_beta, merge_old_cluster = merge_old_cluster,
                 mclus_beta_multi = mclus_beta_multi, new_vocabulary_for_beta = include_new_vocabulary, mature_threshold=MC_MATURE_THRESHOLD,
                                             dataset_name=ntpath.basename(file_name), label_count=label_count, max_cluster_for_each_label=MAX_MC_FOR_LABEL, labelled_data_percentage=LABELED_INSTANCE_PERCENTAGE, total_instances=total_instances, initial_cardinality = initial_cardinality,   label_coccurence_prob = label_cooccurence_prob)
    osgm_model = OSGM.Model(configuration=model_configuration)



    tfidfmodel, dictionary = ut.get_tfidf_model(dataset.X[0:D_INIT-1])


    # label_indexes = ut.sample_indexes_group_by_labels(D_INIT,y)   #split the D_INIT sample group by label, so that a document may be part of multiple group, as per its labels

    for label_id, sample_index_list in label_indexes.items():     # traverse documents for each label
        class_instances = []        # raw instances of that particular label
        class_instances_document = []  # instances of that particular label in form of Document object
        for idx in sample_index_list:
            sample = dataset.X[idx]
            doc_instance = Document(sample, idx, osgm_model)
            class_instances_document.append(doc_instance)
            class_instances.append(sample)

        topic_class_doc_map = ut.apply_LDA(tfidfmodel,dictionary,MAX_MC_FOR_LABEL,class_instances)   # split the document of same label into multiple topics using LDA
        for topic_id, list_of_doc_index  in topic_class_doc_map.items():   # after identifying the topics, use each topic to construct one micro_cluster
            docs_topic_class = [class_instances_document[d_idx] for d_idx in list_of_doc_index]
            MC_instance = MCluster(osgm_model,MC_MATURE_THRESHOLD, label_id)
            for doc in docs_topic_class:
                MC_instance.add_document(doc)

            osgm_model.add_cluster(MC_instance)
            assert MC_instance.CF[datastruct.MCluster.Z_w].__len__() == MC_instance.CF[datastruct.MCluster.Z_v].__len__(), "The created cluster have problem"


    # apply feature decay function
    a= 10

    # print("need correction in document label id input in createNewCluster()")
    save_model_into_file(filename=file_name, timestamp=osgm_model.currentTimestamp, g_model=osgm_model)
    output_string = ut.evalute_model(osgm_model, dataset.Y)

    start_streaming(osgm_model, dataset.X, dataset.Y, file_name=file_name, save_interval=save_point_interval)
    osgm_model.close()


def start_streaming(osgm_model, X, y, file_name = "", save_interval=1000):
    output_string = ut.evalute_model(osgm_model,  y)
    osgm_model.config.output_file.write(output_string[0]+"\n")
    osgm_model.config.mlog(output_string[0])

    # start streaming process
    for idx in range(osgm_model.last_index_traversed,osgm_model.config.total_instances):
        if idx in osgm_model.config.initial_sample_indexes:
            continue

        if(osgm_model.currentTimestamp%save_interval == 0):
            osgm_model.last_index_traversed = idx  # if model is saved then it will start from same index if re-run
            # save_model_into_file(filename=file_name, timestamp=osgm_model.currentTimestamp, g_model=osgm_model)
            output_string = ut.evalute_model(osgm_model,  y)
            osgm_model.config.output_file.write(output_string[1]+"\n")
            osgm_model.config.mlog(output_string[1])

        # labels = None
        # if idx in labeled_sample_indexes:
        #     labels = y.rows[idx]
        labels = y[idx]
        sample = X[idx]
        doc_instance = Document(sample, idx, osgm_model)
        osgm_model.processDocument(doc_instance, labels)

    output_string = ut.evalute_model(osgm_model, y)
    osgm_model.config.output_file.write(output_string[1]+"\n")
    osgm_model.config.output_file.close()
    osgm_model.config.mlog(output_string[1])

    osgm_model.config.mlog("need to deal with this situation - No matched Cluster \t total terms - \t- total clusters\n {0}".format(osgm_model.skipped_documents_notify_text))
    osgm_model.config.mlog("Skipped-documents-{0}".format(osgm_model.number_of_skipped_documents))



def save_model_into_file(filename, timestamp, g_model):
    i = 0
    # obj_file = ntpath.basename(filename) + "classifier-{0}".format(timestamp) + "-.osgm"
    # print(obj_file)
    # file_handler = open(obj_file, 'wb')
    # pickle.dump(g_model, file_handler)  # if you want to uncomment this, you should not call set_output_file function of OSGM model , because osgm opens a file
    # file_handler.close()

# def temp_function_for_denominator()


