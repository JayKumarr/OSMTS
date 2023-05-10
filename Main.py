from ssclassifier.classifier import multi_label_classification
import argparse
import copy
from multiprocessing import Pool
from ssclassifier import  ccutils as utils
# from my_logger import plog as log
# from my_logger import define_logger

def argument_parser():
    ap = argparse.ArgumentParser(description='-d \"News.arff\" -icf -cww -decay 0.000006 -alpha 0.002 -beta 0.0004')
    ap.add_argument("-d", "--dataset_dir", required=True,  help="the dataset file path. file format should be json on each line of file { Id: '' , clusterNo: '', textCleaned:'' }")
    # ap.add_argument("-o", "--output_dir", required=False,  help="[the directory where the output will be saved] ")
    # ap.add_argument("-e", "--evalaute_dir", default=False, required=False,  help="the directory where the evaluation results will be saved. if given [then the prediction results will be saved in this directory] ")
    # ap.add_argument("-gs", "--generate_summary", default=False, action='store_true', required=False,
    #                 help="True/False [if you want to generate summary of generated results]")
    ap.add_argument("-icf", "--icf", default=True, action='store_true', required=False, help="True/False [if you want to apply ICF weight]")
    ap.add_argument("-cww", "--cww", default=True, action='store_true', required=False,  help="True/False [if you want to apply CWW weight]")
    ap.add_argument("-alpha", "--alpha", default=False, type=float ,required=False, help="customized value of alpha")
    ap.add_argument("-beta", "--beta",  default=False, required=False, help="customized value of beta")
    ap.add_argument("-decay", "--decay", default=False, type=float, required=False, help="Default value is [False]. Value is set as lambda in for exponential decay i.e. 0.000006")
    ap.add_argument("-ft", "--feature_threshold", default=0, type=int, required=False, help="triangular weight feature threshold")
    ap.add_argument("-dinit", "--d_init", default=960, type=int, required=False, help="initial number of instances for training")
    # ap.add_argument("-sa", "--start_alpha", default=0, required=False, help="start alpha when we are doing grid search")
    # ap.add_argument("-sb", "--start_beta", default=0, required=False, help="start beta when we are doing grid search")
    ap.add_argument("-asl", "--alpha_start_limit", default=0.0001, type=float, required=False, help="start alpha from grid search array ")
    ap.add_argument("-bsl", "--beta_start_limit", default=0.0001, type=float, required=False, help="start beta from grid search array")
    ap.add_argument("-all", "--alpha_last_limit", default=0.09, required=False,help="end alpha from grid search array")
    ap.add_argument("-bll", "--beta_last_limit", default=0.4, required=False, help="end beta from grid search array")
    ap.add_argument("-sll", "--sigma_last_limit", type=int ,default=40, required=False, help="end Sigma from grid search array")
    # ap.add_argument("-log", "--log_file", default=False, required=False, help="log file flag ")
    ap.add_argument("-stc", "--single_term_consider", default=False, action='store_true', required=False, help="True/False [Atleast one term should be matched before calculating cluster probability]")
    # ap.add_argument("-lastindex", "--last_index_for_grid_search", default=0, type=int, required=False,
    #                 help="start index while doing grid search")
    ap.add_argument("-lcb", "--local_cluster_beta", default=False, action='store_true', required=False, help="True -[beta will be calculated according to cluster vocabulary] ")
    ap.add_argument("-mclus", "--merge_old_cluster", default=False, action='store_true', required=False,
                    help="True/False [if you want to merge deleted cluster]")
    # ap.add_argument("-mclusbm", "--mclus_beta_multi", default=1, type=int, required=False, help="multiply the beta of merging cluster")
    ap.add_argument("-invb", "--include_new_vocabulary", default=False, action='store_true', required=False,
                    help="new vocabulary add before calculating beta")

    ap.add_argument("-mid", "--minimum_instance_dimension", default=0, type=int, required=False, help="minimum instance dimension / filter objects having dimension length less than")
    ap.add_argument("-threads", "--threads", default=0, type=int, required=False, help="parallel execution")
    ap.add_argument("-pof", "--previous_output_file", required=False, default=None, help="Previous output file (if any), then it will skip the already executed iteration")

    args = vars(ap.parse_args())
    for k, v in args.items():
        print(k, " -> ",v)
    return args

def grid_search(args):
    v_alpha=args['alpha']
    v_beta = args['beta']
    _values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003,
               0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
               0.09, 0.1, 0.2, 0.3, 0.4, 0.5]

    alpha_list = []
    if v_alpha != False:
        alpha_list.append(v_alpha)
    else:
        si = 0
        ei = _values.__len__() - 1
        if args['alpha_start_limit'] != -1:
            si = _values.index(args['alpha_start_limit'])
        if args['alpha_last_limit'] != -1:
            ei = _values.index(args['alpha_last_limit']) + 1
        alpha_list = _values[si:ei]

    beta_list = []
    if v_beta != False:
        beta_list.append(v_beta)
    else:
        beta_si = 0
        beta_ei = _values.__len__() - 1
        if args['beta_start_limit'] != -1:
            beta_si = _values.index(args['beta_start_limit'])
        if args['beta_last_limit'] != -1:
            beta_ei = _values.index(args['beta_last_limit'])

        beta_list = _values[beta_si:beta_ei + 1]

    alphas = []
    betas = []
    for a in alpha_list:
        for b in beta_list:
            alphas.append(a)
            betas.append(b)
    return alphas, betas

def grid_search_feature_threshold(args):
    feature_th = args['feature_threshold']
    fts = [5,10,15,20,25,30,35,40]
    if feature_th == -1:
        sll=args['sigma_last_limit']
        last_index = fts.index(sll)
        return fts[0:last_index]
    else:
        ret = [int(feature_th)]
        return ret


#-------------- For multi-processing/multithreading----------------#
class P_Bean:  # when we don't want a file to be output
    def __init__(self, iteration,total_iteration,cloned_dataset,redefined_model_config,labelID_LDA_clusters,saving_points_interval):
        self.redefined_model_config = redefined_model_config
        self.iteration = iteration
        self.total_iteration = total_iteration
        self.cloned_dataset = cloned_dataset
        self.labelID_LDA_clusters = labelID_LDA_clusters
        self.saving_points_interval = saving_points_interval

def function_for_thread(arg_bean):
    arg_bean.redefined_model_config.mlog("iteration- {0}/{1}".format(arg_bean.total_iteration, arg_bean.iteration))
    cloned_dataset = copy.deepcopy(arg_bean.cloned_dataset)
    process_m_l_classification(cloned_dataset, arg_bean.redefined_model_config, arg_bean.labelID_LDA_clusters, arg_bean.saving_points_interval)
#----------------------------------------------------------

if __name__ == '__main__':
    import os

    # define_logger()
    try:
        os.makedirs("results/")
    except OSError as e:
        print('results/ directory already exist')

    args = argument_parser()

    file_name = args['dataset_dir']
    # file_name= "moa experimentaion/39Yahoo_Social-d52350.arff"
    # file_name = "moa experimentaion/26Yahoo_Arts-d23150.arff"
    # file_name = "moa experimentaion/20NG-d1006.arff"


    D_INIT = args['d_init']
    MAX_MC_FOR_LABEL = 3 # number of maximum micro-cluster a class can have
    MC_MATURE_THRESHOLD = 2
    LABELED_INSTANCE_PERCENTAGE = 20
    MINIMUM_DEMINSION_FOR_DOC = args['minimum_instance_dimension']
    LAMDA = 0.00004
    applyICF = args['icf']
    applyCWW = args['cww']
    stc = args['single_term_consider']
    feature_threshold_list = grid_search_feature_threshold(args)
    decay = True
    thread_size = args['threads']
    skipped_iteration = list()
    if args['previous_output_file'] != None:
        skipped_iteration = utils.skip_already_executed(args['previous_output_file'])

    alphas, betas = grid_search(args)

    from ssclassifier.classifier import init_m_l_classification, process_m_l_classification
    dataset, model_configuration, labelID_LDA_clusters = init_m_l_classification(file_name,D_INIT,MAX_MC_FOR_LABEL, MC_MATURE_THRESHOLD, LABELED_INSTANCE_PERCENTAGE, MINIMUM_DIMENSION_DOC=MINIMUM_DEMINSION_FOR_DOC)

    total_iteration = alphas.__len__()*feature_threshold_list.__len__()
    iteration = 0
    arguments_for_thread = list()
    for ALPHA, BETA in zip(alphas, betas):
        for fthreshold in feature_threshold_list:
            #--------------without thread START------
            # iteration = iteration + 1
            # print("iteration- {0}/{1}".format(total_iteration, iteration))
            # local_cluster_vocabulary_beta = True
            # merge_old_cluster = True
            # mclus_beta_multi = 1
            # include_new_vocabulary = True
            #
            # stored_model_file = None
            # starting_point = None
            #
            # saving_points_interval = 1000
            #
            # # starting_point = 2700
            # # stored_model_file = ntpath.basename(file_name)+"classifier-{0}-.osgm".format(starting_point)
            #
            # # multi_label_classification(file_name,D_INIT,MAX_MC_FOR_LABEL, MC_MATURE_THRESHOLD, LABELED_INSTANCE_PERCENTAGE, MINIMUM_DEMINSION_FOR_DOC, ALPHA, BETA, LAMDA, decay, applyICF, applyCWW, stc, feature_threshold,local_cluster_vocabulary_beta,merge_old_cluster,mclus_beta_multi,include_new_vocabulary,  save_point_interval = saving_points_interval, load_file = None)
            #
            # redefined_model_config = copy.deepcopy(model_configuration)
            # cloned_dataset = copy.deepcopy(dataset)
            #
            # redefined_model_config.redefine(ALPHA=ALPHA, BETA=BETA, LAMDA=LAMDA, decay=decay, applyICF=applyICF,
            #                                 applyCWW=applyCWW, single_term_clustering=stc, feature_threshold=fthreshold,
            #                                 local_cluster_vocabulary_beta=local_cluster_vocabulary_beta,
            #                                 merge_old_cluster=merge_old_cluster, mclus_beta_multi=mclus_beta_multi,
            #                                 include_new_vocabulary=include_new_vocabulary)
            # process_m_l_classification(cloned_dataset, redefined_model_config, labelID_LDA_clusters,
            #                            saving_points_interval)
            # redefined_model_config = None
            #--------------------------without thread END-------

            iteration = iteration + 1
            if iteration in skipped_iteration:
                continue
            #print("iteration- {0}/{1}".format(total_iteration, iteration))
            local_cluster_vocabulary_beta = True
            merge_old_cluster = True
            mclus_beta_multi = 1
            include_new_vocabulary = True
            stored_model_file = None
            starting_point = None
            saving_points_interval = 1000
            # starting_point = 2700
            # stored_model_file = ntpath.basename(file_name)+"classifier-{0}-.osgm".format(starting_point)
            redefined_model_config = copy.deepcopy(model_configuration)


            redefined_model_config.redefine(ALPHA=ALPHA, BETA=BETA, LAMDA=LAMDA, decay=decay, applyICF=applyICF,
                                            applyCWW=applyCWW, single_term_clustering=stc, feature_threshold=fthreshold,
                                            local_cluster_vocabulary_beta=local_cluster_vocabulary_beta,
                                            merge_old_cluster=merge_old_cluster, mclus_beta_multi=mclus_beta_multi,
                                            include_new_vocabulary=include_new_vocabulary)
            bean = P_Bean(iteration=iteration, total_iteration=total_iteration, cloned_dataset=dataset, redefined_model_config=redefined_model_config, labelID_LDA_clusters=labelID_LDA_clusters, saving_points_interval=saving_points_interval)
            arguments_for_thread.append(bean)
            redefined_model_config = None
    print("-> bean list is ready, multiprocess started {0}".format(arguments_for_thread.__len__()))
    print("Total_iteration-{0} \t Skipped_iteration-{1} \t remaining-{2} ".format(total_iteration, skipped_iteration.__len__(), (total_iteration - skipped_iteration.__len__()) ))
    with Pool(thread_size) as p:
        p.map(function_for_thread, arguments_for_thread)












