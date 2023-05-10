import gensim
from gensim import models
import numpy as np
import math
import statistics as stat

def intersection(listA, listB):
    return list(set(listA) & set(listB))


def union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def read_ARFF(filepath, label_location="end", arff_file_is_sparse=False):
    from skmultilearn.dataset import load_from_arff
    label_count = read_label_count(filepath)
    data, labels  = load_from_arff(filepath,label_count=abs(label_count),label_location=label_location,load_sparse=arff_file_is_sparse)
    return data, labels, abs(label_count)



def read_label_count(filepath):
    file_reader = open(filepath, 'r')
    relation_line = ""
    while True:
        line = file_reader.readline()
        if line.strip().lower().startswith("@relation"):
            relation_line = line.strip().lower()
            break;
        elif (line.strip().lower().startswith("@attribute")):
            file_reader.close()
            raise Exception('FileFormatError', 'The first line should be starts with @relation where number of labels should be defined')

    if (relation_line.startswith("@relation")):
        space = relation_line.index(" ", relation_line.index("-c") + 1)
        end = 0
        try:
            end = relation_line.index("'", space)
        except:
            end = relation_line.index('"', space)
        label_count = int(relation_line[space + 1:end])
        return label_count
    else:
        file_reader.close()
        raise Exception('FileFormatError','The first line should be starts with @relation where number of labels should be defined')

    file_reader.close()


"""
* input:
       samples {int}   D_INIT
       label_count {int}  total number of labels in dataset
       y   {lil_matrix}   2D matrix of labels, 2nd argument returned by read_ARFF()
* output: 
        returns {boolean} either the number of selected samples have all the labels for training
"""
def check_all_labels_existence(samples,label_count,y):
    labels = set()
    for last_idx in range(y.__len__()):
        label_list = y[last_idx]
        for l in label_list:
            labels.add(l)
        if labels.__len__() == label_count:
            break

    label_instances = {}   #label:[idx1,idx2...]
    combined_labels = []
    label_cooccurence = np.zeros((label_count, label_count), dtype=int)
    series = True
    sample_indexes = []

    if last_idx > (samples-1):
        series = False
        avg_sample_per_class = int( math.ceil(samples/label_count) )
        for idx in range(y.__len__()):
            label_list = y[idx]
            if (intersection(label_list, labels).__len__() < 1):
                continue
            for l in label_list:
                if l not in label_instances:
                    label_instances[l] = [idx]
                else:
                    label_instances[l].append(idx)
                if label_instances[l].__len__() >= avg_sample_per_class:
                    if l in labels:
                        labels.remove(l)
                combined_labels.append(l)
                sample_indexes.append(idx)
                #  label_cooccurence counts
                for l_a in label_list:
                    for l_b in label_list:
                        if l_a != l_b:
                            label_cooccurence[l_a][l_b] = (label_cooccurence[l_a][l_b] + 1)
            if labels.__len__() == 0:
                break
    else:
        sample_indexes = [index for index in range(0, samples)]
        labels = y[0:samples-1]
        for idx, label_list in enumerate(labels):
            combined_labels.extend(label_list)
            for l_a in label_list:
                if l_a not in label_instances:
                    label_instances[l_a] = [idx]
                else:
                    label_instances[l_a].append(idx)
                for l_b in label_list:
                    if l_a != l_b:
                        label_cooccurence[l_a][l_b] = (label_cooccurence[l_a][l_b]+1)

    print("before: ", len(combined_labels))
    unique_labels = set(combined_labels)
    print("after: ", len(unique_labels))
    import collections
    labels_counter = collections.Counter(combined_labels)   # counts the instances for each label

    label_cooccurence_probability = np.zeros((label_count, label_count))
    for l_a in unique_labels:
        occurence_of_la = labels_counter[l_a]
        for l_b in unique_labels:
            if l_a != l_b:
                la_lb = label_cooccurence[l_a][l_b] / occurence_of_la
                label_cooccurence_probability[l_a][l_b] = la_lb
    # True/False,  initial_cardinality, label_cooccurence prob. , indexes of each label <labelID:[idx_d1, idx_d2, idx_d3]>, indexes for making initial clusters[idx1, idx2]
    return series, (combined_labels.__len__()/samples), label_cooccurence_probability, label_instances, sample_indexes

"""
* input:
    samples {int}   D_INIT
    y   {lil_matrix}   2D matrix of labels, 2nd argument returned by read_ARFF()
* output:
    return dict, key <label_ID> and values <[array of sample indexes belong to particular label ]>
"""
def sample_indexes_group_by_labels(samples,y):
    label_index_dict = {}
    for sample_index  in range(0,samples):
        labels = y.rows[sample_index]
        for l in labels:
            if l not in label_index_dict.keys():
                label_index_dict[l] = []
            label_index_dict[l].append(sample_index)
    return label_index_dict


def convert_integer_into_string(instances):
    texts = [0 for a in range(0,len(instances))]
    for index,sample in enumerate(instances):
        temp = []
        for integer_word in sample:
            temp.append(str(integer_word))
        texts[index] = temp
    return texts


"""
    instances: {list} [ doc1[tuple(word_id1, occurence),...], doc2[], ...  ] | index must start from zero 
    topics:    {int}  number of topics to be discovered
"""
def apply_LDA_model_on_tfidf(all_instances, topics):
    from ssclassifier.ccutils import convert_integer_into_string
    texts = convert_integer_into_string(all_instances)  # this function preserve indexes of given list instances

    dictionary = gensim.corpora.Dictionary(texts)
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=topics, id2word=dictionary, passes=4, workers=4)
    topic_doc_map = {}
    for i,x in enumerate(texts):
        sorted_by_topic_score = sorted(lda_model_tfidf[corpus_tfidf[i]], key=lambda tup: -1 * tup[1])
        tuple_topic_id_score=sorted_by_topic_score[0]
        cluster_index = tuple_topic_id_score[0]
        if cluster_index not in topic_doc_map:
            topic_doc_map[cluster_index] = []
        # topic_doc_map[cluster_index].append( (i,all_instances[i]) )
        topic_doc_map[cluster_index].append(i)
    return topic_doc_map


"""
    instances: {list} [ doc1[tuple(word_id1, occurence),...], doc2[], ...  ] | index must start from zero
"""
def get_tfidf_model(all_instances):
    texts = convert_integer_into_string(all_instances)  # this function preserve indexes of given list instances
    dictionary = gensim.corpora.Dictionary(texts)
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
    tfidf = models.TfidfModel(bow_corpus)

    return tfidf,dictionary

""" 
    model: {models.TfidfModel} or {bag of words model}
    topics:    {int}  number of topics to be discovered
    dictionary:
    texts:   class related instances
"""
def apply_LDA(tfidfmodel, dictionary, topics, class_instances):
    texts = convert_integer_into_string(class_instances)
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
    corpus_tfidf = tfidfmodel[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=topics, id2word=dictionary, passes=4, workers=4)
    topic_doc_map = {}
    for i,x in enumerate(texts):
        sorted_by_topic_score = sorted(lda_model_tfidf[corpus_tfidf[i]], key=lambda tup: -1 * tup[1])
        tuple_topic_id_score=sorted_by_topic_score[0]
        cluster_index = tuple_topic_id_score[0]
        if cluster_index not in topic_doc_map:
            topic_doc_map[cluster_index] = []
        # topic_doc_map[cluster_index].append( (i,x) )
        topic_doc_map[cluster_index].append(i)
    return topic_doc_map

def generate_random_index(start, end, percentage, excluded_indexes=[]):
    from random import choice
    number_of_index_to_be_generated = int( ((end-start)/100)*percentage )
    random_numbers_set = set()
    while random_numbers_set.__len__() < number_of_index_to_be_generated:
        random_number = choice([i for i in range(start, end) if i not in excluded_indexes])
        random_numbers_set.add(random_number)

    return random_numbers_set

"""
data_dict:  dict  {<clus_id, probability>, ... }   
"""
def export_scatter_plot(file_path, data_dict, idx = 0):
    stophere = 10
    x = range(0, data_dict.__len__())
    import numpy as np
    import matplotlib.pyplot as plt


    area = np.pi * 3

    # Plot
    sort_dict = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots()
    y = []
    n = []
    for tp in sort_dict:
        y.append(tp[1])
        n.append(tp[0])

    if x.__len__() > 12:
        x = x[0:11]
        y = y[0:11]

    plt.scatter(x, y, s=area, alpha=0.5)

    for i, txt in enumerate(n):
        if i == x.__len__():
            break
        ax.annotate(txt, (x[i], y[i]))

    plt.title("Probability "+str(idx))
    plt.xlabel('x')
    plt.ylabel('y')

    file_path = file_path+".jpg"
    fig.savefig(file_path, bbox_inches='tight', dpi=150)
    plt.close()


"""
data_dict:  dict  {<clus_id, probability>, ... }   
"""
def export_scatter_plot(file_path, x, y, annotations, idx = 0, values_above_mean = 0, predicted_labels=""):
    stophere = 10
    import numpy as np
    import matplotlib.pyplot as plt

    assert (x.__len__() == y.__len__()) and (y.__len__() == annotations.__len__()), "The size of X, Y, annotation should be equal"

    area = np.pi * 3

    # Plot
    fig, ax = plt.subplots()

    plt.scatter(x, y, s=area, alpha=0.5)

    for i, txt in enumerate(annotations):
        ax.annotate(txt, (x[i], y[i]))

    plt.title("Label "+str(idx)+"    VaM-{0}".format(values_above_mean)+"   PL:"+predicted_labels)
    plt.xlabel('x')
    plt.ylabel('y')

    file_path = file_path+".jpg"
    fig.savefig(file_path, bbox_inches='tight', dpi=150)
    plt.close()


# Evaluate the model
def evalute_model(osgm_model, original_label):
    label_count=osgm_model.config.label_cooccurence_probability.__len__()
    from datastruct import MCluster
    from ssclassifier import evaluate
    docid_labels = {}
    predicted_labels = []
    original_labels = []
    for clus_id, cluster in osgm_model.active_clusters.items():
        doc_id_list = cluster.CF[MCluster.Z_n]
        for doc_id in doc_id_list:
            if doc_id not in docid_labels:
                docid_labels[doc_id] = [0 for a in range(label_count)]
            docid_labels[doc_id][cluster.CF[MCluster.Z_c]] = 1


    for doc_id, p_label in docid_labels.items():
        original = original_label[doc_id]
        temp = [0 for a in range(label_count)]
        for lab in original:
            temp[lab] = 1
        predicted_labels.append(p_label)
        original_labels.append(temp)

    o_np = np.array(original_labels)
    p_np = np.array(predicted_labels)

    return evaluate.print_evaluation_score(o_np, p_np)


def normalize_dict_by_values(dictionary_, values_):
    factor = max(values_)
    output_dict = {}
    for k, tup_lis in dictionary_.items():
        output_dict[k] = []
        for tup in tup_lis:
            t = (tup[0] , tup[1]/factor)
            output_dict[k] .append(t)

    return output_dict

def count_keys_by_same_value(dictionary_):
    output = {}
    for k, v in dictionary_.items():
        if v not in output:
            output[v] = []
        output[v].append(k)
    return output

def count_values_above_mean(list_):
    v_mean =stat.mean(list_)
    count = 0
    for val  in list_:
        if val > v_mean:
            count+=1
    return count

def prob_sum(key, all_keys , dict_):
    sum_t = 0.0
    for x in all_keys:
        sum_t+=dict_[key][x]
    return sum_t

def sum_of_tup_values(list_):
    sum_t = 0.0
    for t in list_:
        sum_t += t[1]
    return sum_t

def evaluate_moa_prediction_file(source_arff_file, moa_prediction_file):
    X, y, label_count = read_ARFF(source_arff_file)
    file_reader = open(moa_prediction_file, 'r')
    line = file_reader.readline()
    index_counter = 0
    true_labels = []
    predicted_labels = []
    while line != '':
        true_vector = [0 for a in range(label_count)]
        original_labels = y.rows[index_counter]
        for lab_ind in original_labels:
            true_vector[lab_ind] = 1

        pred_vector = [1 for a in range(label_count)]
        string_tokes = line.split(",")
        left_string = string_tokes[0]
        right_string = int(float(string_tokes[1].strip()))
        splited=left_string.replace("Out","").split(" ")
        local_idx = 0
        while local_idx+2 < splited.__len__():
            if splited[local_idx] == '':
                label = int(splited[local_idx+1].replace(":",""))
                label_0 = float(splited[local_idx+2])
                label_1 = float(splited[local_idx + 3])

                if right_string == 0:
                    if label_0== 1.0:
                        pred_vector[label] = 0
                elif right_string == 1:
                    if label_0 == 1.0:
                        pred_vector[label] = 0

            local_idx+=1

        true_labels.append(true_vector)
        predicted_labels.append(pred_vector)

        line = file_reader.readline()
        index_counter+=1

    from ssclassifier import evaluate as ev
    o_np = np.array(true_labels)
    p_np = np.array(predicted_labels)
    headers,common_string = ev.print_evaluation_score(true_labels=o_np, predicted_labels=p_np)
    return headers, common_string

def skip_already_executed(file_name):
    skipped_iteration_list = list()
    with open(file_name,'r') as reader:
        line = reader.readline()

        while line != '':
            ret = line.startswith("iteration- ")
            if ret:
                iteration_number = line[line.index("/") + 1:].strip()
                skipped_iteration_list.append(int(iteration_number))
                line = reader.readline()

            line = reader.readline()
        reader.close()
    return skipped_iteration_list