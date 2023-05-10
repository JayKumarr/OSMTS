import json
from datastruct import MCluster as mc
# Input: This class takes a json object of document
# Output: update global Map of word_wid_map

class Document:
    """
        * Input:
            ws: {list} list of word in the document, each element is a word, a word may exist more than twice
            doc_id: {int} assigned document ID, can be used to assign Micro-cluster
            word_wid_map:  {dict}   key<word>  value<word_id> , the parent will manage this dictionary, so that it can use for multiple tasks
            wid_word_map:  {dict} key<word_id> value<word>  , inverse of word_wid_map,  used for searching purpose
            word_counter: {list}   contains only one element, that first element helps to generate NEWID for words, NEEDS to be initialized with word_counter[0] = -1 
    """
    def __init__(self,ws,docid, osgm_model):
        self.doc_id = docid
        self.CF = {}
        self.CF[mc.Z_w] = {}   # maintaining wordId and the occurence
        self.CF[mc.Z_v] = {}
        self.CF[mc.Z_l] = 0
        if len(ws) < 1:
            # print('There is no word in the document {}'.format(doc_id))
            raise Exception('InstanceError', 'There is no word in the document {}'.format(self.doc_id))


        for w in ws:
            self.CF[mc.Z_l] = self.CF[mc.Z_l]+1
            NEWID = osgm_model.word_counter[0]+ 1
            wid = osgm_model.words.get_wid_of_word(w,NEWID)   #if the key exist in word_wid_map then it will return wid OTHERWISE it will return default value
            if wid == NEWID:  # if a word occuring first time globaly then Add it with new ID
                assert wid not in self.CF[mc.Z_w], "there is a problem while deleting the old term in this cluster, this self.CF[mc.Z_w] should not have this term"
                osgm_model.word_counter[0] =NEWID
                self.CF[mc.Z_w][NEWID] = 1

            else:   # if any word is already came before than update local document widFreq
                tf = 0
                defaultTF = 0
                tf = self.CF[mc.Z_w].get(wid,defaultTF)
                if tf == defaultTF:  # if this word is occuring first time in this document
                    self.CF[mc.Z_w][wid] = 1
                else:
                    tf = tf+1
                    self.CF[mc.Z_w][wid] = tf

        #calculate word to word score
        for w, wFreq in self.CF[mc.Z_w].items():
            self.CF[mc.Z_v][w]={}  # adding wid into self.widToWidFreq
            for w2,w2Freq in self.CF[mc.Z_w].items():
                if w!=w2:
                    total = wFreq+w2Freq
                    score = wFreq/total
                    self.CF[mc.Z_v][w][w2] = score