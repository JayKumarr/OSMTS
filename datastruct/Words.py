

class Words:
    def __init__(self):
        self.word_wid_map = {}  # word, Assigned ID :    updated by Document
        self.wid_word_map = {}  # Assigned ID, word:    updated by Document


    def remove_wid(self, wid):
        word = self.wid_word_map[wid]
        self.word_wid_map.pop(word)
        self.wid_word_map.pop(wid)

    def get_wid_of_word(self, w, default_ID):
        id =  self.word_wid_map.get(w, default_ID)


        if id == default_ID:
            assert id not in self.wid_word_map, "this ID already exist in wid_word_map"
            self.word_wid_map[w] = id    #  defining new ID to word
            self.wid_word_map[id] = w
        return id

