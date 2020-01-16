import pymongo
import json
import collections
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

class MDBSearch:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client['camp']
        self.reverse_dic = collections.defaultdict(list)
        self.data = []
        self.raw_data = {}
        self.collection = self.db['reverse']
        self.score_col = self.db['score']
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def create_dic(self, f='pets.json'):
        with open(f, 'r') as f:
            self.raw_data = json.load(f)

    def create(self, img_list, caption_dict, type=1):
        corpus = []
        for idx in range(len(img_list)):
            img_path = img_list[idx]
            real_img_path=img_path.split("/")[-1]
            sentence = caption_dict[real_img_path]
            if type == 2:
                corpus.append(sentence)
            for word in sentence.split(' '):
                self.reverse_dic[word] += [idx] # (word, [fileID_1, fileID_2, ...])

        for item in self.reverse_dic.items():
            self.data.append({'key': item[0], 'imgIds': item[1]})

        if type == 2: #type 2 represent tf-idf score stored
            self.cal_TFIDF(corpus)


    def cal_TFIDF(self):
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))
        #word = self.vectorizer.get_feature_names()
        weights = tfidf.toarray()
        for i, weight in enumerate(weights):
            self.score_col.insert({'idx': i, 'weight': [weight.tolist()]})

    def query_TFIDF(self, query):
        test_vec = self.vectorizer.transform([query]).toarray()
        tfidf = self.transformer.fit_transform(test_vec)
        return tfidf.todense().tolist()

    def create_table(self, img_list, caption_dict, type=1):
        #self.create_dic() # Notice: If img_list and caption_dict provided, self.raw_data is unnecessary
        self.create(img_list, caption_dict, type=1)
        self.collection.insert_many(self.data) #i nsert pair [{'key':'word1', 'imgIds':[0,1,3]}, {...}, ..] into collection of the database

    def debug(self):
        self.collection.find_one()

    def clear(self):
        self.collection.drop()
        self.score_col.drop()

    def return_top_k(self, query, tmp_res, k, thres):
        res = {}
        qscore = self.query_TFIDF(query)
        for i in tmp_res:
            tmp = []
            for i in self.score_col.find({'idx':i}):
                tmp = i['weight']
            if len(tmp) == 0:
                continue
            score = euclidean_distances(qscore, tmp)
            if score > thres:
                continue
            res[i] = score
        results = sorted(res.items(), key=lambda item: item[1])
        return results[:k]

    def search(self, query):
        results = set()
        flag = False
        for word in query.split(' '):
            myQuery = {'key': word}
            tmp_cursor = self.collection.find(myQuery, {"_id":0, "imgIds": 1})
            tmp_res = []
            for tmp_dict in tmp_cursor:
                tmp_res.extend(tmp_dict['imgIds'])
            if not flag:
                flag = True
                results = set(tmp_res)
            else:
                results = results & set(tmp_res)
        return results

    def insert(self, caption, idx, type=1):
        for word in caption.split(' ')[-1]:
            myQuery = {'key': word}
            for tmp_dict in self.collection.find(myQuery):
                attr = tmp_dict['imgIds']
                attr += [idx]
                self.collection.update_one(myQuery, {'$set': {'imgIds': attr}})
        if type == 2:
            qscore = self.query_TFIDF(caption)
            self.score_col.insert({'idx': idx, 'weight': qscore})

    def fuzzy_search(self, query):
        results = []
        for word in query.split(' '):
            myQuery = {'key': {'$regex': word+'*'}}
            tmp_cursor = self.collection.find(myQuery, {"_id":0, "imgIds": 1})
            for tmp_dict in tmp_cursor:
                results.extend(tmp_dict['imgIds'])
        return set(results) # at pic_idx_list form

if __name__ == '__main__':
    search = MDBSearch() #init and start
    img_list = [] #Replace by img_path_list
    img2cap_dic = {} #Replace by img_path2caption_dict
    search.create_table(img_list, img2cap_dic, type=2) #type=1 no weights, type=2 weighted

    f_res = search.fuzzy_search('lala') # In supportive of Regex search/prefix search
    search.search('dog is eating') # Output is a set!!! Precisely search for each word

    print(search.return_top_k('lala', f_res, k=20, thres=0.8)) #same keyword







