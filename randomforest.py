class randomforest(classifier):
    def __init__(self, num_trees=5, tree_max_depth=-1,criterion='entropy'):
        self.num_trees = num_trees
        self.tree_max_depth = tree_max_depth
        self.tree_list = None
        self.criterion = criterion

    def subsample(self, sample_df, label_series):
        sampled_row_indexes = []
        bagsize = len(X)
        while len(sampled_row_indexes) < bagsize:
            index = random.randrange(bagsize)
            sampled_row_indexes.append(index)
        bagging_sample_df = sample_df.iloc[sampled_row_indexes].copy()
        bagging_label_series = label_series[sampled_row_indexes].copy()
        return bagging_sample_df, bagging_label_series 
    
    def sample_of_features(self, sample_df, num_features=None):
        #return index of sampled features
        if num_features is None:
            num_features = math.floor(math.sqrt(len(sample_df.columns)))
        feature_sample = random.sample(range(len(sample_df.columns)), int(num_features))
        return feature_sample  
        
    def create_list(self, num_trees, max_depth, criterion):
        treelist = []
        for count in range(0,num_trees):
            onetree = decision_tree(criterion,max_depth)
            treelist.append(onetree)
        return treelist
    
    def fit(self, X, Y, num_features=None): #fit of random forest
        self.tree_list = self.create_list(self.num_trees, self.tree_max_depth, self.criterion) # decision_tree list
        cnt = 0
        for t in self.tree_list:
            subsample_x, subsample_y = self.subsample(X, Y) # Bagging
            feature_list = self.sample_of_features(X, num_features) # Random features
            t.fit(subsample_x, subsample_y,feature_list, self.tree_max_depth)  #fit of decision tree
            cnt += 1
            print('tree {} trained successfully'.format(cnt))
            
    def predict(self, one_sample):
        hypothesis_list = []
        for t in self.tree_list:
            res = t.predict(one_sample)
            hypothesis_list.append(res)
        counts = defaultdict(int)
        for h in hypothesis_list:
            counts[h] += 1
        if "" in counts.keys():
            counts.pop("")
        highest_cnt = 0;
        highest_res = None
        for res in counts.keys():
            if counts[res] >= highest_cnt:
                highest_cnt = counts[res]
                highest_res = res
        return highest_res