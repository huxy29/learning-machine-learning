import numpy as np

class PLA():

    def read_file(self, fp):
        '''
        Read data from txt file
            read X and add x0=1 at the first column as np.ndarray(m x d+1) , read y as np.ndarray(m x 1)
        '''
        with open(fp, 'r') as f:
            lines = f.readlines()
        m = len(lines)
        line0 = lines[0].split('\t')[0].split(' ')
        d = len(line0)
        matrix = np.ones((m, d+1))
        labels = np.zeros(m)
        i = 0
        for line in lines:
            line = line.strip()
            matrix[i, 1:] = line.split('\t')[0].split(' ')
            labels[i] = line.split('\t')[1]
            i += 1
        return matrix, labels

    def sign(self, x):
        if x<=0:
            return -1
        return 1

    def train(self, train_X, train_Y, random=False, eta=1, silent=True):
        '''
        PLA version1(default):
            visiting examples in the naive cycle using the order of examples in the data set

        PLA version2(random=True):
            visiting examples in fixed, pre-determined random cycles throughout the algorithm

        PLA version3(random=True, eta=0.5):
            visiting examples in fixed, pre-determined random cycles throughout the algorithm,
            while changing the update rule to be w(t+1) = w(t) + eta*y*x
        '''
        m, d = train_X.shape
        w = np.zeros(d)
        num_of_corrects = 0    # when it is equal to m, all examples are corecctly classified
        num_of_updates = 0
        index_of_last_mistake = 0
        index = np.arange(m)
        if random:
            np.random.shuffle(index)    # random visiting
        i = 0
        is_finished = False
        if not silent:
            print('iter\tindex_of_mistake')
        while not is_finished:
            if train_Y[index[i]]==self.sign(np.dot(w, train_X[index[i]])):
                num_of_corrects += 1
            else:
                # update w(t+1)=w(t)+y*x
                w += eta * train_Y[index[i]] * train_X[index[i]]
                num_of_updates += 1
                num_of_corrects = 0
                index_of_last_mistake = index[i]
                if not silent:
                    print('{0}\t{1}'.format(num_of_updates, i))
            if i==m-1:
                i=0
            else:
                i+=1
            if num_of_corrects==m:
                is_finished = True
        self.w = w
        self.num_of_updates = num_of_updates
        self.index_of_last_mistake = index_of_last_mistake

    def calc_err_rate(self, w, train_X, train_Y):
        err_cnt = 0
        m = len(train_X)
        for i in range(m):
            if self.sign(np.dot(w, train_X[i]))!=train_Y[i]:
                err_cnt += 1
        return err_cnt/m

    def train_pocket(self, train_X, train_Y, iteration=50, silent=True):
        m, d = train_X.shape
        w_pocket = np.zeros(d)
        pocket_err_rate = self.calc_err_rate(w_pocket, train_X, train_Y)
        w = np.zeros(d)
        it = 0
        if not silent:
            print('iter\tw_err\tpocket_err')
        while it<iteration:
            index = np.random.randint(0, m)
            if train_Y[index]!=self.sign(np.dot(w, train_X[index])):
                # update w(t+1)=w(t)+y*x
                w += train_Y[index] * train_X[index]
                w_err_rate = self.calc_err_rate(w, train_X, train_Y)
                if w_err_rate < pocket_err_rate:
                    w_pocket = w.copy()
                    pocket_err_rate = self.calc_err_rate(w_pocket, train_X, train_Y)
                if not silent:
                    print('{0}\t{1}\t{2}'.format(it, w_err_rate ,pocket_err_rate))
                it += 1
        self.w_pocket = w_pocket
        self.w = w

    def verify_pocket(self, test_X, test_Y, pocket=True):
        if pocket:
            return self.calc_err_rate(self.w_pocket, test_X, test_Y)
        else:
            return self.calc_err_rate(self.w, test_X, test_Y)
