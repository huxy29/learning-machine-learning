import numpy as np

class DecisionStump():
    '''Implementation of "decision stump" algorithm

    Also called "positive and negative rays" (which is one-dimensional perceptron)
    for one-dimensional data, the model contains hypotheses of the form:
                        h_s,theta(x) = s*sign(x-theta)

    Attributes:
        gen_data: Generate one-dimensional data set
        calc_err: Calculate error rate
        train_1d: Find the optimal s and optimal theta that make the Ein value lowest
        calc_Eout: Compute Eout value with the formula that presented in Problem 16
        read_file: Read data from text file
        train: Decision stump alforithm for multi-dimensional data
        verify: Verify the model
    '''

    def gen_data(self, size=20):
        '''Generate a one-dimensional data

        a) Generate x by a uniform distribution in [-1, 1];
        b) Generate y by s_hat(x)+noise where s_hat(x)=sign(x) and the noise
           flips the result with 0.2 probability.

        Args:
            size: number of examples to generate

        Returns:
            X: array of x
            Y: array of y
        '''
        X = np.random.uniform(-1, 1, size=size)
        Y = np.ones(size)
        for i in range(size):
            if X[i]<0:
                Y[i]=-1
            if np.random.random_sample()<0.2:
                Y[i]=-Y[i]
        return X, Y

    def calc_err(self, X, Y, s, theta):
        '''Calculate error rate
        '''
        m = X.size
        err_cnt = 0
        for i in range(m):
            if s*(X[i]-theta)*Y[i]<0:
                err_cnt+=1
        return err_cnt/m

    def train_1d(self, X, Y):
        '''Find the optimal s and optimal theta that make the Ein value lowest
        '''
        sorted_X = np.sort(X)
        theta_candidates = []
        for i in range(sorted_X.size-1):
            theta_candidates.append((sorted_X[i]+sorted_X[i+1])/2)
        s_candidates = [-1, 1]
        opt_s = 1
        opt_theta = 0
        opt_err = self.calc_err(X, Y, opt_s, opt_theta)
        for s in s_candidates:
            for theta in theta_candidates:
                err = self.calc_err(X, Y, s, theta)
                if err<opt_err:
                    opt_err = err
                    opt_s = s
                    opt_theta = theta
        return opt_err, opt_s, opt_theta

    def calc_Eout(self, s, theta, noise=0.2):
        '''Compute Eout value with the formula that presented in Problem 16
        '''
        mu = (1-s+s*np.fabs(theta))/2
        Eout = (1-noise)*mu + noise*(1-mu)
        return Eout

    def read_file(self, fp):
        '''Read data from text file

        Read data from  'hw2_train.dat' and 'hw2_test.dat', save as numpy.ndarray
        '''
        with open(fp, 'r') as f:
            lines = f.readlines()
        m = len(lines)
        row0 = lines[0].strip().split(' ')
        d = len(row0)-1
        X = np.ones((m, d))
        Y = np.zeros(m)
        i = 0
        for line in lines:
            row = line.strip().split(' ')
            X[i, :] = row[:-1]
            Y[i] = row[-1]
            i += 1
        return X, Y

    def train(self, X, Y):
        '''Decision stumps working for multi-dimensional data.

        In paricular, each decision stump now deals with a specific dimension i, as shown below:
                        h_s,i,theta(x) = s*sign(x_i-theta)
        Implementation for multi-dimensional data:
            a) for each dimension i=1,2,...,d, find the best decision stump h_s,i,theta using the
               one-dimensional decision stump algorithm that Implemented in 'train_1d'.
            b) return the "best of best" decision stump in terms of Ein.
        '''
        opt_i = 0
        opt_s = 1
        opt_theta = 0
        opt_err = 1
        m, d = X.shape
        for i in range(d):
            X_i = X[:, i]
            err, s, theta = self.train_1d(X_i, Y)
            if err<opt_err:
                opt_i = i
                opt_s = s
                opt_theta = theta
                opt_err = err
        self.opt_err = opt_err
        self.opt_i = opt_i
        self.opt_s = opt_s
        self.opt_theta = opt_theta

    def verify(self, X, Y):
        '''Verify the model

        Use the returned decision stump to predict label of each example within Dtest, report the
        error rate of Dtest as an estimate of Eout.
        '''
        X_i = X[:, self.opt_i]
        m = X.shape[0]
        err_cnt = 0
        for j in range(m):
            if self.opt_s*(X_i[j]-self.opt_theta)*Y[j]<0:
                err_cnt += 1
        return err_cnt/m
