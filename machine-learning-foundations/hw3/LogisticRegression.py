import numpy as np

class LogisticRegression():
    
    def load_data(self, fp):
        with open(fp, 'r') as f:
            lines = f.readlines()
        N = len(lines)
        d = len(lines[0].strip().split(' ')) - 1
        X = np.ones((N, d+1))
        y = np.zeros(N)
        for i in range(N):
            row = lines[i].strip().split()
            X[i, 1:] = row[:-1]
            y[i] = row[-1]
        return X, y
    
    def train(self, X, y, eta=0.001, T=2000, stochastic='false'):
        N, d = X.shape
        w = np.zeros(d)
        
        def gradient(w):
            g = np.zeros(d)
            for n in range(N):
                g += gradient_one(w, X[n], y[n])
            return g/N
        
        def gradient_one(w, x, y):
            g = -1 * x * y / (1+np.exp(y*np.dot(w, x)))
            return g
        
        if stochastic=='false':
            for i in range(T):
                w -= eta*gradient(w)
        
        if stochastic=='cyclic':
            for i in range(T):
                n = i%N
                w -= eta*gradient_one(w, X[n], y[n])
            
        self.w = w
        
    def test(self, X, y):
        N, d = X.shape
        err = 0
        for c in y*np.dot(X, self.w):
            if c<=0:
                err += 1
        return err/N
