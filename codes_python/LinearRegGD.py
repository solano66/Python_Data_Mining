# 定義類別LinearRegressionGD
class LinearRegressionGD(object):
    # 定義物件初始化方法，物件初始化時帶有兩個屬性
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    # 定義物件的方法fit()，此方法會根據傳入的X 與y 計算屬性
    # w_ 和cost_
    def fit(self, X, y):
        # 隨機初始化屬性w_
        self.w_ = np.random.randn(1 + X.shape[1])
        # 損失函數屬性cost_
        self.cost_ = []
        # 根據物件屬性eta 與n_iter，以及傳入的X 與y 計算屬性
        # w_ 和cost_
        for i in range(self.n_iter):
            output = self.lin_comb(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 # 式(1.1)
            self.cost_.append(cost)
        return self
    # 定義fit 方法會用到的lin_comb 線性組合方法
    def lin_comb(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    # 定義物件的方法predict()
    def predict(self, X):
        return self.lin_comb(X)