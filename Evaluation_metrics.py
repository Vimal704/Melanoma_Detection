class EvaluationMetrics:
    def __init__(self, classifier, X, y):
        y_pred = classifier.predict(X)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        n = len(y)
        for i in range(n):
            if y_pred[i]==1:
                if y[i]==1:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i]==1:
                    fn += 1
                else:
                    tn += 1
        self.TP = tp
        self.TN = tn
        self.FP = fp
        self.FN = fn

    def Accuracy(self):
        return (self.TP + self.TN)/(self.TN + self.FP + self.TN + self.FN)
    
    def Sensitivity(self):
        return (self.TP) / (self.TP + self.FN)
    
    def Specificity(self):
        return self.TN / (self.TN + self.FP)
    
    def PPV(self):
        return self.TP/(self.TP + self.FP)
    
    def NPV(self):
        return self.TN / (self.TN + self.FN)
    
    def ALL(self):
        print(f'Accuracy = {(self.TP + self.TN)/(self.TN + self.FP + self.TN + self.FN)}')
        print(f'Sensitivity = {(self.TP) / (self.TP + self.FN)}')
        print(f'Specificity = {self.TN / (self.TN + self.FP)}')
        print(f'PPV(Positive predictive value) = {self.TP/(self.TP + self.FP)}')
        print(f'NPV(Negative predictive value) = {self.TN / (self.TN + self.FN)}')
        



