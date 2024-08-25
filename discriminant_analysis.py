import numpy as np
import sys

class Discriminant:
    def __init__(self, n_components, case):
        self.n_components = n_components
        self.linear_discriminant = None
        self.case = case

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        SW = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y==c]
            mean_c = np.mean(X_c, axis=0)

            SW = SW + ((X_c - mean_c).T.dot(X_c - mean_c)/len(X_c))
            print(len(X_c))

        eigenvalues, eigenvectors = np.linalg.eigh(SW)

        # Eigen values and vectors are arranged in decending order
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        idxs1 = []
        idxs0 = []
        for i,la in enumerate(eigenvalues):
            if la>1e-5:
                idxs1.append(i)
            else:
                idxs0.append(i)
        # print(idxs1)
                
        match self.case:
            case 1:
                eigenvalues1 = eigenvalues[idxs1]
                eigenvectors1 = eigenvectors[idxs1]
                mults_eigen1 = np.diagflat(np.power(abs(eigenvalues1), -0.5))
                psi = np.dot(eigenvectors1.T, mults_eigen1).T
                m = len(idxs1)
            case 2:
                min_eigenvalue = [min(abs(eigenvalues))]*(len(idxs0))
                eigenvectors0 = eigenvectors[idxs0]
                mults_eigen0 = np.diagflat(np.power(min_eigenvalue, -0.5))
                psi = np.dot(eigenvectors0.T, mults_eigen0).T
                m = len(idxs0)
            case 3:
                eigenvalues1 = eigenvalues[idxs1]
                eigenvectors1 = eigenvectors[idxs1]
                mults_eigen1 = np.diagflat(np.power(abs(eigenvalues1), -0.5))
                psi1 = np.dot(eigenvectors1.T, mults_eigen1).T
                min_eigenvalue = [min(abs(eigenvalues))]*(len(idxs0))
                eigenvectors0 = eigenvectors[idxs0]
                mults_eigen0 = np.diagflat(np.power(min_eigenvalue, -0.5))
                psi0 = np.dot(eigenvectors0.T, mults_eigen0).T
                psi = np.concatenate((psi1, psi0), axis=0)
                m = len(idxs1) + len(idxs0)
            case _:
                print('❌')
                sys.exit('case can only be 1 or 2 or 3')
        
        Y = np.dot(X, psi.T)
        Y_mean_overall = np.mean(Y, axis=0)
        WT = np.zeros((m ,m))

        for c in class_labels:
            Y_c = Y[y==c]
            WT = WT + (Y_c - Y_mean_overall).T.dot(Y_c - Y_mean_overall)

        Y_eigenvalues, Y_eigenvectors = np.linalg.eigh(WT)

        Y_idxs = np.argsort(abs(Y_eigenvalues))[::-1]
        Y_eigenvalues = Y_eigenvalues[Y_idxs]
        Y_eigenvectors = Y_eigenvectors[Y_idxs]

        Y_discriminant = Y_eigenvectors[0:self.n_components]

        self.linear_discriminant = np.dot(psi.T, Y_discriminant.T)

    def transform(self, X):
        return np.dot(X, self.linear_discriminant)
