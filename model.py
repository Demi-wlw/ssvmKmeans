# Implements the structured model of clustering algorithms
# Includes Spectral, Point-Incremental K-means and Discretized Spectral
# Reference: Supervised K-means Clustering, Thomas Finley 2007

import numpy as np
from scipy import sparse, linalg

class StructuredModel(object):
    """Interface definition for Structured Learners.
    This class defines what is necessary to use the structured svm.
    You have to implement joint_feature, loss and inference.
    """
    def __repr__(self):
        return ("%s(size_joint_feature: %d)"
                % (type(self).__name__, self.size_joint_feature))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_joint_feature, the dimensionality of the joint
        features for an instance with labeling (x, y). This is also the dimension
        of the weights w.
        """
        self.size_joint_feature = None

    def _check_size_w(self, w):
        if w.shape != (self.size_joint_feature,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             ('('+str(self.size_joint_feature)+',)', w.shape))

    def initialize(self, X, Y):
        # set any data-specific parameters in the model
        pass

    def joint_feature(self, X, y):
        raise NotImplementedError()

    def loss(self, y, y_hat):
        raise NotImplementedError()

    def inference(self, X, w, y=None):
        # if y is not None, it is loss_augmented_inference
        raise NotImplementedError()

    def loss_augmented_inference(self, X, w, y):
        return self.inference(X, w, y)


class Spectral(StructuredModel):
    """Formulate Supervised clustering with Spectral Clustering
    Inputs X are feature matrix, labels y are 0 to n_classes.
    Notes
    ------
    No bias / intercept is learned. It is recommended to add a constant one
    feature to the data.
    It is also highly recommended to use n_jobs=1 in the learner when using
    this model. Trying to parallelize the trivial inference will slow
    the infernce down a lot!
    
    Parameters
    ----------
    n_clusters: int
        Number of clusters to perform clustering algorithm.
    n_features : int
        Number of features of inputs x.
        If None, it is inferred from data.
    n_classes : int, default=None
        Number of classes in dataset.
        If None, it is inferred from data.
    loss_scale : int, default=100
        Rescale the loss.
    """
    def __init__(self, n_clusters, n_features=None, n_classes=None, loss_scale=100):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_classes = n_classes
        self.loss_scale = loss_scale
        self._set_size_joint_feature()

    def _set_size_joint_feature(self):
        if self.n_features is not None:
            self.size_joint_feature = self.n_features

    def initialize(self, X, y):
        n_features = X.shape[-1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_classes = np.unique(y).size
        if self.n_classes is None:
            self.n_classes = n_classes
        elif self.n_classes != n_classes:
            raise ValueError("Expected %d classes, got %d"
                             % (self.n_classes, n_classes))
        self._set_size_joint_feature()

    def __repr__(self):
        return ("%s(n_clusters: %d, n_features: %d, n_classes: %d, loss_scale: %d)"
                % (type(self).__name__, self.n_clusters, self.n_features, self.n_classes, self.loss_scale))

    def _label_matrix(self, y):
        """Given a labeling y, return the indicator label matrix.

        There are two types of labels that occur within the relaxed
        problem: the first is a simple vector with as many entries as
        there are points.  Two entries i and j are equal if and only if
        the corresponding items i and j are in the same cluster.

        Given such a one dimensional label vector (which may be a list,
        tuple, or other such structure), this will return a matrix with as
        many columns as there are classes/clusters, and as many rows as there are
        items.  For a class/cluster of size N, its corresponding column will
        have all zeros, except for those entries.

        Parameters
        ----------
        y : 1darray, shape (N,) or 2darray, shape (N, n_classes/n_clusters)

        Return
        -------
        Y : sparse matrix, shape (N, n_cls)
            n_cls can be either number of classes or number of clusters

        For convenience for this function's use in other contexts, if what
        is passed in appears to have two dimensions, then that item is
        returned without further processeing.
        """
        if y.ndim == 2:
            # This is already a two dimensional array so we may just return it.
            return sparse.csc_matrix(y)
        elif y.ndim == 1:
            n_cls = len(np.unique(y))
            Y = sparse.lil_matrix((y.size, n_cls))
            # Compute the class/cluster memberships.
            index2members = {} # class: set{index,...}
            for n, c in enumerate(y):
                Y[n,c] = 1
                if c not in index2members:
                    index2members[c] = set([n])
                else:
                    index2members[c].add(n)

            # For each class, scale the appropriate column.
            for cls, s in index2members.items():
                Y[:,cls] = Y[:,cls] / np.sqrt(len(s))
            return Y.tocsc()
        else:
            raise ValueError("y Expected 1 or 2 dimensions, got %d" % (y.ndim))
    
    def joint_feature(self, X, y):
        """Compute joint feature vector Psi(x,y).
        Feature representation joint_feature, such that the discriminant function of
        (x, y) with a weight vector w is given by np.dot(w, joint_feature(x, y)).
        Parameters
        ----------
        X : 2darray, shape (N, n_features)
            Input sample features.
        y : 1darray, shape (N,) or 2darray, shape (N, n_classes/n_clusters)
            Class/cluster labels.
        Returns
        -------
        psi : 1darray, shape (size_joint_feature,)
            Joint feature vector associated with samples (X, y).
        """
        # put feature vector in the place of the weights corresponding to y
        psi = np.zeros((self.size_joint_feature,))
        # For convenience, convert to matrix representation.
        Y = self._label_matrix(y) 
        N = len(X)
        for i in range(N):
            for j in range(N):
                Yij = ((Y[i,:].multiply(Y[j,:])).sum(1)).item() # scalar
                if not Yij: continue # if Yij is 0, continue
                psi_ij = X[i,:]*X[j,:] # elementwise multiplication shape (n_features,)
                psi += Yij * psi_ij
        factor = 1/N # rescale?
        psi = psi*factor
        return psi

    def discriminant_function(self, X, w, y):
        psi = self.joint_feature(X,y)
        return np.dot(w, psi)

    def loss(self, y, y_hat):
        """Return loss_scale*[1 - 1/k trace(Y_hat' * Y * Y' * Y_hat)].

        Parameters
        ----------
        y : 1darray, shape (N,) or 2darray, shape (N, n_classes)
            Truth labels.
        y_hat : 1darray, shape (N,) or 2darray, shape (N, n_clusters)
            Predicted labels.

        Returns
        -------
        scaler : Scaled loss value with loss_scale.
        """
        Y = self._label_matrix(y) 
        Y_hat = self._label_matrix(y_hat)
        # Y: shape (N, n_classes)   Y_hat: shape (N, n_clusters)
        YY = Y_hat.T@Y # shape (n_clusters, n_classes)
        trace = (YY@YY.T).diagonal().sum()
        return self.loss_scale - (self.loss_scale / YY.shape[0]) * trace

    def _kernel_matrix(self, X, w, y):
        """Get the representative kernel matrix K.

        Given a matrix of example vectors, and the weights w, return
        the representative kernel matrix.  This will be a square matrix
        with as many columns and rows as elements in x.

        In the event that the argument 'y' is included, the matrix will be
        adjusted so that, if we are trying to find a maximizing Y for
        trace(Y'KY), this will incorporate the loss of 'y' versus our
        maximizing Y into this maximization problem.  (For structural
        learning, this makes the predictive inference problem be the
        constraint inference problem for margin-scaled 1-norm SVM
        structural learning.)

        Parameters
        ----------
        X : 2darray, shape (N, n_features)
        w : 1darray, shape (size_joint_feature,)
        y : 1darray, shape (N,) or 2darray, shape (N, n_classes)
            Truth labels. If None, construct the original K. If it is provided,
            construct K incorporating the loss of y.

        Returns
        -------
        K : 2darray, shape (N, N)
        """
        N = len(X)
        K = np.array([np.dot(w, X[i,:]*X[j,:]) for i in range(N) for j in range(N)])
        K = K.reshape(N, N)
        K /= N # rescale?
        if y is not None:
            Y = self._label_matrix(y)
            YY = (Y@Y.T) / self.n_clusters * self.loss_scale
            K -= YY
            #delta = loss_scale / N
            #for i in range(N):
            #    K[i,i] += delta
        return np.array(K)

    def _spectral(self, K, y):
        """Compute a matrix's relaxed spectral clustering.
        
        Effectively, this will return a matrix with the n_clusters eigenvectors
        associated with the largest eigenvalues.  The 'best' clustering
        for a matrix K is the matrix Y (with as many rows as K and as many
        columns as clusters) that maximizes trace(Y'KY) which, if we have
        relaxed conditions of Y so that the columns are mearly mutually
        orthonormal, are these eigenvectors.  Some further discretization
        procedure must take place if one desires discrete solutions.

        Parameters
        ----------
        K : 2darray, shape (N,N)
            Kernel matrix.

        Returns
        -------
        Y_hat : 2darray, shape (N, n_clusters)
                Estimated continuous clustering matrix.
        """
        N = len(K)
        Y_hat = np.zeros((N, self.n_clusters))
        _, eig_vec = linalg.eigh(K, subset_by_index=[N-self.n_clusters, N-1])
        # eig_vec, shape (N, n_clusters) but columns from smaller eig_value to largest eig_value
        Y_hat = np.flip(eig_vec, axis=1)
        return Y_hat

    def inference(self, X, w, y=None):
        """Inference for X using parameters w.
        Correspond to find_most_violated_constraint.
        
        Parameters
        ----------
        X : 2darray, shape (N, n_features)
            Input sample features.
        w : 1darray, shape (size_joint_feature,)
            Weight parameters.
        y : 1darray, shape (N,)
            If None, it makes predictions. If not None, it is loss augmented inference.
        
        Returns
        -------
        Y_hat : 2darray, shape (N, n_clusters)
            Predicted cluster label matrix.
        """
        K = self._kernel_matrix(X,w,y)
        return self._spectral(K,y)


class Kmeans(Spectral):
    """Formulate Supervised clustering with Iterative Point-Incremental K-means.
    Inherits from Spectral class.
    """
    def _cluster2values(self, y, K):
        """Return mappings from cluster to value, and cluster to count.

        In the discrete case, each cluster contributes a certain amount to
        the discriminant function, and each cluster has a certain number
        of elements.  This will return mappings to both (counts, then
        values).

        Parameters
        ----------
        y : 1darray, shape (N,)
            classes/clusters labels.
        K : 2darray, shape (N, N)
            kernel matrix.

        Returns
        -------
        cluster2count : dict, {cls: count}
        cluster2values : dict, {cls: value}
        """
        # Make the mapping of clusters to sets of elements.
        cluster2set = {}
        for n,c in enumerate(y):
            if c not in cluster2set: cluster2set[c] = set([n])
            else: cluster2set[c].add(n)
        # Make the mapping of counts.
        cluster2count = dict((c,len(s)) for c,s in cluster2set.items())
        # Make the mapping of values.
        cluster2value = dict((c,sum(K[i,j] for i in s for j in s)/len(s)) for c,s in cluster2set.items())
        return cluster2count, cluster2value

    def _kmeans(self, K, y):
        """Runs kmeans.

        This will run a point-iterative version of kmeans given the
        indicated similarity matrix (kernel matrix).  It will start with 
        the indicated argument y as a given based clustering -- the method 
        will search for a superior clustering to that, and if it does not find one,
        that is the one that shall be returned.  If y is None, then no
        initial 'best known' clustering is supposed.

        Parameters
        ----------
        K : 2darray, shape (N, N)
            kernel matrix.
        y : 1darray, shape (N,)
            classes/clusters labels.
        
        Returns
        -------
        maxlabel : 1darray, shape (N,)
        """
        if y is not None:
            maxdiscrim = sum(self._cluster2values(y,K)[1].values()) 
            maxlabel = y
        else:
            maxdiscrim = None
            maxlabel = None
        for bail in range(10):
            # Make a random assignment of clusters to values.
            np.random.seed(0)
            y_hat = np.random.randint(self.n_clusters, size=len(K))
            if maxdiscrim == None:
                maxdiscrim = sum(self._cluster2values(y_hat,K)[1].values())
                maxlabel = y_hat
            n_iter = 0
            c2counts, c2vals = self._cluster2values(y_hat, K)

            for bailout in range(100):
                n_iter+=1
                y_hatnew = y_hat
                change = 0.0
                n_changed = 0 
                # Come up with the order in which the points shall be assigned.
                assignment_order = np.arange(len(K))
                np.random.seed(0)
                np.random.shuffle(assignment_order)
                for i in assignment_order:
                    # What is the current cluster of this object?
                    current_cluster = y_hatnew[i]
                    # For each point, and each cluster, calculate what adding
                    # this point to this cluster would do to the cluster
                    # objective value.
                    cluster2sum = [K[i,i]]*self.n_clusters # list of size n_clusters
                    for j in range(len(K)):
                        if i==j: continue
                        cluster2sum[y_hatnew[j]] += 2*K[i,j]
                    # Calculate what removing the item from its current
                    # cluster would do to that cluster's contribution.
                    ccv = (c2vals[current_cluster]*c2counts[current_cluster]
                        - cluster2sum[current_cluster]) / max(c2counts[current_cluster]-1,1)
                    # If we remove, the score would increase by this amount.
                    ccv_delta = ccv - c2vals[current_cluster] 
                    # For each cluster, calculate the change in objective
                    # value that would result from adding this item.
                    c2delta = {current_cluster: -ccv_delta}
                    for c, v in c2vals.items():
                        if c == current_cluster: continue
                        num = c2counts[c]
                        delta = -v/(num+1) + cluster2sum[c]/(num+1)
                        c2delta[c] = delta
                    #pdb.set_trace()
                    # Find the maximizing cluster, and set it.
                    maxv, maxc = max((v,c) for c,v in c2delta.items())
                    #maxc, maxv = rgen.choice(c2delta.items())
                    y_hatnew[i] = maxc
                    # Update our opinions about cluster values?
                    c2vals[maxc] += maxv
                    c2vals[current_cluster] += ccv_delta
                    c2counts[current_cluster] -= 1
                    c2counts[maxc] += 1
                    if maxc != current_cluster: n_changed+=1
                    # What is the overall change?
                    change += maxv + ccv_delta
                # If the change is pretty near 0, we may as well terminate.
                y_hat = y_hatnew
                if change <= 1e-8: break
            
            discrim = sum(c2vals.values())
            if discrim > maxdiscrim:
                maxlabel = y_hat
                maxdiscrim = discrim
        return maxlabel

    def inference(self, X, w, y=None):
        """Inference for X using parameters w.
        Correspond to find_most_violated_constraint.
        
        Parameters
        ----------
        X : 2darray, shape (N, n_features)
            Input sample features.
        w : 1darray, shape (size_joint_feature,)
            Weight parameters.
        y : 1darray, shape (N,)
            If None, it makes predictions. If not None, it is loss augmented inference.
        
        Returns
        -------
        y_hat : 1darray, shape (N,)
            Predicted cluster labels.
        """
        K = self._kernel_matrix(X,w,y)
        return self._kmeans(K,y)


class DiscretizedSpectral(Kmeans):
    """Formulate Supervised clustering with discretized spectral method via 
    Bach and Jordan post-processing. It is a combination of the Spectral and 
    Iterative Point-Incremental K-means.
    """
    def inference(self, X, w, y=None):
        """Inference for X using parameters w.
        Correspond to find_most_violated_constraint.
        
        Parameters
        ----------
        X : 2darray, shape (N, n_features)
            Input sample features.
        w : 1darray, shape (size_joint_feature,)
            Weight parameters.
        y : 1darray, shape (N,)
            If None, it makes predictions. If not None, it is loss augmented inference.
        
        Returns
        -------
        y_hat : 1darray, shape (N,)
            Predicted cluster labels.
        """
        K = self._kernel_matrix(X,w,y)
        Y_hat = self._spectral(K,y) # 2darray, shape (N, n_clusters)
        K = np.dot(Y_hat, Y_hat.T) # K_new = Y_hat * Y_hat'
        return self._kmeans(K,y)