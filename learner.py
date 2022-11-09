# Implements structured SVM as described in Joachims et. al.
# Cutting-Plane Training of Structural SVMs
# Specific for clustering model implementations
# Reference: PyStruct, Andreas Mueller

from time import time
import numpy as np
import cvxopt
import cvxopt.solvers
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

class BaseSSVM(BaseEstimator):
    """Base SSVM that implements common functionality."""
    def __init__(self, model, max_iter=100, C=1.0, verbose=0, n_jobs=1, show_loss_every=0):
        self.model = model
        self.max_iter = max_iter
        self.C = C
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.show_loss_every = show_loss_every

    def predict(self, X):
        """Predict output on examples in X.

        Parameters
        ----------
        X : 2darray, shape (N, n_features)
            Example instances. Contains the structured input objects.
        
        Returns
        -------
        y_pred : 1darray, shape (N,)
            Inference results for X using the learned parameters w.
        """
        return self.model.inference(X, self.w)

    def scores(self, X, y):
        """Compute loss and discriminant function value.
        Returns the clustering accuracy over X and y.
        
        Parameters
        ----------
        X : 2darray, shape (N, n_features)
            Evaluation dataset.
        y : 1darray, shape (N,)
            True labels.
        
        Returns
        -------
        loss : float
            Scaled loss value with loss_scale.
        dfv : float
            Discriminant function value.
        """
        y_pred = self.predict(X)
        loss = self.model.loss(y, y_pred)
        dfv =  self.model.discriminant_function(X, self.w, y_pred)
        return loss, dfv

    def _compute_training_loss(self, train_data, Y_train, iteration):
        """Compute training loss for output / training curve

        Args:
            train_data : 3darray, shape (M, N_m, n_features)
            Y_train : 2darray, shape (M, N_m)
        """
        if (self.show_loss_every != 0 and not iteration % self.show_loss_every):
            if not hasattr(self, 'loss_curve_'):
                self.loss_curve_ = []
            display_loss = 0
            for m in range(len(Y_train)):
                train_loss, train_dfv = self.scores(train_data[m], Y_train[m])
                display_loss += train_loss / len(Y_train)
            if self.verbose > 0:
                print("[INFO] Current loss: %.4f" % (display_loss))
            self.loss_curve_.append(display_loss)

    def _objective(self, X, Y):
        def find_constraint(model, X_m, y, w, y_hat=None, compute_difference=True):
            """Find most violated constraint, or, given y_hat,
            find slack and djoint_feature for this constraing.
            As for finding the most violated constraint, it is enough to compute
            joint_feature(X_m, y_hat), not djoint_feature, we can optionally skip
            computing joint_feature(X_m, y) using compute_differences=False
            """

            if y_hat is None:
                y_hat = model.loss_augmented_inference(X_m, w, y)
            delta_joint_feature = -model.joint_feature(X_m, y_hat)
            if compute_difference:
                delta_joint_feature += model.joint_feature(X_m, y)

            loss = model.loss(y, y_hat)
            slack = max(loss - np.dot(w, delta_joint_feature), 0)
            return y_hat, delta_joint_feature, slack, loss
       
        objective = 0
        constraints = Parallel(n_jobs=self.n_jobs)(delayed(find_constraint)(self.model, X_m, y, self.w) for X_m, y in zip(X, Y))
        slacks = list(zip(*constraints))[2]
        objective = max(np.sum(slacks), 0) * self.C + np.sum(self.w ** 2) / 2.
        return objective

class NoConstraint(Exception):
    # raised if we can not construct a constraint from cache
    pass

class OneSlackSSVM(BaseSSVM):
    """Structured SVM solver for the 1-slack QP with l1 slack penalty.
    Implements margin rescaled structural SVM using
    the 1-slack formulation and cutting plane method, solved using CVXOPT.
    The optimization is restarted in each iteration.
    Parameters
    ----------
    model : StructuredModel
        Object containing the model structure. Has to implement
        `loss`, `inference` and `loss_augmented_inference`.
    max_iter : int, default=10000
        Maximum number of passes over dataset to find constraints.
    C : float, default=1
        Regularization parameter.
    check_constraints : bool
        Whether to check if the new "most violated constraint" is
        more violated than previous constraints. Helpful for stopping
        and debugging, but costly.
    verbose : int
        Verbosity.
    negativity_constraint : list of ints
        Indices of parmeters that are constraint to be negative.
        This is useful for learning submodular CRFs (inference is formulated
        as maximization in SSVMs, flipping some signs).
    break_on_bad : bool default=False
        Whether to break (start debug mode) when inference was approximate.
    n_jobs : int, default=1
        Number of parallel jobs for inference. -1 means as many as cpus.
    show_loss_every : int, default=0
        Controlls how often the hamming loss is computed (for monitoring
        purposes). Zero means never, otherwise it will be computed very
        show_loss_every'th epoch.
    tol : float, default=1e-3
        Convergence tolerance. If dual objective decreases less than tol,
        learning is stopped. The default corresponds to ignoring the behavior
        of the dual objective and stop only if no more constraints can be
        found.
    inference_cache : int, default=0
        How many results of loss_augmented_inference to cache per sample.
        If > 0 the most violating of the cached examples will be used to
        construct a global constraint. Only if this constraint is not violated,
        inference will be run again. This parameter poses a memory /
        computation tradeoff. Storing more constraints might lead to RAM being
        exhausted. Using inference_cache > 0 is only advisable if computation
        time is dominated by inference.
    cache_tol : float, None or 'auto' default='auto'
        Tolerance when to reject a constraint from cache (and do inference).
        If None, ``tol`` will be used. Higher values might lead to faster
        learning. 'auto' uses a heuristic to determine the cache tolerance
        based on the duality gap, as described in [3].
    inactive_threshold : float, default=1e-5
        Threshold for dual variable of a constraint to be considered inactive.
    inactive_window : float, default=50
        Window for measuring inactivity. If a constraint is inactive for
        ``inactive_window`` iterations, it will be pruned from the QP.
        If set to 0, no constraints will be removed.
    switch_to : None or string, default=None
        Switch to the given inference method if the previous method does not
        find any more constraints.
    logger : logger object, default=None
        Pystruct logger for storing the model or extracting additional
        information.
    Attributes
    ----------
    w : nd-array, shape=(model.size_joint_feature,)
        The learned weights of the SVM.
    old_solution : dict
        The last solution found by the qp solver.
    ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.
    ``objective_curve_`` : list of float
       Cutting plane objective after each pass through the dataset.
    ``primal_objective_curve_`` : list of float
        Primal objective after each pass through the dataset.
    ``timestamps_`` : list of int
       Total training time stored before each iteration.
    References
    ----------
    [1] Thorsten Joachims, and Thomas Finley and Chun-Nam John Yu:
        Cutting-plane training of structural SVMs, JMLR 2009
    [2] Andreas Mueller: Methods for Learning Structured Prediction in
        Semantic Segmentation of Natural Images, PhD Thesis.  2014
    [3] Andreas Mueller and Sven Behnke: Learning a Loopy Model For Semantic
        Segmentation Exactly, VISAPP 2014
    """

    def __init__(self, model, max_iter=10000, C=1.0, check_constraints=False,
                 n_jobs=1, verbose=0, negativity_constraint=None,
                 break_on_bad=False, show_loss_every=0, tol=1e-3,
                 inference_cache=0, inactive_threshold=1e-5,
                 inactive_window=50, cache_tol='auto'):

        BaseSSVM.__init__(self, model, max_iter, C, verbose=verbose, n_jobs=n_jobs, show_loss_every=show_loss_every)
        self.negativity_constraint = negativity_constraint
        self.check_constraints = check_constraints
        self.break_on_bad = break_on_bad
        self.tol = tol
        self.cache_tol = cache_tol
        self.inference_cache = inference_cache
        self.inactive_threshold = inactive_threshold
        self.inactive_window = inactive_window

    def _solve_1_slack_qp(self, constraints, n_samples):
        # constraints: [(joint_feature, loss), ...]
        C = np.float(self.C) * n_samples  # this is how libsvm/svmstruct do it
        joint_features = [c[0] for c in constraints]
        losses = [c[1] for c in constraints]

        joint_feature_matrix = np.vstack(joint_features)
        n_constraints = len(joint_features)
        P = cvxopt.matrix(np.dot(joint_feature_matrix, joint_feature_matrix.T))
        # q contains loss from margin-rescaling
        q = cvxopt.matrix(-np.array(losses, dtype=np.float))
        # constraints: all alpha must be >zero
        idy = np.identity(n_constraints)
        tmp1 = np.zeros(n_constraints)
        # positivity constraints:
        if self.negativity_constraint is None:
            # empty constraints
            zero_constr = np.zeros(0)
            joint_features_constr = np.zeros((0, n_constraints))
        else:
            joint_features_constr = joint_feature_matrix.T[self.negativity_constraint]
            zero_constr = np.zeros(len(self.negativity_constraint))

        # put together
        G = cvxopt.sparse(cvxopt.matrix(np.vstack((-idy, joint_features_constr))))
        h = cvxopt.matrix(np.hstack((tmp1, zero_constr)))

        # equality constraint: sum of all alpha must be = C
        A = cvxopt.matrix(np.ones((1, n_constraints)))
        b = cvxopt.matrix([C])

        # solve QP model
        cvxopt.solvers.options['feastol'] = 1e-5
        try:
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        except ValueError:
            solution = {'status': 'error'}
        if solution['status'] != "optimal":
            print("[INFO] Regularizing QP!")
            P = cvxopt.matrix(np.dot(joint_feature_matrix, joint_feature_matrix.T)
                              + 1e-8 * np.eye(joint_feature_matrix.shape[0]))
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            if solution['status'] != "optimal":
                raise ValueError("QP solver failed. Try regularizing your QP.")

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.old_solution = solution
        self.prune_constraints(constraints, a)

        # Support vectors have non zero lagrange multipliers
        sv = a > self.inactive_threshold * C
        if self.verbose > 1:
            print("[INFO] %d support vectors out of %d points!" % (np.sum(sv), n_constraints))
        self.w = np.dot(a, joint_feature_matrix)
        # we needed to flip the sign to make the dual into a minimization model
        return -solution['primal objective']

    def prune_constraints(self, constraints, a):
        # append list for new constraint
        self.alphas.append([])
        assert(len(self.alphas) == len(constraints))
        for constraint, alpha in zip(self.alphas, a):
            constraint.append(alpha)
            constraint = constraint[-self.inactive_window:]

        # prune unused constraints:
        # if the max of alpha in last 50 iterations was small, throw away
        if self.inactive_window != 0:
            max_active = [np.max(constr[-self.inactive_window:])
                          for constr in self.alphas]
            # find strongest constraint that is not ground truth constraint
            strongest = np.max(max_active[1:])
            inactive = np.where(max_active
                                < self.inactive_threshold * strongest)[0]

            for idx in reversed(inactive):
                # if we don't reverse, we'll mess the indices up
                del constraints[idx]
                del self.alphas[idx]

    def _check_bad_constraint(self, violation, djoint_feature_mean, loss, old_constraints, break_on_bad, tol=None):
        violation_difference = violation - self.last_slack_
        if self.verbose > 1:
            print("[INFO] New violation: %.4f difference to last: %.4f"
                  % (violation, violation_difference))
        if violation_difference < 0 and violation > 0 and break_on_bad:
            raise ValueError("Bad inference: new violation is smaller than"
                             " old.")
        if tol is None:
            tol = self.tol
        if violation_difference < tol:
            if self.verbose:
                print("[INFO] New constraint too weak!")
            return True
        equals = [True for djoint_feature_, loss_ in old_constraints
                  if (np.all(djoint_feature_ == djoint_feature_mean) and loss == loss_)]

        if np.any(equals):
            return True

        if self.check_constraints:
            for con in old_constraints:
                # compute violation for old constraint
                violation_tmp = max(con[1] - np.dot(self.w, con[0]), 0)
                if self.verbose > 5:
                    print("violation old constraint: %.4f" % violation_tmp)
                # if violation of new constraint is smaller or not
                # significantly larger, don't add constraint.
                # if smaller, complain about approximate inference.
                if violation - violation_tmp < -1e-5:
                    if self.verbose:
                        print("bad inference: %.4f" % (violation_tmp - violation))
                    if break_on_bad:
                        raise ValueError("Bad inference: new violation is"
                                         " weaker than previous constraint.")
                    return True
        return False

    @classmethod
    def constraint_equal(cls, y_1, y_2):
        """
        This now more complex. y_1 and/or y_2 (I think) can be: array, pair of
        arrays, pair of list of arrays (multitype)
        We need to compare those!
        """
        if isinstance(y_1, tuple):
            # y_1 is relaxed Y
            # y_1 and y_2 might be lists of ndarray (multitype) instead of
            #    ndarray (single type)
            u_m_1, pw_m_1 = y_1
            if isinstance(y_2, tuple):  # we then compare two relaxed Ys
                u_m_2, pw_m_2 = y_2
                # now, do we multitype or single type relaxed marginals??
                if isinstance(u_m_1, list):
                    return all(np.all(_um1 == _um2) for _um1, _um2
                               in zip( u_m_1,  u_m_2)) \
                        and all(np.all(_pw1 == _pw2) for _pw1, _pw2
                                in zip(pw_m_1, pw_m_2))
                else:
                    return np.all(u_m_1 == u_m_2) and np.all(pw_m_1, pw_m_2)
            else:
                # NOTE original code was possibly comparing array and scalar
                # return np.all(y_1[0] == y_2[0]) and np.all(y_1[1] == y_2[1])
                return False
        # might compare array and tuple... :-/  Was like that, I keep
        return np.all(y_1 == y_2)

    def _update_cache(self, X, Y, Y_hat):
        """Updated cached constraints."""
        if self.inference_cache == 0:
            return
        if (not hasattr(self, "inference_cache_")
                or self.inference_cache_ is None):
            self.inference_cache_ = [[] for y in Y_hat]

        for sample, X_m, y, y_hat in zip(self.inference_cache_, X, Y, Y_hat):
            already_there = [self.constraint_equal(y_hat, cache[2])
                             for cache in sample]
            if np.any(already_there):
                continue
            if len(sample) > self.inference_cache:
                sample.pop(0)
            # we computed both of these before, but summed them up immediately
            # this makes it a little less efficient in the caching case.
            # the idea is that if we cache, inference is way more expensive
            # and this doesn't matter much.
            sample.append((self.model.joint_feature(X_m, y_hat),
                           self.model.loss(y, y_hat), y_hat))

    def _constraint_from_cache(self, X, Y, joint_feature_gt, constraints):
        if (not getattr(self, 'inference_cache_', False) or
                self.inference_cache_ is False):
            if self.verbose > 10:
                print("[INFO] Empty cache.")
            raise NoConstraint
        gap = self.primal_objective_curve_[-1] - self.objective_curve_[-1]
        if (self.cache_tol == 'auto' and gap < self.cache_tol_):
            # do inference if gap has become to small
            if self.verbose > 1:
                print("[INFO] Last gap too small (%.4f < %.4f), not loading constraint from cache."
                      % (gap, self.cache_tol_))
            raise NoConstraint

        Y_hat = []
        joint_feature_acc = np.zeros(self.model.size_joint_feature)
        loss_mean = 0
        for cached in self.inference_cache_:
            # cached has entries of form (joint_feature, loss, y_hat)
            violations = [np.dot(joint_feature, self.w) + loss
                          for joint_feature, loss, _ in cached]
            joint_feature, loss, y_hat = cached[np.argmax(violations)]
            Y_hat.append(y_hat)
            joint_feature_acc += joint_feature
            loss_mean += loss

        djoint_feature = (joint_feature_gt - joint_feature_acc) / len(X)
        loss_mean = loss_mean / len(X)

        violation = loss_mean - np.dot(self.w, djoint_feature)
        if self._check_bad_constraint(violation, djoint_feature, loss_mean, constraints,
                                      break_on_bad=False):
            if self.verbose > 1:
                print("[INFO] No constraint from cache!")
            raise NoConstraint
        return Y_hat, djoint_feature, loss_mean

    def _find_new_constraint(self, X, Y, joint_feature_gt, constraints, check=True):
        Y_hat = []
        joint_feature_acc = np.zeros(self.model.size_joint_feature)
        loss_mean = 0
        for m in range(len(Y)):
            y_hat = self.model.loss_augmented_inference(X[m], self.w, Y[m])
            joint_feature_acc += self.model.joint_feature(X[m], y_hat)
            loss_mean += self.model.loss(Y[m], y_hat)
            Y_hat.append(y_hat)

        # compute the mean over joint_features and losses
        djoint_feature = (joint_feature_gt - joint_feature_acc) / len(X)
        loss_mean = loss_mean / len(X)

        violation = loss_mean - np.dot(self.w, djoint_feature)
        if check and self._check_bad_constraint(
                violation, djoint_feature, loss_mean, constraints,
                break_on_bad=self.break_on_bad):
            raise NoConstraint
        return Y_hat, djoint_feature, loss_mean

    def fit(self, X, Y, constraints=None, warm_start=False, initialize=True):
        """Learn parameters using cutting plane method.
        Parameters
        ----------
        X iterable: 3darray, shape (M, N_m, n_features) / list of 2darray, shape (N_m, n_features)
            Training instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.
        Y iterable: 2darray, shape (M, N_m) / list of 1darray, shape (N_m,)
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.
        contraints : ignored
        warm_start : bool, default=False
            Whether we are warmstarting from a previous fit.
        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        if self.verbose:
            print("[INFO] Training 1-slack dual structural SVM...")
        cvxopt.solvers.options['show_progress'] = self.verbose > 3
        if initialize:
            self.model.initialize(X[0], Y[0])

        # parse cache_tol parameter
        if self.cache_tol is None or self.cache_tol == 'auto':
            self.cache_tol_ = self.tol
        else:
            self.cache_tol_ = self.cache_tol

        if not warm_start:
            self.w = np.zeros(self.model.size_joint_feature)
            constraints = []
            self.objective_curve_, self.primal_objective_curve_ = [], []
            self.cached_constraint_ = []
            self.alphas = []  # dual solutions
            # append constraint given by ground truth to make our life easier
            # constraints: [(joint_feature, loss), ...]
            constraints.append((np.zeros(self.model.size_joint_feature), 0))
            self.alphas.append([self.C])
            self.inference_cache_ = None
            self.timestamps_ = [time()]
        elif warm_start == "soft":
            self.w = np.zeros(self.model.size_joint_feature)
            constraints = []
            self.alphas = []  # dual solutions
            # append constraint given by ground truth to make our life easier
            constraints.append((np.zeros(self.model.size_joint_feature), 0))
            self.alphas.append([self.C])
        else:
            constraints = self.constraints_

        self.last_slack_ = -1

        # get the joint_feature of the ground truth
        joint_feature_gt = np.zeros(self.model.size_joint_feature)
        for m in range(len(Y)):
            joint_feature_gt += self.model.joint_feature(X[m], Y[m])
        
        try:
            # catch ctrl+c to stop training
            for iteration in range(self.max_iter):
                # main loop
                cached_constraint = False
                if self.verbose > 0:
                    print("[INFO] Iteration %d..." % iteration)
                if self.verbose > 2:
                    print(self)
                try:
                    Y_hat, djoint_feature, loss_mean = self._constraint_from_cache(
                        X, Y, joint_feature_gt, constraints)
                    cached_constraint = True
                except NoConstraint:
                    try:
                        Y_hat, djoint_feature, loss_mean = self._find_new_constraint(
                            X, Y, joint_feature_gt, constraints)
                        self._update_cache(X, Y, Y_hat)
                    except NoConstraint:
                        if self.verbose:
                            print("[INFO] No additional constraints!")
                        break

                self.timestamps_.append(time() - self.timestamps_[0])
                self._compute_training_loss(X, Y, iteration)
                constraints.append((djoint_feature, loss_mean))

                # compute primal objective
                last_slack = -np.dot(self.w, djoint_feature) + loss_mean
                primal_objective = (self.C * len(X) * max(last_slack, 0) + np.sum(self.w ** 2) / 2)
                self.primal_objective_curve_.append(primal_objective)
                self.cached_constraint_.append(cached_constraint)

                objective = self._solve_1_slack_qp(constraints, n_samples=len(X))

                # update cache tolerance if cache_tol is auto:
                if self.cache_tol == "auto" and not cached_constraint:
                    self.cache_tol_ = (primal_objective - objective) / 4

                self.last_slack_ = np.max([(-np.dot(self.w, djoint_feature) + loss_mean)
                                           for djoint_feature, loss_mean in constraints])
                self.last_slack_ = max(self.last_slack_, 0)

                if self.verbose > 0:
                    # the cutting plane objective can also be computed as
                    # self.C * len(X) * self.last_slack_ + np.sum(self.w**2)/2
                    print("[INFO] Cutting plane objective: %.4f, Primal objective: %.4f" % (objective, primal_objective))
                # we only do this here because we didn't add the gt to the
                # constraints, which makes the dual behave a bit oddly
                self.objective_curve_.append(objective)
                self.constraints_ = constraints
                if self.verbose > 5:
                    print(self.w)
        except KeyboardInterrupt:
            pass
        # compute final objective:
        self.timestamps_.append(time() - self.timestamps_[0])
        primal_objective = self._objective(X, Y)
        self.primal_objective_curve_.append(primal_objective)
        self.objective_curve_.append(objective)
        self.cached_constraint_.append(False)

        if self.verbose > 0:
            print("[INFO] Final primal objective: %.4f, Gap: %.4f" % (primal_objective, primal_objective - objective))
        return self