# ssvmKmeans
The implementation of structural SVM for clustering
- Coding Reference: [PyStruct](https://github.com/pystruct/pystruct) &nbsp; [superKmeans](https://www.cs.cornell.edu/~tomf/projects/supervisedkmeans/)
- Paper Reference: Supervised K-means Clustering, Thomas Finley 2007
#### Requirements
```
cvxopt
numpy
scikit-learn
scipy
```
## Learner
### One-slack SSVM
Cutting-Plane Training of Structural SVMs
```
trainer = OneSlackSSVM(model, C=1, n_jobs=1, verbose=1, show_loss_every=1)
```
#### Functions
**.fit(X, Y)**: X is iterable containing 2d training examples; Y is iterable containing 1d true labels
```
trainer.fit(X, Y)
```
**.predict(X)**: X is one 2d example
```
y_pred = trainer.predict(X)
```
**.scores(X, y)**: X is one 2d example; y is one 1d true labels. Return loss and discriminant function value
```
loss, discrim_fv = trainer.scores(X, y)
```
## Model
### Spectral
Compute a matrix's relaxed spectral clustering. This will return a matrix with the number of eigenvectors same as the number of clusters associated with the largest eigenvalues.
```
model = Spectral(n_clusters=3)
```
### Point-Incremental K-means
```
model = Kmeans(n_clusters=3)
```
### Discretized Spectral
It is a combination of the Spectral and Iterative Point-Incremental K-means.
```
model = DiscretizedSpectral(n_clusters=3)
```





