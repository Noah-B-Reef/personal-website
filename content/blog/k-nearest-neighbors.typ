#import "@tola/source:0.0.0": tola-meta

#tola-meta((
  title: "K-Nearest Neighbors: A Mathematical Introduction",
  date: "2026-06-17",
  summary: "A detailed mathematical introduction to k-nearest neighbors for classification and regression.",
  tags: ("machine-learning", "classification", "regression", "statistics",),
))

The k-nearest neighbors (KNN) algorithm is one of the simplest supervised learning methods: to predict the output for a new point, find the $k$ training examples closest to it and aggregate their labels. That local rule is easy to state, but its behavior depends sharply on the geometry of the feature space, the choice of distance, the value of $k$, and the way the neighbors are aggregated.

= Problem Setup
<problem-setup>
Let the training data be

$ cal(D)_n = {\(x_i\,y_i\)}_(i = 1)^n\, $

where $x_i in bb(R)^d$ is a feature vector. For classification, $y_i in cal(C) = { 1\,dots.h\,C }$ is a class label. For regression, $y_i in bb(R)$, or more generally $y_i in bb(R)^m$.

Given a query point $x in bb(R)^d$, define a distance function $d\(x\,x_i\)$ and sort the training indices by increasing distance:

$ d\(x\,x_(\(1\))\)lt.eq d\(x\,x_(\(2\))\)lt.eq dots.h.c lt.eq d\(x\,x_(\(n\))\). $

The neighbor set is

$ N_k\(x\)= {\(1\)\,\(2\)\,dots.h\,\(k\)} . $

KNN is a non-parametric method: it does not learn a fixed-size coefficient vector during training. Instead, "training" mostly means storing the data, and prediction performs the local computation.

= Classification
<classification>
For unweighted KNN classification, the predicted class is the plurality vote among the $k$ nearest labels:

$ hat(y)\(x\)= arg max_(c in cal(C)) sum_(j in N_k\(x\)) upright(bold(1)) { y_j = c } . $

This estimates the class posterior probability locally by

$ hat(P)\(Y = c divides X = x\)= 1 / k sum_(j in N_k\(x\)) upright(bold(1)) { y_j = c } . $

The classifier then chooses the empirical maximum a posteriori class:

$ hat(y)\(x\)= arg max_(c in cal(C)) hat(P)\(Y = c divides X = x\). $

For binary classification with labels $y_i in { 0\,1 }$, this reduces to

$ hat(y)\(x\)= upright(bold(1)) {1 / k sum_(j in N_k\(x\)) y_j gt.eq 1 / 2}\, $

with a tie rule needed when the average is exactly $1\/2$.

= Regression
<regression>
For unweighted KNN regression, the prediction is the local average:

$ hat(f)\(x\)= 1 / k sum_(j in N_k\(x\)) y_j . $

This can be interpreted as a piecewise-constant estimate of the regression function

$ f\(x\)= bb(E)\[Y divides X = x\]. $

For vector-valued targets, the same formula applies componentwise:

$ hat(f)\(x\)= 1 / k sum_(j in N_k\(x\)) y_j in bb(R)^m . $

= Distance Metrics
<distance-metrics>
The distance function defines what "near" means. Common choices include:

== Minkowski Distances
<minkowski-distances>
For $p gt.eq 1$, the Minkowski distance is

$ d_p\(x\,z\)= (sum_(ell = 1)^d \| x_ell - z_ell \|^p)^(1\/p) . $

Important special cases are:

$ d_1\(x\,z\)= sum_(ell = 1)^d\|x_ell - z_ell\| $

for Manhattan distance, and

$ d_2\(x\,z\)= sqrt(sum_(ell = 1)^d\(x_ell - z_ell\)^2) $

for Euclidean distance. As $p arrow.r oo$, the metric becomes Chebyshev distance:

$ d_oo\(x\,z\)= max_ell\|x_ell - z_ell\|. $

== Mahalanobis Distance
<mahalanobis-distance>
If features are correlated or have different variances, a covariance-aware metric can be useful:

$ d_M\(x\,z\)= sqrt(\(x - z\)^top Sigma^(- 1)\(x - z\)) . $

Here $Sigma$ is a positive definite covariance matrix. When $Sigma$ is diagonal, this rescales each coordinate by its variance. When $Sigma$ is full, it also accounts for feature correlation.

== Categorical and Mixed Features
<categorical-and-mixed-features>
For categorical inputs, Hamming distance is often used:

$ d_H\(x\,z\)= sum_(ell = 1)^d upright(bold(1)) { x_ell eq.not z_ell } . $

Real datasets often contain both continuous and categorical variables. In that case, the metric should be chosen deliberately, often after encoding categorical variables and scaling continuous variables so that no feature family dominates simply because of units.

= Feature Scaling
<feature-scaling>
KNN is distance-based, so feature scaling is not optional in most real applications. If one feature is measured in dollars and another in percentages, the dollar-valued coordinate may dominate Euclidean distance.

A common standardization is

$ x'_(i ell) = frac(x_(i ell) - mu_ell, s_ell)\, $

where $mu_ell$ and $s_ell$ are the training-set mean and standard deviation of feature $ell$. Min-max scaling is another option:

$ x'_(i ell) = frac(x_(i ell) - min_j x_(j ell), max_j x_(j ell) - min_j x_(j ell)) . $

Scaling parameters must be fit on the training data only, then applied to validation and test data. Fitting a scaler before the train/test split leaks information from the test set into the model selection process.

= Weighted KNN
<weighted-knn>
Instead of giving every neighbor equal influence, weighted KNN assigns larger weights to closer points. Let

$ w_j\(x\)= K\(d\(x\,x_j\)\)\, $

where $K$ is a non-increasing kernel. A common inverse-distance weight is

$ w_j\(x\)= frac(1, \(d\(x\,x_j\)+ epsilon.alt\)^q)\, $

with $epsilon.alt > 0$ to avoid division by zero and $q > 0$ controlling how quickly influence decays.

Weighted classification predicts

$ hat(y)\(x\)= arg max_(c in cal(C)) sum_(j in N_k\(x\)) w_j\(x\)upright(bold(1)) { y_j = c } . $

Weighted regression predicts

$ hat(f)\(x\)= frac(sum_(j in N_k\(x\)) w_j\(x\)y_j, sum_(j in N_k\(x\)) w_j\(x\)) . $

If a query point exactly matches one or more training points, many implementations give those zero-distance neighbors all the weight and ignore nonzero-distance neighbors.

= Tie Handling
<tie-handling>
Ties can occur in two places:

+ Distance ties: multiple training points have the same distance at the $k$th boundary.
+ Vote ties: two or more classes receive the same winning vote total.

Distance ties are especially common with discrete or low-precision features. Possible policies include including all tied boundary points, selecting a deterministic subset by index order, selecting randomly with a fixed seed, or using distance weights to reduce the chance that equally counted classes tie.

Vote ties should also be resolved explicitly. Common policies are choosing the class with the smallest label, choosing the class whose nearest member is closest to the query, using prior class probabilities, or decreasing/increasing $k$. Deterministic tie handling is important for reproducibility.

= Choosing $k$
<choosing-k>
The parameter $k$ controls locality. With $k = 1$, the classifier memorizes the training labels exactly: every training point is classified correctly unless duplicate feature vectors have conflicting labels. As $k$ grows, the decision rule becomes smoother.

The best $k$ is usually selected by validation or cross-validation:

$ hat(k) = arg min_(k in cal(K)) hat(R)_(v a l)\(k\)\, $

where $hat(R)_(v a l)$ is validation error for classification or validation loss for regression.

For binary classification, odd values of $k$ are often used to reduce vote ties, though this does not eliminate ties in multiclass problems or in weighted variants.

= Bias and Variance
<bias-and-variance>
KNN has a clear bias-variance tradeoff:

- Small $k$: low bias, high variance. The estimate reacts strongly to individual samples and noise.
- Large $k$: higher bias, lower variance. The estimate averages over a wider neighborhood and may wash out local structure.

For regression, suppose

$ Y = f\(X\)+ epsilon\, $

where $bb(E)\[epsilon divides X\]= 0$ and $"Var"\(epsilon divides X\)= sigma^2$. A small neighborhood can approximate $f\(x\)$ well when $f$ is smooth, but the average of few noisy labels has high variance. Increasing $k$ reduces the noise variance roughly like $sigma^2\/k$, but it increases bias because the neighbors may come from points where $f\(x_j\)$ differs meaningfully from $f\(x\)$.

In the large-sample limit, KNN classification is consistent under regularity conditions when

$ k arrow.r oo quad upright("and") quad k / n arrow.r 0 quad upright("as") quad n arrow.r oo . $

The first condition reduces variance; the second keeps the neighborhood local enough to control bias.

= Computational Complexity
<computational-complexity>
Naive KNN prediction compares a query point to every training point.

For $n$ training samples, $d$ features, and $m$ query points:

- Training memory: $O\(n d\)$.
- Training time: $O\(n d\)$ to store or copy the data.
- Naive prediction time: $O\(m n d\)$ for distance computations, plus sorting or selection.
- Full sorting per query: $O\(n log n\)$.
- Selecting only the $k$ smallest distances: $O\(n\)$ expected time per query after distances are computed.

Spatial indexes such as KD-trees and ball trees can accelerate neighbor search in low or moderate dimensions. In high dimensions, their advantage often collapses because distances concentrate and pruning becomes ineffective. Approximate nearest-neighbor methods can trade exactness for speed at large scale.

= Practical Limitations
<practical-limitations>
KNN is useful because it is simple, flexible, and has few assumptions. Its limitations are just as important:

- It can be slow at prediction time because the training set remains part of the model.
- It is sensitive to feature scaling, irrelevant variables, and noisy dimensions.
- It struggles in high-dimensional spaces because neighborhoods become sparse.
- It can perform poorly on imbalanced classification unless weighting, resampling, or class-aware metrics are used.
- It does not provide an interpretable global model like a linear regression coefficient vector or a decision tree.
- It requires careful handling of missing values, categorical variables, and duplicate points.

KNN is often strongest as a baseline, a local interpolation method, or a component inside larger systems. It is rarely the most scalable model, but it is an excellent way to reason about geometry, neighborhoods, and non-parametric learning.

= Python Example with scikit-learn
<python-example-with-scikit-learn>
The following script trains and evaluates a KNN classifier on the built-in breast cancer dataset from scikit-learn. The pipeline standardizes features before fitting KNN, which is necessary because the dataset's measurements have different units and scales.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))


if __name__ == "__main__":
    main()
```

The `Pipeline` ensures that `StandardScaler` is fit only on the training fold and then applied consistently to the test set. `KNeighborsClassifier(n_neighbors=7)` uses seven neighbors; `weights="distance"` gives closer neighbors more influence; `metric="minkowski", p=2` selects Euclidean distance. The evaluation reports accuracy, the confusion matrix, and precision/recall/F1 for each class.
