---
title: "Decision Tree Algorithms: A Mathematical Introduction"
date: 2026-06-17
description: "A detailed mathematical explanation of classification and regression decision trees."
author: "Noah Reef"
tags: ["machine-learning", "decision-trees", "statistics"]
---

Decision trees are supervised learning algorithms that approximate an unknown function by recursively partitioning the feature space into simple regions. Their appeal is that the fitted model is explicit: a prediction is made by following a finite sequence of feature tests until a leaf node is reached. Despite this simple representation, the training problem involves a rich set of statistical and computational tradeoffs.

Let the training data be

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n,
$$

where $x_i \in \mathcal{X} \subseteq \mathbb{R}^p$ is a feature vector and $y_i$ is either a class label for classification or a real value for regression. A decision tree partitions $\mathcal{X}$ into disjoint terminal regions $R_1,\dots,R_M$. The prediction function has the form

$$
\hat{f}(x) = \sum_{m=1}^M c_m \mathbf{1}\{x \in R_m\},
$$

where $c_m$ is the prediction stored at leaf $m$. For classification, $c_m$ is usually the majority class or a vector of estimated class probabilities. For regression, $c_m$ is usually the mean response in that leaf.

## Recursive Partitioning

The central algorithm is top-down recursive partitioning. Starting with all training points at the root node, the algorithm repeatedly chooses a feature and threshold that best improves a node impurity objective. For a node $t$, let

$$
I_t = \{i : x_i \in R_t\}
$$

be the index set of samples in that node, and let $N_t = |I_t|$. A binary split $s$ divides $I_t$ into left and right children:

$$
I_L(s) = \{i \in I_t : x_{ij} \leq a\},
\qquad
I_R(s) = \{i \in I_t : x_{ij} > a\},
$$

for some feature $j$ and threshold $a$. The split is chosen by maximizing impurity reduction:

$$
\Delta(s,t) =
i(t) -
\frac{N_L}{N_t} i(t_L) -
\frac{N_R}{N_t} i(t_R),
$$

where $i(t)$ is the impurity of node $t$. Equivalently, the algorithm minimizes the weighted child impurity. This greedy choice is local: the algorithm does not generally find the globally optimal tree, because optimizing tree structure exactly is computationally hard.

## Classification Trees

For classification with $K$ classes, define the empirical class probability at node $t$ as

$$
\hat{p}_{mk}(t) =
\frac{1}{N_t}\sum_{i \in I_t} \mathbf{1}\{y_i = k\}.
$$

The leaf prediction is

$$
\hat{c}(t) = \arg\max_{k \in \{1,\dots,K\}} \hat{p}_{mk}(t).
$$

The same probabilities can also be returned directly when probabilistic predictions are needed.

### Gini Impurity

The Gini impurity is

$$
i_G(t) = \sum_{k=1}^K \hat{p}_{k}(t)(1-\hat{p}_{k}(t))
= 1 - \sum_{k=1}^K \hat{p}_{k}(t)^2.
$$

It is the probability of misclassification if a label were sampled according to the empirical class distribution in the node and then compared with an independently sampled class from the same distribution. A pure node has Gini impurity $0$. For two classes, if $\hat{p}$ is the probability of class $1$, then

$$
i_G(t) = 2\hat{p}(1-\hat{p}).
$$

### Entropy and Information Gain

Entropy impurity is

$$
i_H(t) =
- \sum_{k=1}^K \hat{p}_k(t)\log_2 \hat{p}_k(t),
$$

with the convention $0\log 0 = 0$. The corresponding split score is information gain:

$$
IG(s,t) =
H(t) -
\frac{N_L}{N_t}H(t_L) -
\frac{N_R}{N_t}H(t_R).
$$

Entropy is largest when classes are uniformly mixed and smallest at pure nodes. Compared with Gini impurity, entropy penalizes highly mixed nodes more strongly, but in practice the two criteria often choose similar splits.

### Misclassification Error

The misclassification impurity is

$$
i_E(t) = 1 - \max_k \hat{p}_k(t).
$$

This is the training error rate that would be obtained by predicting the majority class in node $t$. It is intuitive but less sensitive to class-probability changes than Gini impurity or entropy. Because it can be flat across many candidate splits, it is more commonly used for pruning or evaluation than for growing the tree.

## Regression Trees

For regression, a leaf usually predicts the sample mean in that leaf:

$$
\hat{c}(t) =
\frac{1}{N_t}\sum_{i \in I_t} y_i.
$$

The standard node impurity is the within-node sum of squared errors or its normalized version, the mean squared error:

$$
i_{MSE}(t) =
\frac{1}{N_t}\sum_{i \in I_t}(y_i - \bar{y}_t)^2,
\qquad
\bar{y}_t = \frac{1}{N_t}\sum_{i \in I_t} y_i.
$$

The split objective is variance reduction:

$$
\Delta_{reg}(s,t) =
i_{MSE}(t)
-
\frac{N_L}{N_t}i_{MSE}(t_L)
-
\frac{N_R}{N_t}i_{MSE}(t_R).
$$

This is equivalent to choosing the split that most reduces squared prediction error within the node. Other regression criteria are possible, such as mean absolute error, which predicts the median instead of the mean and can be more robust to outliers.

## Split Selection

For continuous features, candidate thresholds are usually considered between sorted unique values. If feature $j$ has sorted values $v_{(1)} \leq \dots \leq v_{(m)}$ in node $t$, candidate thresholds are often midpoints

$$
a_r = \frac{v_{(r)} + v_{(r+1)}}{2}.
$$

The best split is

$$
s^\* = \arg\max_{s \in \mathcal{S}_t} \Delta(s,t),
$$

where $\mathcal{S}_t$ is the set of valid feature-threshold splits at node $t$. Efficient implementations avoid recomputing impurities from scratch for every threshold by sorting feature values and updating sufficient statistics as the split point moves.

Categorical variables can be handled by one-hot encoding, ordinal encoding, or native categorical split search. One-hot encoding is simple but can fragment high-cardinality categorical variables across many binary decisions. Native categorical splits may test subsets of categories, but exhaustive subset search is expensive when cardinality is large.

Missing values require an explicit policy. Common options include imputation before training, routing missing values to a learned default branch, or using surrogate splits that approximate the primary split using another feature. The best choice depends on whether missingness is random or carries signal.

## Stopping Criteria

If allowed to grow without constraints, a tree can keep splitting until leaves are pure or contain only a few observations. Practical training uses stopping rules such as:

- maximum tree depth,
- minimum samples required to split an internal node,
- minimum samples required at a leaf,
- minimum impurity decrease required for a split,
- maximum number of leaf nodes,
- pure-node termination.

These criteria regularize the tree by limiting how finely the feature space can be partitioned. They also reduce training and prediction cost.

## Cost-Complexity Pruning

Another strategy is to first grow a large tree $T_0$ and then prune it. Cost-complexity pruning defines an objective

$$
R_\alpha(T) = R(T) + \alpha |T|,
$$

where $R(T)$ is the empirical risk of tree $T$, $|T|$ is the number of terminal leaves, and $\alpha \geq 0$ controls the penalty for complexity. In classification, $R(T)$ may be measured using leaf misclassification error; in regression, it is often residual sum of squares.

For a subtree $T_t$ rooted at internal node $t$, compare the risk of keeping the subtree with the risk of replacing it by a single leaf. The effective complexity parameter is

$$
\alpha(t) =
\frac{R(t) - R(T_t)}
{|T_t| - 1}.
$$

Weakest-link pruning repeatedly removes the subtree with the smallest $\alpha(t)$, producing a nested sequence of candidate trees. Cross-validation is then used to choose the value of $\alpha$ that gives the best validation performance, or a slightly simpler tree within one standard error of the best.

## Overfitting and Bias-Variance Behavior

Decision trees have low bias and high variance when grown deeply. A small change in the training data can alter an early split, causing a different downstream tree structure. This instability is one reason ensembles such as random forests and gradient-boosted trees are so effective: they reduce variance or bias by combining many trees.

Shallow trees have higher bias because they can only represent coarse partitions of the feature space. Deep trees have lower training error but can memorize noise, especially in small data sets or in settings with many irrelevant features. Regularization through depth limits, minimum leaf sizes, pruning, and feature engineering is therefore central to good performance.

## Interpretability

Decision trees are often described as interpretable because each prediction follows a visible path of feature tests. This is true for small trees. A tree with a few levels can be inspected directly, converted into business rules, and debugged with domain experts.

Interpretability decreases quickly as depth and feature count increase. Large trees may contain hundreds of leaves, repeated splits on correlated features, and brittle thresholds that are hard to explain. Feature importance scores can help summarize the tree, but impurity-based importances are biased toward variables with many possible split points and should be interpreted carefully.

## Computational Complexity

Let $n$ be the number of samples, $p$ the number of features, and $d$ the final tree depth. A naive split search can be expensive because every node may evaluate many thresholds for many features. With sorted continuous features, training cost is often approximated as

$$
O(pn\log n)
$$

for balanced trees, though exact cost depends on implementation details, stopping rules, sparsity, and how sorted orders are maintained. Prediction is cheap: one sample follows one path from root to leaf, so prediction cost is

$$
O(d)
$$

per sample.

Memory usage is proportional to the number of nodes, which is at most linear in the number of leaves. A full binary tree with $M$ leaves has $2M-1$ total nodes.

## Practical Limitations

Decision trees handle nonlinear interactions naturally, require little feature scaling, and can mix numeric and encoded categorical variables. However, they also have important limitations:

- They are unstable with respect to small data perturbations.
- Axis-aligned splits can approximate diagonal or smooth boundaries only with many rectangular regions.
- Deep trees often overfit unless regularized.
- Greedy split selection can miss globally better tree structures.
- Probability estimates in leaves can be poorly calibrated when leaves contain few samples.
- High-cardinality features can dominate impurity-based splitting.
- Extrapolation is weak in regression because predictions are piecewise constant.

These limitations do not make trees obsolete. They explain why single trees are best used when transparency matters, as baselines, or as components inside stronger ensemble methods.

## Python Example with scikit-learn

The following script trains and evaluates a classification tree on the built-in breast cancer data set from scikit-learn. It reports accuracy, a classification report, the confusion matrix, and a text representation of the learned tree.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


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

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    print("Tree structure:")
    print(
        export_text(
            tree,
            feature_names=list(data.feature_names),
            max_depth=4,
        )
    )


if __name__ == "__main__":
    main()
```

The depth and minimum-leaf constraints regularize the model. Without them, the tree could grow highly specific branches that improve training accuracy while hurting test-set generalization.

To visualize the fitted tree instead of printing it as text, add:

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(18, 10))
plot_tree(
    tree,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
)
plt.tight_layout()
plt.show()
```

For regression, the same workflow uses `DecisionTreeRegressor`, a continuous target, and metrics such as mean squared error or mean absolute error:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

regressor = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R^2: {r2_score(y_test, y_pred):.3f}")
```

The regression tree still partitions the feature space recursively, but each leaf stores a real-valued prediction rather than a class distribution.

## Summary

A decision tree is a piecewise-constant estimator built by recursively selecting feature-threshold splits that reduce impurity. Classification trees commonly use Gini impurity, entropy, or misclassification error; regression trees commonly use variance or MSE reduction. The same structure that makes trees interpretable also makes them unstable and prone to overfitting, so stopping criteria and pruning are essential. Used carefully, a single decision tree is a transparent and useful model; used inside ensembles, trees become the foundation for some of the most powerful practical machine learning methods.
