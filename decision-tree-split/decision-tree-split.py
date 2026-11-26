def gini_impurity(labels):
    total = len(labels)
    sum_of_squared_probabilities = 0

    for class_name in set(labels):
        particular_class = labels.count(class_name)
        prob = particular_class / total
        sum_of_squared_probabilities += prob ** 2

    return 1 - sum_of_squared_probabilities


def get_threshold(feature):
    unique_values = sorted(set(feature))
    threshold = []

    for i in range(len(unique_values) - 1):
        left = unique_values[i]
        right = unique_values[i + 1]
        midpoint = (left + right) / 2
        threshold.append(midpoint)

    return threshold


def weighted_gini(labels_left, labels_right):
    total = len(labels_left) + len(labels_right)

    left_weight = len(labels_left) / total
    right_weight = len(labels_right) / total

    left_gini = gini_impurity(labels_left)
    right_gini = gini_impurity(labels_right)

    return (left_weight * left_gini) + (right_weight * right_gini)


def decision_tree_split(X, y):
    best_gini = float("inf")
    best_feature = None
    best_threshold = None

    for feature_idx in range(len(X[0])):
        feature = [row[feature_idx] for row in X]
        thresholds = get_threshold(feature)

        for threshold in thresholds:
            left = [
                label for value, label in zip(feature, y)
                if value <= threshold
            ]

            right = [
                label for value, label in zip(feature, y)
                if value > threshold
            ]

            current_gini = weighted_gini(left, right)

            if current_gini < best_gini:
                best_gini = current_gini
                best_feature = feature_idx
                best_threshold = threshold

    return [best_feature, best_threshold]