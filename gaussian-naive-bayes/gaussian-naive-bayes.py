import math

def gaussian_probability(x, mean, variance):
    epsilon = 1e-9
    variance += epsilon
    exponent = math.exp(-(x - mean)**2 / (2 * variance))

    return (1 / math.sqrt(2 * math.pi * variance)) * exponent


def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """

    classes = {}

    for features, label in zip(X_train, y_train):
        if label not in classes:
            classes[label] = []

        classes[label].append(features)

    means = {}
    for label, samples in classes.items():
        means[label] = []

        num_features = len(samples[0])

        for j in range(num_features):
            avg = sum(sample[j] for sample in samples) / len(samples)
            means[label].append(avg)

    variances = {}
    for label, samples in classes.items():
        variances[label] = []

        num_features = len(samples[0])

        for j in range(num_features):
            mean = means[label][j]

            variance = sum((sample[j] - mean) ** 2 for sample in samples) / len(samples)
            variances[label].append(variance)

    priors = {}
    for label, samples in classes.items():
        priors[label] = len(samples) / len(X_train)

    predictions = []
    for sample in X_test:
        class_scores = {}

        for label in classes:
            score = math.log(priors[label])

            for j in range(len(sample)):
                probability = gaussian_probability(
                    sample[j],
                    means[label][j],
                    variances[label][j]
                )

                score += math.log(max(probability, 1e-300))

            class_scores[label] = score
        predicted = max(class_scores, key=class_scores.get)
        predictions.append(predicted)
    return predictions