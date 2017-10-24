from __future__ import division

import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None

    # TODO: finish this.
    decision_tree_root = DecisionNode(None, None, lambda feature: feature[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)

    decision_tree_root.right = DecisionNode(None, None, lambda feature: feature[2] == 1)
    decision_tree_root.right.left = DecisionNode(None, None, lambda feature: feature[3] == 1)
    decision_tree_root.right.left.left = DecisionNode(None, None, None, 1)
    decision_tree_root.right.left.right = DecisionNode(None, None, None, 0)

    decision_tree_root.right.right = DecisionNode(None, None, lambda feature: feature[3] == 1)
    decision_tree_root.right.right.left = DecisionNode(None, None, None, 0)
    decision_tree_root.right.right.right = DecisionNode(None, None, None, 1)

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0
    for index, output in enumerate(classifier_output):
        label = true_labels[index]
        if output and label:
            true_positive += 1
        elif not output and not label:
            true_negative += 1
        elif not output and label:
            false_negative += 1
        else:
            false_positive += 1

    return [[true_positive, false_negative], [false_positive, true_negative]]


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    cm = confusion_matrix(classifier_output, true_labels)
    return cm[0][0] / (cm[0][0] + cm[1][0])


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    cm = confusion_matrix(classifier_output, true_labels)
    return cm[0][0] / (cm[0][0] + cm[0][1])


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    cm = confusion_matrix(classifier_output, true_labels)
    return (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    if len(class_vector) == 0:
        return 0.
    prob = {}
    gini_square = 0
    for c in set(class_vector):
        prob[c] = 1.0 * class_vector.count(c) / len(class_vector)
        gini_square += prob[c] ** 2

    gini_i = 1 - gini_square
    return gini_i


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    
    total = 0
    count = 0
    for c in current_classes:
        #print c, gini_impurity(c)
        total +=  gini_impurity(c) * len(c) / len(previous_classes)
        count += 1
    gini_prev = gini_impurity(previous_classes)
	#print "gini_prev", gini_prev
    gain = gini_prev - total
    return gain	


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        use_median = False
        if len(set(classes)) == 1:
            return DecisionNode(None, None, None, classes[0])
        elif depth >= self.depth_limit:
            most_freq_class = max(set(classes), key=classes.count)
            return DecisionNode(None, None, None, most_freq_class)
        else:
            depth += 1
            alpha_gain_best = float('-inf')
            alpha_best = -1
            alpha_threshold = -1
            for index in range(0, features.shape[1]):
                alpha = np.array(features[:, index])
                #alpha_sorted = sorted(alpha)
                if len(set(alpha))==1:
                    most_freq_class = max(set(classes), key=classes.count)
                    return DecisionNode(None, None, None, most_freq_class)
                if use_median:
                    threshold = np.median(alpha)
                else:
                    threshold = np.mean(alpha)
                curr_classes = [[],[]]
                classes_ = np.array(classes)
                curr_classes[0] = classes_[np.where(alpha < threshold)].tolist()
                curr_classes[1] = classes_[np.where(alpha >= threshold)].tolist()
                alpha_gain = gini_gain(classes, curr_classes)
                if alpha_gain > alpha_gain_best:
                    alpha_best = index
                    alpha_gain_best = alpha_gain
                    alpha_threshold = threshold

            node = DecisionNode(None, None, lambda feature: feature[alpha_best] < alpha_threshold)
            
            left_ind = np.where(features[:,alpha_best] < alpha_threshold)
            left_features = features[left_ind]
            left_classes = np.array(classes)[left_ind].tolist()
            node.left = self.__build_tree__(left_features, left_classes, depth)
            
            right_ind = np.where(features[:,alpha_best] >= alpha_threshold)
            right_features = features[right_ind]
            right_classes = np.array(classes)[right_ind].tolist()
            node.right = self.__build_tree__(right_features, right_classes, depth)
            
            return node

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """

        class_labels = []

        # TODO: finish this.
        class_labels = [self.root.decide(feature) for feature in features]
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """

    # TODO: finish this.
    feature, label = dataset
    label_ = np.reshape(label, (-1, 1))
    data = np.append(feature, label_, 1)
    sample_size = data.shape[0] // k
    folds = []
    np.random.shuffle(data)
    for i in range(k):
        test_set = data[0:sample_size]
        training_set = data[sample_size:]
        traning_data = (training_set[:, 0:-1], training_set[:, -1])
        test_data = (test_set[:, 0:-1], test_set[:, -1])
        folds.append((traning_data, test_data))
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """

        # TODO: finish this.
        raise NotImplemented()


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        raise NotImplemented()

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        raise NotImplemented()

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        raise NotImplemented()
        
def return_your_name():
    # return your name
    # TODO: finish this
    raise NotImplemented()
