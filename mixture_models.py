from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None, return_mean = False):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    # TODO: finish this function
    # Flatten image is slower than reshape. So used reshape
    if len(image_values.shape) == 3:
        height, width, channel = image_values.shape
    else:
        height, width = image_values.shape
        channel = 1
    #start = time.time()
    #flattened_image = flatten_image_matrix(image_values)
    #end = time.time()
    #print(end - start)
    #start = time.time()
    flattened_image = image_values.reshape(-1,channel)
    if initial_means is None:
        random_elements = np.random.choice(len(flattened_image), k, replace=False)
        initial_means = flattened_image[random_elements]

    dist = np.sqrt(((flattened_image - initial_means[:, np.newaxis]) ** 2).sum(axis=2))
    ind = np.argmin(dist, axis=0)
    old_means = None
    while not np.array_equal(initial_means, old_means):
        old_means = initial_means
        curr_means = np.zeros((initial_means.shape))
        for rnk in range(k):
            cluster_mean = flattened_image[ind == rnk].mean(axis=0)
            curr_means[rnk] = cluster_mean
        initial_means = np.array(curr_means)
        dist = np.sqrt(((flattened_image - initial_means[:, np.newaxis]) ** 2).sum(axis=2))
        ind = np.argmin(dist, axis=0)

    image_values_out = np.zeros(flattened_image.shape)
    for i, mean in enumerate(initial_means):
        image_values_out[ind == i] = mean
    if return_mean:
        return initial_means.reshape(3)
    else:
        return image_values_out.reshape(-1, width, channel)

def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)
        if len(image_matrix.shape) == 3:
            self.flattened_image = image_matrix.reshape(-1,image_matrix.shape[2])
        else:
            self.flattened_image = image_matrix.reshape(-1,1)

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        # TODO: finish this
        var = np.array(self.variances)
        means = np.array(self.means)
        first_term = -0.5 * np.log(2 * np.pi * var)
        prob = first_term - (((val - means) ** 2) / (2 * var))

        return logsumexp(prob, b=self.mixing_coefficients)

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        # TODO: finish this
        ind = np.random.choice(len(self.flattened_image), self.num_components,  replace=False)
        self.means = self.flattened_image[ind].reshape(len(self.means))
        self.variances = np.array([1] * self.num_components)
        self.mixing_coefficients = np.array([1.0 * 1 / self.num_components] * self.num_components)

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        # TODO: finish this
        convergence = False
        conv_ctr = 0
        prev_likelihood = self.likelihood()
        while not convergence:
            numerator = self.e_step_numerator()
            denominator = np.sum(numerator, axis=1).reshape(numerator.shape[0], 1)
            gamma = numerator / denominator
            n_k = gamma.sum(axis=0)
            self.means = np.sum(gamma * self.flattened_image, axis=0) / n_k
            self.variances = np.sum(gamma * ((self.flattened_image - self.means)**2), axis=0) / n_k
            self.mixing_coefficients = n_k / self.flattened_image.shape[0]
            curr_likelihood = self.likelihood()
            conv_ctr, convergence = convergence_function(prev_likelihood, curr_likelihood, conv_ctr)
            prev_likelihood = curr_likelihood


    def e_step_numerator(self):

        return (self.mixing_coefficients / np.sqrt(self.variances * 2 * np.pi)) * np.exp(-((self.flattened_image - self.means) ** 2) / (2 * self.variances))

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        numerator = self.e_step_numerator()
        denominator = np.sum(numerator, axis=1).reshape(numerator.shape[0], 1)
        gamma = numerator / denominator
        ind = np.argmax(gamma * self.flattened_image, axis=1)

        image_matrix_out = np.zeros(self.flattened_image.shape)
        for i, mean in enumerate(self.means):
            image_matrix_out[ind == i] = mean

        return image_matrix_out.reshape(self.image_matrix.shape)

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        # TODO: finish this
        # Calling joint_prob was slower
        #start = time.time()
        #joint_p = []
        #for i in self.flattened_image:
        #    joint_p.append(self.joint_prob(i))
        #joint_p = np.array(joint_p).reshape(1000,1)
        #end = time.time()
        #print(end - start)

        #arr1 = []
        #for mean, var, mc in zip(self.means, self.variances, self.mixing_coefficients):
        #    arr1.append((mc / np.sqrt(var * 2 * np.pi)) * np.exp(-((self.flattened_image - mean) ** 2) / (2 * var)))
        arr = self.e_step_numerator()
        log_likelihood = np.sum(np.log(np.sum(arr, 1)))


        return log_likelihood

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        best_likelihood = self.likelihood()
        best_segment = self.segment()
        best_mean = 0
        best_var = 0
        best_mc = 0
        for i in xrange(iters):
            self.initialize_training()
            self.train_model()
            likelihood = self.likelihood()
            if best_likelihood < likelihood:
                best_likelihood = likelihood
                best_segment = self.segment()
                best_mean = self.means
                best_var = self.variances
                best_mc = self.mixing_coefficients

        self.means = best_mean
        self.variances = best_var
        self.mixing_coefficients = best_mc
        return best_segment


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        # TODO: finish this
        self.means = k_means_cluster(self.image_matrix, self.num_components, None, True)
        self.variances = np.array([1] * self.num_components)
        self.mixing_coefficients = np.array([1.0 * 1 / self.num_components] * self.num_components)


def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    #increase_convergence_ctr = False
    count = 0
    for prev, new in zip(previous_variables, new_variables):
        #if abs(prev.all()) * 0.9 < abs(new.all())  < abs(prev.all()) * 1.1:
        if abs(prev[0]) * 0.9 < abs(new[0]) < abs(prev[0]) * 1.1\
                and abs(prev[1]) * 0.9 < abs(new[1]) < abs(prev[1]) * 1.1\
                and abs(prev[2]) * 0.9 < abs(new[2]) < abs(prev[2]) * 1.1:
            count += 1

    if count == 3:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function
        convergence = False
        conv_ctr = 0
        prev_likelihood = [self.means, self.variances, self.mixing_coefficients]
        while not convergence:
            numerator = self.e_step_numerator()
            denominator = np.sum(numerator, axis=1).reshape(numerator.shape[0], 1)
            gamma = numerator / denominator
            n_k = gamma.sum(axis=0)
            self.means = np.sum(gamma * self.flattened_image, axis=0) / n_k
            self.variances = np.sum(gamma * ((self.flattened_image - self.means) ** 2), axis=0) / n_k
            self.mixing_coefficients = n_k / self.flattened_image.shape[0]
            curr_likelihood = [self.means, self.variances, self.mixing_coefficients]
            conv_ctr, convergence = convergence_function(prev_likelihood, curr_likelihood, conv_ctr)
            prev_likelihood = curr_likelihood


def bayes_info_criterion(gmm):
    # TODO: finish this function
    return (np.log(gmm.image_matrix.size) * (3 * gmm.num_components)) - (2 * gmm.likelihood())


def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    """
    # TODO: finish this method
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    max_likelihood = float('-inf')
    min_BIC = float('inf')
    max_likelihood_model = None
    min_BIC_model = None

    for i, num_components in enumerate(xrange(2, 8)):
        model = GaussianMixtureModel(image_matrix, num_components)
        model.initialize_training()
        model.means = comp_means[i]
        model.train_model()
        likelihood = model.likelihood()
        if max_likelihood < likelihood:
            max_likelihood = likelihood
            max_likelihood_model = model
            #max_component = num_components
        BIC = bayes_info_criterion(model)
        if BIC < min_BIC:
            min_BIC = BIC
            min_BIC_model = model
            #bic_component = num_components
    #print 'Max', max_component, 'Bic', bic_component
    return min_BIC_model, max_likelihood_model


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    raise NotImplementedError()
    bic = 7
    likelihood = 7
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name
    # TODO: finish this
    return 'Sridhar Sampath'

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """
    # TODO: fill in the bonus function
    # REMOVE THE LINE BELOW IF ATTEMPTING BONUS
    dists = np.sqrt(((points_array[:,np.newaxis] - means_array) ** 2).sum(axis=2))
    return dists