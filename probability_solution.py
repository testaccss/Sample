"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''


from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    # TODO: finish this function    
    A_node = BayesNode(0, 2, name="alarm")
    F_A_node = BayesNode(1, 2, name="faulty alarm")
    G_node = BayesNode(2, 2, name="gauge")
    F_G_node = BayesNode(3, 2, name="faulty gauge")
    T_node = BayesNode(4, 2, name="temperature")

    T_node.add_child(G_node)
    G_node.add_parent(T_node)
    T_node.add_child(F_G_node)
    F_G_node.add_parent(T_node)
    G_node.add_parent(F_G_node)
    F_G_node.add_child(G_node)

    F_A_node.add_child(A_node)
    A_node.add_parent(F_A_node)

    G_node.add_child(A_node)
    A_node.add_parent(G_node)

    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    return BayesNet(nodes)

def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")

    # TODO: set the probability distribution for each node

    # Temperature
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([],[])
    # The temperature is hot (call this "true") 20% of the time.
    T_distribution[index] = [0.80,0.20]
    T_node.set_dist(T_distribution)

    # Faulty alarm
    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([],[])
    # The alarm is faulty 15% of the time.
    F_A_distribution[index] = [0.85,0.15]
    F_A_node.set_dist(F_A_distribution)

    # Faulty gauge
    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)
    # When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
    dist[0,:] = [0.95, 0.05]
    dist[1,:] = [0.20, 0.80]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node,F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)

    # Guage
    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    # The gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty.
    dist[0,0,:] = [0.95, 0.05]
    dist[0,1,:] = [0.2, 0.8]
    dist[1,0,:] = [0.05, 0.95]
    dist[1,1,:] = [0.80, 0.20]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(G_distribution)

    # Alarm
    dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    # The alarm responds correctly to the gauge 55% of the time when the alarm is faulty,
    # and it responds correctly to the gauge 90% of the time when the alarm is not faulty.
    dist[0,0,:] = [0.90, 0.10]
    dist[0,1,:] = [0.55, 0.45]
    dist[1,0,:] = [0.10, 0.90]
    dist[1,1,:] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    # Test the network
    #  For instance, running inference on P(T=true) should return 0.19999994 (i.e. almost 20%).
    #prob = check_temp_prob(bayes_net, 1)
    #prob = get_temperature_prob(bayes_net, 1)

    return bayes_net

def check_temp_prob(bayes_net, temp):
    """Calculate the marginal
    probability of the alarm
    ringing (T/F) in the
    power plant system."""
    # TODO: finish this function

    T_node = bayes_net.get_node_by_name("temperature")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp], range(Q.nDims))
    temp_prob = Q[index]

    return temp_prob

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    # TODO: finish this function

    A_node = bayes_net.get_node_by_name("alarm")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings], range(Q.nDims))
    alarm_prob = Q[index]

    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    # TOOD: finish this function

    G_node = bayes_net.get_node_by_name("gauge")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot], range(Q.nDims))
    gauge_prob = Q[index]

    return gauge_prob


def get_temperature_prob(bayes_net, temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function

    T_node = bayes_net.get_node_by_name("temperature")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    A_node = bayes_net.get_node_by_name("alarm")

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot], range(Q.nDims))
    temp_prob = Q[index]
    print temp_prob
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    # TODO: fill this out

    A_node = BayesNode(0,4,name='A')
    B_node = BayesNode(1,4,name='B')
    C_node = BayesNode(2,4,name='C')
    AvB_node = BayesNode(3,3,name='AvB')
    BvC_node = BayesNode(4,3,name='BvC')
    CvA_node = BayesNode(5,3,name='CvA')

    A_node.add_child(AvB_node)
    AvB_node.add_parent(A_node)

    A_node.add_child(CvA_node)
    CvA_node.add_parent(A_node)

    B_node.add_child(AvB_node)
    AvB_node.add_parent(B_node)

    B_node.add_child(BvC_node)
    BvC_node.add_parent(B_node)

    C_node.add_child(BvC_node)
    BvC_node.add_parent(C_node)

    C_node.add_child(CvA_node)
    CvA_node.add_parent(C_node)

    skill_distribution = [0.15, 0.45, 0.30, 0.10]

    A_distribution = DiscreteDistribution(A_node)
    index = A_distribution.generate_index([],[])
    A_distribution[index] = skill_distribution
    A_node.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([],[])
    B_distribution[index] = skill_distribution
    B_node.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([],[])
    C_distribution[index] = skill_distribution
    C_node.set_dist(C_distribution)

    dist = zeros([A_node.size(), B_node.size(), AvB_node.size()], dtype=float32)
    dist[0,0,:] = [0.10, 0.10, 0.80]
    dist[0,1,:] = [0.20, 0.60, 0.20]
    dist[0,2,:] = [0.15, 0.75, 0.10]
    dist[0,3,:] = [0.05, 0.90, 0.05]

    dist[1,0,:] = [0.60, 0.20, 0.20]
    dist[1,1,:] = [0.10, 0.10, 0.80]
    dist[1,2,:] = [0.20, 0.60, 0.20]
    dist[1,3,:] = [0.15, 0.75, 0.10]

    dist[2,0,:] = [0.75, 0.15, 0.10]
    dist[2,1,:] = [0.60, 0.20, 0.20]
    dist[2,2,:] = [0.10, 0.10, 0.80]
    dist[2,3,:] = [0.20, 0.60, 0.20]

    dist[3,0,:] = [0.90, 0.05, 0.05]
    dist[3,1,:] = [0.75, 0.15, 0.10]
    dist[3,2,:] = [0.60, 0.20, 0.20]
    dist[3,3,:] = [0.10, 0.10, 0.80]

    AvB_distribution = ConditionalDiscreteDistribution(nodes=[A_node, B_node, AvB_node], table=dist)
    BvC_distribution = ConditionalDiscreteDistribution(nodes=[B_node, C_node, BvC_node], table=dist)
    CvA_distribution = ConditionalDiscreteDistribution(nodes=[C_node, A_node, CvA_node], table=dist)

    AvB_node.set_dist(AvB_distribution)
    BvC_node.set_dist(BvC_distribution)
    CvA_node.set_dist(CvA_distribution)

    nodes = [A_node, B_node, C_node, AvB_node, BvC_node, CvA_node]

    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""

    posterior = [0,0,0]
    # TODO: finish this function    

    AvB_node = bayes_net.get_node_by_name('AvB')
    BvC_node = bayes_net.get_node_by_name('BvC')
    CvA_node = bayes_net.get_node_by_name('CvA')

    engine = EnumerationEngine(bayes_net)
    engine.evidence[AvB_node] = 0 # A won against B
    engine.evidence[CvA_node] = 2 # A tied against C
    Q = engine.marginal(BvC_node)[0]
    posterior = Q.table
    posterior_list = posterior.tolist()

    return posterior_list # List


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)    
    # TODO: finish this function
    raise NotImplementedError
    return sample

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A= bayes_net.get_node_by_name("A")      
    AvB= bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_state)    
    # TODO: finish this function
    raise NotImplementedError    
    return sample


def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    raise NotImplementedError        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    raise NotImplementedError
