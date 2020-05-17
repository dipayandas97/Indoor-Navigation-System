import numpy as np
import matplotlib.pyplot as plt
from SNN_localization_preprocessing import transform_data
from neuron_models import LIF
from sklearn.model_selection import train_test_split
from GA import GA
import pickle

#Spike distance----------------------------------------------------------------------------------------------------------------------------------------
def error(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise Exception('Shape of y_true and y_pred is not equal')
    
    true_timestamps, pred_timestamps = np.zeros((y_true.shape[0],))-1, np.zeros((y_pred.shape[0],))-1
    
    for d in range(y_true.shape[0]):
    
        for t in range(y_true.shape[1]):
            if y_true[d][t] == 1:           # |delta(n)| = 1
                true_timestamps[d] = t
                break
        for t in range(y_pred.shape[1]):
            if y_pred[d][t] == 5:           # |V_spike| = 5
                pred_timestamps[d] = t
                break                        
        #if no spikes have been predicted
        if pred_timestamps[d] == -1:
            pred_timestamps[d] = y_true.shape[1] - true_timestamps[d]

    pred_timestamps[pred_timestamps==-1] = 0
    true_timestamps[true_timestamps==-1] = 0
    #print(true_timestamps, pred_timestamps)                    
    e = [(true_timestamps[i]-pred_timestamps[i])**2 for i in range(y_true.shape[0])]
    e = np.sqrt(np.sum(e))
    return e
    
#Simulation for calculating fitness of each synapse_matrix solution------------------------------------------------------------------------------------

simulation_time = 748 #Max temporal lenght of (input,output) signal

def fitness(population): #array of vectors
    train_score, test_score = [], []
    synapses = ga.vector_to_matrix(population, synapse_dim)
    
    for idx, s in enumerate(synapses):
        print('Solution:',idx+1,'/',synapses.shape[0])
        synapse = s[0] #since single hidden layer

        #Error on train_data
        temp_train_score = []      
        for i in range(X_train.shape[0]):     
            signal = X_train[i] #Input
            activations=[]      #Output
      
            for l in range(n):
                activations.append([0])
                h_layer[l].initialize()

            for t in range(simulation_time):            
                for idx,neuron in enumerate(h_layer):
                    input_I = np.dot(synapse[idx], signal[:,t])
                    activations[idx].append(neuron.update(input_I, t))
            activations = np.asarray(activations)
            
            temp_train_score.append( error(y_train[i],activations) )
            
        #calculate mean train error for this particular synapse matrix
        train_score.append(np.mean(temp_train_score))
        
        #Error on test data
        temp_test_score = []        
        for i in range(X_test.shape[0]):     
            signal = X_test[i] #Input
            activations=[]     #Output
      
            for l in range(n):
                activations.append([0])
                h_layer[l].initialize()

            for t in range(simulation_time):            
                for idx,neuron in enumerate(h_layer):
                    input_I = np.dot(synapse[idx], signal[:,t])
                    activations[idx].append(neuron.update(input_I, t))
            activations = np.asarray(activations)
            
            temp_test_score.append( error(y_test[i],activations) )
            
        #calculate mean test error for this particular synapse matrix
        test_score.append(np.mean(temp_test_score))   

    return train_score, test_score     
    
#Data--------------------------------------------------------------------------------------------------------------------------------------------------
#signal #Input signal : shape(batchsize, features, timesteps)
transform = transform_data()
X, Y = transform.get_spiketime_data()
signal_X, signal_Y = transform.get_temporal_vector_data(X,Y)
X_train, X_test, y_train, y_test = train_test_split(signal_X, signal_Y, test_size=0.2, shuffle=True)
print('Train sizes: ',X_train.shape, y_train.shape)
print('Test sizes: ',X_test.shape, y_test.shape)

#Network-----------------------------------------------------------------------------------------------------------------------------------------------
dt = 0.125 #ms
m, n = 4, 2 
h_layer = []
for i in range(n):
    neuron = LIF(threshold=0.01, dt=dt)
    h_layer.append(neuron)
    
synapse = np.random.uniform(0,2,size=(n,m)) #Parameters to be optimized
synapse_dim = (1,1,n,m)

#Genetic algorithm parameters:------------------------------------------------------------------------------------------------------------------------
#    Mating Pool Size (Number of Parents)
#    Population Size
#    Number of Generations
#    Mutation Percent

sol_per_pop = 100
num_parents_mating = 25
num_generations = 500
mutation_percent = 30
ga = GA()

#initial population :         
initial_population_weights = np.random.uniform(-1.,2.,size=(sol_per_pop, 1, n, m))
3
population_matrices = initial_population_weights
population_vectors = ga.matrix_to_vector(initial_population_weights)

train_error, test_error = [], []

for generation in range(num_generations):
    print('============================================================================')
    print("Generation : ", generation+1,'/',num_generations)
    print('============================================================================')    
    
    #Calculate fitness for each population_vector in population    
    fitness_train, fitness_test = fitness(population_vectors)
    train_error.append(np.min(fitness_train))
    test_error.append(np.min(fitness_test))
    print('Train Error:', train_error[-1])
    print('Test Error:', test_error[-1])
    
    #Select best parents 
    parents = ga.select_mating_pool(population=population_vectors, fitness=fitness_train.copy(), mode='min', num_parents=num_parents_mating)
    print(parents.shape[0],'New parents generated...')
    
    #Crossover parents
    offsprings = ga.crossover(parents=parents, num_offsprings=sol_per_pop-num_parents_mating)
    print(offsprings.shape[0],'New offsprings produced...')
    
    #Mutate offsprings
    mutated_offsprings = ga.mutate(offsprings, mutation_percent=mutation_percent)
    print(mutated_offsprings.shape[0],'offsprings mutated')
    
    #Create new population
    population_vectors[:parents.shape[0],:] = parents
    population_vectors[parents.shape[0]:,:] = mutated_offsprings
    print('New population created')
    
    #Update the result of this generation in report files
    f = open('train_error.pkl','wb')
    pickle.dump(train_error, f)
    f.close()
    f = open('test_error.pkl','wb')
    pickle.dump(test_error, f)
    f.close()
    f = open('best_paramater_found.pkl','wb')
    pickle.dump(parents[0], f)
    f.close()

plt.plot(train_error)
plt.plot(test_error)
plt.show()                                            

        
        
        
        
        
        
        
        
        
        
        
        
        

