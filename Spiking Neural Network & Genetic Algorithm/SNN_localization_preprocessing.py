import numpy as np
import pickle

class transform_data:
    def __init__(self, path='/home/dipayan/Desktop/Indoor_Nav/NLOS_data/total_data.pkl'):
        self.path = path

    #function to transform scaler data to spike time points 
    #[a,b] : Min,Max value of temporal encoding
    #[M:m] : Max,Min value that variable f can take

    def scaler_to_temporal_value(self, f,a=10,b=100,M=-32,m=-100):
        return (((b-a)/(M-m))*f) + (((a*M)-(b*m))/(M-m))
        
    def temporal_to_scaler_value(self, y,a=10,b=100,M=-32,m=-100):
        return (y - (((a*M)-(b*m))/(M-m))) / ((b-a)/(M-m))

    def get_temporal_vector_data(self, X, Y):
        temporal_X = []
        temporal_Y = []
        for i in range(X.shape[0]):
        
            x,y = X[i], Y[i]
            
            temp_x = np.zeros((4,749))
            temp_y = np.zeros((2,749)) 
            
            for idx in range(temp_x.shape[0]):
                t = int(x[idx])
                temp_x[idx][t] = 1
            for idx in range(temp_y.shape[0]):
                t = int(y[idx])
                temp_y[idx][t] = 1

            temporal_X.append(temp_x)
            temporal_Y.append(temp_y)

        return np.asarray(temporal_X), np.asarray(temporal_Y)           
            

    def get_spiketime_data(self):
     
        f = open(self.path, 'rb')
        data = pickle.load(f)
        f.close()

        X, Y = data[0], data[1]
        
        min_temporal_val, max_temporal_val = 0, 748
        min_x, max_x = np.min(X), np.max(X)
        min_y, max_y = np.min(Y), np.max(Y)

        trans_X = []
        trans_Y = []

        for i in range(X.shape[0]):
            tx = self.scaler_to_temporal_value(X[i],
                                          a = min_temporal_val,
                                          b = max_temporal_val,
                                          M = max_x,
                                          m = min_x)
            ty = self.scaler_to_temporal_value(Y[i],
                                          a = min_temporal_val,
                                          b = max_temporal_val,
                                          M = max_y,
                                          m = min_y)
            trans_X.append(tx)
            trans_Y.append(ty)  

        trans_X = np.asarray(trans_X)
        trans_Y = np.asarray(trans_Y)
        return trans_X, trans_Y

                                
