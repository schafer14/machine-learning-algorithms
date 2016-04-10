import numpy as np

def get_numpy_data(sframe, features, label):
    sframe['intercept'] = 1
    features = ['intercept'] + features
    feature_sframe = sframe[features]
    feature_matrix = feature_sframe.to_numpy()
    label_affay = sframe[label].to_numpy()
    
    return feature_matrix, label_affay

def sigmoid(x):
    return 1 / ( 1 + ( math.exp(-x ) ) )

def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    score = np.dot(feature_matrix, coefficients)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    sig = np.vectorize(sigmoid)
    predictions = sig(score)
    
    # return predictions
    return predictions

def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors, feature)
    
    # Return the derivative
    return derivative

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment == +1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative(errors, feature_matrix[:,j])
            
            # add the step size times the derivative to the current coefficient
            coefficients[j] = coefficients[j] + step_size * derivative

    return coefficients