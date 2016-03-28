import numpy as np

def extract_to_np(df, features, output):
	df['contants'] = 1

	features = ['contants'] + features

	feature_matrix = df[features]

	return (feature_matrix.as_matrix(), df[output].as_matrix())

def predict_outcome(features, weights):
	return features.dot(weights)

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
	if feature_is_constant:
		return 2 * errors.dot(feature)
	else: 
		return 2 * errors.dot(feature) + 2 * weight * l2_penalty

def ridge_regression_gradient_descent(feature_matrix, output, inital_weights, step_size, l2_penalty, max_iterations=1000):
	weights = np.array(inital_weights)

	while max_iterations > 0:
		prediction = predict_outcome(feature_matrix, weights)
		error = prediction - output
		
		for i in xrange(len(weights)):
			is_constant = True if i == 0 else False
			
			derivative = feature_derivative_ridge(error, feature_matrix[:,i], weights[i], l2_penalty, is_constant)

			weights[i] = weights[i] - step_size * derivative

		max_iterations = max_iterations - 1

	return weights

def RSS(weights, features, output):
	prediction = predict_outcome(features, weights)
	residual = output - prediction
	square = np.vectorize(lambda x: x * x)
	return square(residual).sum()


# EXAMPLE ******************************
features = ['Feature 1', 'Feature 2']
my_output = 'Target'

feature_matrix, output = extract_to_np(train_data, features, my_output)
test_feature_matrix, test_output = extract_to_np(test_data, features, my_output)

initial_weights = np.array([0., 0., 0.])
step_size = 1e-12
max_iterations = 1000
l2_penalty = 1e5

weights = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)


