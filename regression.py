import numpy as np

def extract_to_np(df, features, output):
	df['contants'] = 1

	features = ['contants'] + features

	feature_matrix = df[features]

	return (feature_matrix.as_matrix(), df[output].as_matrix())

def predict_outcome(features, weights):
	return weights.dot(features)

def feature_derivative(errors, features):
	return 2 * errors.dot(features)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
	converged = False
	weights = np.array(initial_weights)
	
	while not converged:
		prediction = feature_matrix.dot(weights)
		error = prediction - output

		gradient_sum_squares = 0

		for i in range(len(weights)):
			derivative = feature_derivative(error, feature_matrix[:, i])

			gradient_sum_squares += derivative * derivative

			weights[i] = weights[i] - step_size * derivative


		gradient_magnitude = np.sqrt(gradient_sum_squares)
		if gradient_magnitude < tolerance:
			converged = True

	return weights

######################## Example ##########################
my_features = ['Feature 1', 'Feature 2']
my_target = 'Target'

train_features, output = extract_to_np(train_frame, my_features, my_target)

initial_weights = np.array([0., 0., 0.])
step_size = 4e-12
tolerance = 1e9

weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)