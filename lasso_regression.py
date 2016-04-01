import numpy as np

def extract_to_np(df, features, output):
	df['contants'] = 1

	features = ['contants'] + features

	feature_matrix = df[features]

	return (feature_matrix.as_matrix(), df[output].as_matrix())


def predict_outcome(features, weights):
	return features.dot(weights)

def normalize_features(X):
	norms = np.linalg.norm(X, axis=0)
	features = X / norms

	return features, norms

def ro(i, matrix, output, prediction, weights):
	return (matrix[:, i] * (output - prediction + (weights[i] * matrix[:, i]))).sum()

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
	prediction = predict_outcome(feature_matrix, weights)

	ro_i = ro(i, feature_matrix, output, prediction, weights)

	# Don't set intercepts to zero
	if i == 0:
		new_weight_i = ro_i
	elif ro_i < -l1_penalty / 2.:
		new_weight_i = ro_i + l1_penalty / 2.
	elif ro_i > l1_penalty / 2.:
		new_weight_i = ro_i - l1_penalty / 2.
	else:
		new_weight_i = 0.

	return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, inial_weights, l1_penalty, tolerance):
	weights = inial_weights
	changed = True

	while changed:
		changed = False

		for i in range(len(weights)):
			old_weight_i = weights[i]

			weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)

			if old_weight_i - weights[i] > tolerance:
				changed = True

	return weights

def RSS(weights, features, output):
	prediction = predict_outcome(features, weights)
	residual = output - prediction
	square = np.vectorize(lambda x: x * x)
	return square(residual).sum()


######################## Example #########################
my_features = ['Feature 1', 'Feature 2']
my_target = 'Target'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

feature_matrix, output = extract_to_np(df, my_features, my_target)

normalized_feature_matrix, norms = normalize_features(feature_matrix)

weights = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)