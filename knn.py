import numpy as np

def extract_to_np(df, features, output):
	df['contants'] = 1

	features = ['contants'] + features

	feature_matrix = df[features]

	return (feature_matrix.as_matrix(), df[output].as_matrix())

def normalize_features(X):
	norms = np.linalg.norm(X, axis=0)
	features = X / norms

	return features, norms

def euclid_dist(x, y):
	square = np.vectorize(lambda x: x * x)
	return np.sqrt(np.sum(square(x - y)))

def kNN(target, data, k, alg=euclid_dist):
	nn = []

	for i in range(len(data)):
		dist = alg(target, data[i])

		if len(nn) < k:
			nn.append((data[i], dist, i))
		elif dist < nn[0][1]:
			del nn[0]
			nn.append((data[i], dist, i))

		nn = sorted(nn, key=lambda x: x[1], reverse=True)

	return nn
			
	