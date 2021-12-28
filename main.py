from icecream import ic
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import adjusted_rand_score

# import local files
from make_dataset import iris_data, binary_encoding, fashion_mnist
from cure import CURE

# initial values
seed = 42

# get data
classes = ['setosa', 'virginica']
X, y = iris_data(classes)
#y = binary_encoding(y)
X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)

# perform cure and make predictions
cure = CURE(random_state=seed)
cure.fit(X_train)
y_pred = cure.predict(X_test)

# evaluate the predictions
adj_rand = adjusted_rand_score(y_test, y_pred) # adj_rand ∈ [-1, 1]
ic(adj_rand)

# dir = './data/raw'
# testset = fashion_mnist(dir)
# ic(type(testset))