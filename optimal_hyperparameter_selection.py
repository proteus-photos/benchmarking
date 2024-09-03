import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression

with open("/home/singh/benchmarking/hyperparameter_tuning_results.json") as f:
    samples = json.load(f)

fields = samples[0]["params"].keys()

######## WHICH FEATURES CAUSE INSTABILITY ########
X = []
y = []
for sample in samples:
    params = sample["params"]
    X.append([params[field] for field in fields])
    y.append(sample["val_loss"])
    
sorted_samples = sorted(list(filter(lambda x: x["val_loss"]>0, samples)), key=lambda x: x["val_loss"])
lowest_samples = sorted_samples[:10]

print("Lowest 10 samples:")
for sample in lowest_samples:
    print(f"val_loss: {sample['val_loss']}, params: {sample['params']}")

X = np.array(X)
y_binary = np.array([1 if loss > 0 else 0 for loss in y])
X = (X - X.mean(axis=0)) / X.std(axis=0)
print(f"Fraction of positive samples: {y_binary.sum() / len(y_binary)}")

model = LogisticRegression(class_weight="balanced", max_iter=10000)
model.fit(X, y_binary, )
y_pred = model.predict(X)

print("For stability:")
for i, (feature, coef) in enumerate(zip(fields, model.coef_[0])):
    print(f"{feature}: {coef}")


######## WHICH FEATURES ARE CORRELATED ########
model = LinearRegression()
X = []
y = []
for sample in samples:
    params = sample["params"]
    if sample["val_loss"] < 0:
        continue
    X.append([params[field] for field in fields])
    y.append(sample["val_loss"])

X = np.array(X)
y = -np.log(np.array(y))
X = (X - X.mean(axis=0)) / X.std(axis=0)

model.fit(X, y)
y_pred = model.predict(X)

r_squared = r2_score(y, y_pred)
print("For performance:")
for i, (feature, coef) in enumerate(zip(fields, model.coef_)):
    model.fit(np.delete(X, i, axis=1), y)
    y_pred = model.predict(np.delete(X, i, axis=1))

    r_squared_without_feature = r2_score(y, y_pred)
    print(f"{feature}: {coef}")
    print(f"R-squared difference {feature}: {r_squared - r_squared_without_feature}")
    print("#######")