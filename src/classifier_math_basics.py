import math

# Naive Bayes — exemplo didático

prior_1 = 0.40
prior_0 = 0.60

p_fire_1 = 0.80
p_fire_0 = 0.20

p_school_1 = 0.50
p_school_0 = 0.50

# TODO: Calcule os escores para cada classe
score_1 = prior_1 * p_fire_1 * p_school_1
score_0 = prior_0 * p_fire_0 * p_school_0

# TODO: Determine a classe com o maior escore
pred_nb = 1 if score_1 > score_0 else 0


# Logistic Regression — exemplo didático

x = [2, 1]
w = [0.5, -0.4]
b = -0.2

# TODO: Calcule a combinação linear
z = (w[0] * x[0]) + (w[1] * x[1]) + b

# TODO: Implemente a função sigmoide
prob = 1 / (1 + math.exp(-z))

# TODO: Converta a probabilidade em uma classe
pred_lr = 1 if prob >= 0.5 else 0


assert math.isclose(score_1, 0.16, rel_tol=1e-9)
assert math.isclose(score_0, 0.06, rel_tol=1e-9)
assert math.isclose(z, 0.4, rel_tol=1e-9)
assert math.isclose(prob, 0.59868766, abs_tol=1e-7)
assert pred_nb == 1
assert pred_lr == 1

print("Naive Bayes:")
print(f"Score classe 1: {score_1:.4f}")
print(f"Score classe 0: {score_0:.4f}")
print(f"Previsão: {pred_nb}")

print("\nLogistic Regression:")
print(f"Z: {z:.4f}")
print(f"prob: {prob:.4f}")
print(f"Previsao: {pred_lr}")