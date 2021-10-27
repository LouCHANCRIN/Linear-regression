with open('theta.txt') as f:
    theta = f.read().split(',')
theta[0] = float(theta[0])
theta[1] = float(theta[1])
print(theta)

km = float(input("Entrer le kilométrage de la voiture : "))

print(f"Le prix d'une voiture avec {km}km devrait être : {theta[0] + theta[1] * km}")