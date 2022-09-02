with open('theta.txt') as f:
    theta = f.read().split(',')
theta[0] = float(theta[0])
theta[1] = float(theta[1])
# theta[2] = float(theta[2])

km = float(input("Entrer le kilométrage de la voiture : "))
# size = float(input("Entrer la superficie : "))
# nb_bedrooms = float(input("Entrer le nombre de chambre : "))

print(f"Le prix d'une voiture avec {km}km est estimé à : {theta[0] + theta[1] * km}")
# print(f"Le prix d'une maison avec {nb_bedrooms} chambres et une superficie de {size}m2 est estimé à : {theta[0] + theta[1] * size + theta[2] * nb_bedrooms}")
