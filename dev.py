import numpy as np


def fromAtoB(PA, Ra2b, Ta2b_basea):
    '''
    Fait les calculs suivants : Pb = Ra2b.T x Pa - Ra2b.T x Ta2b_basea
    :param PA: Coordonné d'un point dans le repère A
    :param Ra2b: Rotation de A vers B
    :param Ta2b_basea: Vecteur de translation OaOb exprimé dans la base A
    :return: PB : Coordonnées du point P dans le repère B
    '''

    T = np.transpose(Ra2b).dot(np.transpose(Ta2b_basea))
    print(np.transpose(T))
    PB = np.transpose(Ra2b).dot(np.transpose(PA)) - np.transpose(T)
    return PB

def rotation_matrix(theta):
    theta = np.pi/180 * theta
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R

R = rotation_matrix(90)
PA = np.array([1, 0, 0])

# T est exprimé dans la base de départ
print(fromAtoB(PA, R, np.array([6, 1, 0])))

