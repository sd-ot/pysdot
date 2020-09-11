import pysdot

positions = np.random.rand(200, 2)
# ou une fonction qui prend une fonction pour donner les poids, les masses
#  positions = lambda f:
#    f( np.random.rand(200, 2) )
#    f( np.random.rand(300, 2) )
#    ...

# obligation de donner la densité, forcer de donner des poids

lag = pysdot.solve_optimal_transport(positions, masses, pysdot.densities.box( ... ), lag, positions_have_changed = False) # ou adjust_kantorovith_potentials

for i in ...
    lag.positions[0,i] += 1 # ou immutable, avec set_positions qui met à jour date

lag = pysdot.solve_optimal_transport(lag, pysdot.densities.box( ... )) # ou adjust_kantorovith_potentials


lag = pysdot.optimal_transport(positions, pysdot.densities.box( ... ) ) # ou adjust_kantorovith_potentials
lag.potentials

lag.masses(density)
lag.barycenters(density)
lag.variances(density) # \int (x-xi)^2

# LaguerreWithDensity

lag.gradient_of_masses_wrt_potentials()
lag.gradient_of_masses_wrt_positions()

with_density( local_density = "...",  )

# Voir si ce qu'on fait est compatible avec PyTorch, TF, ...

# ou
# ot = pysdot.transport_plan()
# pysdot.adjust_dirac_weights(ot, positions)
# 
# ou
# ot = pysdot.transport_plan()
# ot.positions = ...
# ot.adjust_dirac_weights()

# pour éviter les recalculs
# ot.adjust_dirac_weights(positions_have_changed = False)
# pysdot.adjust_dirac_weights(ot, positions_have_changed = False)

ot.display()
print(ot.dirac_weights)


# pour le out-of-core et MPI
for chunk in ot.all_the_local_chunks:
    ot.set_active_chunks( [ chunk ] )
    print(ot.dirac_weights)
    # print( chunk.nb_diracs )
    # ...

    # pour les données voisines
    print(ot.get_dirac_weights( other_chunk ))


# ou
for _ in ot.for_each_chunk:
    ot.set

# données additionnelles. Si les chunks sont assez gros, on peut tolérer des données dynamiques
# On pourrait les nommer ou les indexer. Prop: en C++ en interne, c'est indexé, mais il y a une map pour trouver fonction du nom


# Rq: statégie de partitionnement. Proposition actuelle: on fait un histogramme / zindex puis un scan. Pb d'heuristique potentiel.
# Il 
# Si tout loge dans un chunk, on ne va pas chercher plus loin.

# Pour le 