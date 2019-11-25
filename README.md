## For the people using Python

To get the code to work you will need to install two packages (but we expect everybody who uses Python already has these):
* NumPy
* SciPy

## For the people using Julia:

To get the code to work you will need to add one package (this one is used for Krylov-based algorithms):
* KrylovKit


## The leg conventions for the various tensors are as follows:

* One-site tensor: 1, 2, 3	(So bond, physical, bond)

* Two-site tensor: 1, 2, 4, 3 (So bond, physical, bond, physical)

* (2-site) Hamiltonian: 3, 4
			1, 2
