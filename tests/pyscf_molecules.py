import numpy as np

# Molecule data taken from https://pennylane.ai/datasets/collection/qchem


class LiH:
    symbols = ["Li", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [[0.39247583, 0.0, 0.0], [-1.1774275, 0.0, 0.0]],
    )
    charge = 0


class H2O:
    symbols = ["O", "H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [
            [0.11727158, 0.0, 0.0],
            [-0.46908633, -0.75696403, 0.0],
            [-0.46908633, 0.75696403, 0.0],
        ]
    )
    charge = 0


class CH4:
    symbols = ["C", "H", "H", "H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [[0.0, 0.0, 0.0], [-0.6276, -0.6276, 0.6276], [0.6276, 0.6276, 0.6276], [-0.6276, 0.6276, -0.6276], [0.6276, -0.6276, -0.6276]]
    )
    charge = 0


class H2:
    symbols = ["H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [
            [-0.37100079, 0.0, 0.0],
            [0.37100079, 0.0, 0.0],
        ]
    )
    charge = 0


class H3plus:
    symbols = ["H", "H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [
            [0.0, 0.504432, 0.0],
            [-0.43685093, -0.252216, 0.0],
            [0.43685093, -0.252216, 0.0],
        ]
    )
    charge = +1


class H4:
    symbols = ["H", "H", "H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [
            [-1.3200028, 0.0, 0.0],
            [-0.44000093, 0.0, 0.0],
            [0.44000093, 0.0, 0.0],
            [1.3200028, 0.0, 0.0],
        ]
    )
    charge = 0


class H6:
    symbols = ["H", "H", "H", "H", "H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [
            [-2.30000488, 0.0, 0.0],
            [-1.38000293, 0.0, 0.0],
            [-0.46000098, 0.0, 0.0],
            [0.46000098, 0.0, 0.0],
            [1.38000293, 0.0, 0.0],
            [2.30000488, 0.0, 0.0],
        ]
    )
    charge = 0


class H8:
    symbols = ["H", "H", "H", "H", "H", "H", "H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [
            [-1.75000371, 0.0, 0.0],
            [-1.25000265, 0.0, 0.0],
            [-0.75000159, 0.0, 0.0],
            [-0.25000053, 0.0, 0.0],
            [0.25000053, 0.0, 0.0],
            [0.75000159, 0.0, 0.0],
            [1.25000265, 0.0, 0.0],
            [1.75000371, 0.0, 0.0],
        ]
    )
    charge = 0


class H10:
    symbols = ["H", "H", "H", "H", "H", "H", "H", "H", "H", "H"]
    basis = "STO-3G"
    geometry = np.array(
        [
            [-4.50000955, 0.0, 0.0],
            [-3.50000743, 0.0, 0.0],
            [-2.50000531, 0.0, 0.0],
            [-1.50000318, 0.0, 0.0],
            [-0.50000106, 0.0, 0.0],
            [0.50000106, 0.0, 0.0],
            [1.50000318, 0.0, 0.0],
            [2.50000531, 0.0, 0.0],
            [3.50000743, 0.0, 0.0],
            [4.50000955, 0.0, 0.0],
        ]
    )
    charge = 0


class H_chain:
    def __init__(self, symbols: list[str], basis: str, geometry: np.ndarray, charge: int):
        self.symbols = symbols
        self.basis = basis
        self.geometry = geometry
        self.charge = charge

    @classmethod
    def build_hydrogen_chain(cls, n: int, bondlength: float = 0.74200158, basis="STO-3G"):
        if n <= 0:
            raise ValueError(f"n needs to be larger than zero but is {n}")
        if n % 2 == 1:
            raise ValueError(f"n needs to be even but is {n}")

        symbols = ["H"] * n
        basis = basis
        geom_x_plus = [x * bondlength / 2 for x in range(1, n // 2 + 1)]
        geom_x_minus = [-x for x in geom_x_plus]
        geom_x = geom_x_minus + geom_x_plus
        geom = [[x, 0.0, 0.0] for x in geom_x]
        geometry = np.array(geom)
        charge = 0
        return cls(symbols, basis, geometry, charge)
