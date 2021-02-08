"""
  Values to calculate effective parameters for energy transport
"""

# Dictionaries of parameters
# Empirically determined parameters indexed by bath gas
# prob need to be uncombined and units changed
LJ_DCT = {
    'InChI=1S/N2/c1-2': {
        'InChI=1S/H': (162.69397, 1.72028),
        'InChI=1S/H2/h1H': (91.51536, 2.38028),
        'InChI=1S/N2/c1-2': (325.74586, 3.41972)
    }
}

LJ_EST_DCT = {

    # N2 Bath Gas
    'InChI=1S/N2/c1-2': {
        'n-alkane': (3.68, 0.16, 100.0, 0.25),
        '1-alkyl': (3.72, 0.15, 69.00, 0.39),
    },

    # He Bath Gas
    'InChI=1S/He': {
        'n-alkane': (3.33, 0.17, 21.3, 0.31),
        'n-alcohol': (2.90, 0.21, 22.0, 0.28),
        'n-hydroperoxide': (2.90, 0.21, 10.0, 0.75),
        '1-alkyl': (3.20, 0.18, 20.00, 0.42),
    },

    # Ar Bath Gas
    'InChI=1S/Ar': {
        'n-alkane': (3.40, 0.18, 113.0, 0.31),
        'n-alcohol': (3.05, 0.20, 150.0, 0.29),
        'n-peroxide': (2.90, 0.25, 85.0, 0.32),
        'epoxide': (2.90, 0.23, 75.0, 0.50),
        'n-hydroperoxide': (3.05, 0.20, 110.0, 0.39),
        '1-alkyl': (3.50, 0.17, 90.0, 0.38),
        'ether': (3.15, 0.22, 110.0, 0.15) 
    },

    # H2 Bath Gas
    'InChI=1S/H2/h1H': {
        'n-alkane': (3.15, 0.18, 75.0, 0.30)
    },
}

Z_ALPHA_EST_DCT = {

    # N2 Bath Gas
    'InChI=1S/N2/c1-2': {
        'n-alkane': {
            300: (0.019689, -2.109835, 49.893116, 7.130308),
            1000: (0.323108, -11.8867, 140.285802, 56.212929),
            2000: (0.527413, -20.87865, 236.980272, 213.765542)
        },
        '1-alkyl': {
            300: (0.1123, -3.9325, 44.4515, 7.3137),
            1000: (0.2740, -9.7140, 99.5144, 93.4755),
            2000: (0.5837, -19.3363, 181.9322, 250.0161)
        }
    },

    # He Bath Gas
    'InChI=1S/He': {
        'n-alkane': {
            300: (0.071882, -3.16556, 51.303812, 9.570797),
            1000: (0.305537, -10.232371, 113.6291, 180.205289),
            2000: (0.414858, -15.030357, 394911, 158, 534.047802)
        },
        'n-alcohol': {
            300: (0.07044, -3.28388, 50.87526, -1.05945),
            1000: (0.3905, -12.61792, 129.09928, 122.81934),
            2000: (0.12611, -6.70779, 83.35158, 649.16698)
        },
        'n-hydroperoxide': {
            300: (0.06934, -2.16846, 28.09786, 57.99002),
            1000: (-0.01873, 0.9463, -9.19049, 421.53109),
            2000: (-0.33498, 7.81371, -60.10907, 848.6624)
        },
        '1-alkyl': {
            300: (0.1113, -3.6213, 36.8624, 36.1446),
            1000: (0.1766, -5.2337, 43.6089, 261.9713),
            2000: (0.2416, -6.7709, 47.0932, 590.3983)
        }
    },

    # Ar Bath Gas
    'InChI=1S/Ar': {
        'n-alkane': {
            300: (0.054986, -2.948084, 52.534594, -2.984767),
            1000: (0.273835, -10.208643, 125.075803, 31.934663),
            2000: (0.457981, -19.131921, 228.021958, 110.533659)
        },
        'n-alcohol': {
            300: (0.1634, -6.39251, 83.21585, -64.1994),
            1000: (0.22482, -9.06605, 114.55153, 63.80238),
            2000: (0.45001, -18.13079, 206.53042, 163.72987)
        },
        'n-peroxide': {
            300: (0.084, -2.9394, 31.0133, 38.0167),
            1000: (-0.011, -0.6012, 11.7443, 280.8983),
            2000: (-0.0117, -2.0414, 34.5284, 518.2758)
        },
        'n-hydroperoxide': {
            300: (0.04706, -2.28117, 35.3249, 31.83379),
            1000: (0.09842, -3.83485, 49.11378, 192.03656),
            2000: (0.19506, -6.74549, 69.65477, 469.06881)
        },
        '1-alkyl': {
            300: (0.0873, -3.3639, 39.6649, 8.5801),
            1000: (0.3216, -10.2583, 98.7330, 62.6353),
            2000: (0.4581, -16.4145, 167.6469, 178.9579)
        },
        'ether': {
            300: (0.1057, -4.1378, 52.8870, 1.5181),
            1000: (0.0116, -2.2454, 40.9493, 251.0031),
            2000: (-0.1170, -0.4961, 40.0212, 580.5769)
        },
        'epoxide': {
            300: (0.5025, -18.4928, 200.3532, -246.9419),
            1000: (0.7488, -25.8914, 247.3055, 25.3337),
            2000: (0.896, -30.0515, 262.9319, 412.6931)
        },
    },

    # H2 Bath Gas
    'InChI=1S/H2/h1H': {
        'n-alkane': {
            300: (0.185, -7.2964, 102.6128, 46.6884),
            1000: (0.6173, -19.5856, 194.2888, 295.3819),
            3000: (0.3867, 14.6607, 153.5746, 922.4951)
        }
    }
}

# Bond dissociation energies (kcal/mol)
D0_DCT = {
    'm-peroxide': 35.0,
    'ro2_qooh': 35.0,
    'qooh': 35.0,
    'ketohydroperoxide': (35.0, 51.0),
    '1-alkyl': 40.0,
    'n-hydroperoxide': 45.0,
    'epoxide': 60.0,
    'm-ether': 82.0,
    'n-alcohol': 90.0,
    'n-alkane': 95.0,
    'n-alkene': 95.0,
}
