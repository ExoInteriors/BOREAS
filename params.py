k_b = 1.380649e-16      # Boltzmann constant, erg K-1
m_H = 1.6735575e-24     # Mass of hydrogen atom, g
am_h = 1                # Αtomic mass hydrogen, u
am_o = 16               # Αtomic mass oxygen, u
m_O = am_o * m_H        # Μass of oxygen atom, grams
mmw_H = 2.35            # Mean molecular weight (H/He-ish envelope)
mmw_eq = 2*am_h + am_o  # Mean molecular weight (water)

# kappa_op = 4e-3       # original value, opacity to optical, cm2 g-1?
kappa_p = 1             # original value 1e-2 for H (?), opacity to outgoing thermal radiation, i.e. mean opacity in infrared
E_photon = 20 * 1.6e-12 # photon energy

G = 6.67430e-8          # Gravitational constant, cm3 g-1 s-2
rearth = 6.371e8        # Radius earth in cgs
mearth = 5.97424e27     # Mass earth in cgs

FEUV = 450.             # received EUV flux, ergs cm-2 s-1
sigma_EUV = 1.89e-18    # EUV cross-section (of H? H2?), cm2
alpha_rec = 2.6e-13     # Recombination coefficient, cm3 s-1
eff = 0.3               # Mass-loss efficiency factor