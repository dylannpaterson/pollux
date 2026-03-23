import numpy as np
from astroquery.gaia import Gaia

class RomanPhotometryTransformer:
    """
    Performs fast, high-precision conversion from Gaia photometry to Roman AB magnitudes.
    Uses STScI official analytic formulas (Roman-STScI-000825, Scheme B0).
    m_Roman = m_GBP + c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4, where x = m_GRP - m_GBP
    """
    def __init__(self):
        # Coefficients from Table 9 (Scheme B0) for 'none' class (ignoring giant/dwarf distinction for speed)
        # band: [c0, c1, c2, c3, c4]
        self.coeffs = {
            'F062': [ 1.7116e-01,  3.0045e-01,  8.4891e-02, -3.9733e-03, -3.4576e-04],
            'F087': [ 4.6180e-01, -4.5151e-02, -1.5013e-01,  3.7244e-02, -2.7668e-03],
            'F106': [ 7.0631e-01, -2.6156e-01, -2.3948e-01,  6.2219e-02, -5.0079e-03],
            'F129': [ 1.0341e+00, -6.2022e-01, -2.2411e-01,  6.8086e-02, -5.9348e-03],
            'F146': [ 1.1595e+00, -6.7478e-01, -2.5632e-01,  7.8865e-02, -6.8761e-03],
            'F158': [ 1.4984e+00, -1.1197e+00, -1.4930e-01,  6.4087e-02, -6.1885e-03],
            'F184': [ 1.8147e+00, -1.2516e+00, -1.8273e-01,  7.6974e-02, -7.3995e-03],
            'F213': [ 2.0479e+00, -1.1905e+00, -2.4614e-01,  8.9890e-02, -8.3284e-03]
        }

    def convert(self, g_mag, bp_mag, rp_mag, filter_name):
        """ Wholesale conversion from Gaia mags to Roman AB Mag. """
        if filter_name not in self.coeffs:
            print(f"Warning: Filter {filter_name} not supported by STScI formulas. Using Gaia G as proxy.")
            return g_mag
        
        x = bp_mag - rp_mag
        c = self.coeffs[filter_name]
        
        # np.polyval expects highest power first: c4*x^4 + c3*x^3 + ... + c0
        offset = np.polyval(c[::-1], x)
        
        return rp_mag + offset

def get_gaia_reference(ra, dec, filter_name, radius_deg=0.15):
    """
    Query Gaia DR3 and perform fast wholesale conversion to Roman magnitudes.
    """
    transformer = RomanPhotometryTransformer()

    print(f"Querying Gaia DR3 around RA={ra:.5f}, Dec={dec:.5f}...")
    query = f"""
        SELECT ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag 
        FROM gaiadr3.gaia_source 
        WHERE distance({ra}, {dec}, ra, dec) < {radius_deg}
        AND phot_g_mean_mag IS NOT NULL AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL
    """
    job = Gaia.launch_job_async(query)
    result = job.get_results()
    
    if len(result) == 0: return None

    g = np.array(result['phot_g_mean_mag'])
    bp = np.array(result['phot_bp_mean_mag'])
    rp = np.array(result['phot_rp_mean_mag'])
    
    # Apply direct STScI analytic formula
    m_roman = transformer.convert(g, bp, rp, filter_name)
    flux_jy = 10**((m_roman - 8.90) / -2.5)
    
    print(f"Wholesale conversion of {len(result)} Gaia stars to {filter_name} complete.")
    return {
        'ra': np.array(result['ra']),
        'dec': np.array(result['dec']),
        'flux': flux_jy,
        'ab_mag': m_roman
    }
