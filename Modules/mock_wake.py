# Mock wake generator
def generate_mock_wake(
    
    N_total=100_000,
    box_half_size=300.0,                 # kpc; box is [-L, L]^3
    f_overdense=0.99,                  # fraction of particles in wake component
    mu=(-60.0, 0.0, 0.0),                # kpc; wake center (x0, y0, z0)
    sigma_major=80.0,                    # kpc; wake length-scale along major axis
    sigma_minor=40.0,                    # kpc; width in the orthogonal in-plane axis
    sigma_z=40.0,                         # kpc; thickness
    theta_deg=0.0,                       # rotation of wake in XY plane (deg)
    rng_seed=7
):
    """
    Return a structured ndarray with fields: x,y,z,component (1=wake, 0=bg).
    """
    rng = np.random.default_rng(rng_seed)
    N_over = int(N_total * f_overdense)
    N_bg   = N_total - N_over
    mu = np.asarray(mu, dtype=float)

    # Rotation in the XY plane
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])

    # Covariance in XY for the anisotropic Gaussian
    Lambda_xy = np.diag([sigma_major**2, sigma_minor**2])
    Sigma_xy  = R @ Lambda_xy @ R.T

    # Wake samples
    xy_over = rng.multivariate_normal(mean=mu[:2], cov=Sigma_xy, size=N_over)
    z_over  = rng.normal(loc=mu[2], scale=sigma_z, size=N_over)

    # Clip to keep inside box
    xy_over[:, 0] = np.clip(xy_over[:, 0], -box_half_size, box_half_size)
    xy_over[:, 1] = np.clip(xy_over[:, 1], -box_half_size, box_half_size)
    z_over        = np.clip(z_over,        -box_half_size, box_half_size)

    # Uniform background
    x_bg = rng.uniform(-box_half_size, box_half_size, size=N_bg)
    y_bg = rng.uniform(-box_half_size, box_half_size, size=N_bg)
    z_bg = rng.uniform(-box_half_size, box_half_size, size=N_bg)

    # Concatenate positions
    x = np.concatenate([xy_over[:, 0], x_bg])
    y = np.concatenate([xy_over[:, 1], y_bg])
    z = np.concatenate([z_over,        z_bg])

    # Pack catalog
    comp = np.concatenate([np.ones(N_over, dtype=np.int8),
                           np.zeros(N_bg, dtype=np.int8)])  # 1=wake, 0=bg
    

    cat = np.zeros(N_total, dtype=[('x','f4'),('y','f4'),('z','f4'),('component','i1')])
    
    cat['x'], cat['y'], cat['z']   = x, y, z
    cat['component'] = comp
    return cat
