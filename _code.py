import numpy as np
import os

def read_lammps_disp(file, maxFrames=1000):
    '''
    Reads a displacements file generated with lammps and return a numpy array.
    1st dimension are frames, 2nd the atoms and 3rd dimension are x, y and z components.
    A maximum number of frames that should be read can be specified.
    '''
    maxAtoms = 100  # dummy value,true value will be inferred later on anyway

    if not os.path.isfile(file):
        print(f'input file not found: {file}')
        raise IOError(f'input file not found: {file}')

    # Open file and check for number of atoms
    with open(file, 'r') as fh:
        bRead = False
        nLine = 0
        for line in fh:
            nLine += 1
            if nLine > 20:
                print('ERROR: The number of Atoms could not be inferred within the first 20 lines')
                return None
            if bRead:
                maxAtoms = np.int(line)
                break
            if 'ITEM: NUMBER OF ATOMS' in line:
                bRead = True

    # open file again and read all data
    data = np.zeros((maxFrames, maxAtoms, 3))
    with open(file, 'r') as fh:
        kFrame = 0
        kAtom = 0
        bSave = False

        for line in fh:
            if kFrame >= maxFrames:
                break
            if bSave:
                if 'ITEM' in line:
                    bSave = False
                    kAtom = 0
                    kFrame += 1
                    continue
                data[kFrame, kAtom, :] = np.array(line.split(), dtype=np.float32)
                kAtom += 1
            if 'ITEM: ATOMS' in line:
                if kFrame >= maxFrames:
                    break
                bSave = True
                print(f' Frame {kFrame + 1} of {maxFrames} Frames', end='\r')

    # If the file had less frames than maxFrames then we delete the unnecessarily allocated data in the end
    if data.shape[0] > kFrame:
        data = np.delete(data, range(kFrame, data.shape[0]), 0)

    return data


def calc_S(r, g, qs):
    ''' Calculate the Structure factor from a radial distribution
    r  = radius = independent variable for g(r)
    g  = rdf = g(r)
    qs = vector of q values to be evaluated =  independent variable for S(q)

    returns
    qs = vector of q values to be evaluated =  independent variable for S(q)
    S  = structure factor S(q)
    '''

    S = np.zeros(qs.shape)

    for k, q in enumerate(qs):
        S[k] = 1 + 4.0 * np.pi * 1e-1 * np.trapz(x=r, y=r * (g - 1.0) * np.sin(q * r)) / q
        #S[k] = 1 + 4.0 * np.pi * 1e-1 * np.trapz(x=r, y=r**2 * (g-1.0) * np.sinc(q * r / np.pi))

    return qs, S


def _sph2cart(azimuth, elevation, r):
    '''
    Simple helper function to transform spherical coordinates to cartesian ones
    This is helpful because random directions in space are best samples in this
    coordinate system rather than in the cartesian one
    '''
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def calc_F(disp, dt=0.1, q=1.84, maxt=1000, nLagSteps=10, numSteps=400, nDirections=20):
    '''
    Calculates the self-intermediate scattering function (F) and the chi_4 (X) from atomic displacements.
    Everything is calculated on a logarithmic selection of points for the time.

    disp        = displacements read with read_displacements or equivalent format
    dt          = time between frames, defines the time unit
    q           = inverse length scale to probe (around 1.8-2.0 A^-1 for water)
    maxt        = longest time to be probed (same unit as dt)
    nLagSteps   = enforce  that only frames with nLagSteps frames between them are used, reduced computation cost
    numSteps    = number of points to be used on the logarithmic x-axis, actual number returned might differ slightly
    nDirections = number of random directions for the q vector or a len=3 list with the direction, e.g. [1,1,1]

    returns
    t = time
    F = self-intermediate scattering function
    X = chi_4
    '''

    nFrames  = disp.shape[0]
    maxStep  = np.int(np.round(maxt / dt))
    Steps    = np.unique(np.round(np.logspace(np.log10(1), np.log10(maxStep), numSteps)))
    numSteps = len(Steps)
    t        = dt * Steps

    # directions for the q vectors
    if isinstance(nDirections, int):
        # create the random directions
        TH = 2 * np.pi * np.random.rand(nDirections)
        PH = np.arcsin(-1 + 2 * np.random.rand(nDirections))
        x, y, z = _sph2cart(TH, PH, 1)
        vecQ = np.array([x, y, z])
    elif isinstance(nDirections, (np.ndarray, list)) and len(nDirections) == 3:
        # use this particular direction (and ensure normalization)
        vecQ = np.reshape(nDirections / np.sqrt(np.sum(np.array(nDirections) ** 2)), (3, -1))
        nDirections = 1
    else:
        print('Error: incorrect usage of the nDirections parameter')
        return None, None, None

    Ndt = np.zeros(numSteps)
    F   = np.zeros(numSteps)
    F2  = np.zeros(numSteps)

    for k, kFrame in enumerate(range(0, nFrames - maxStep, nLagSteps)):
        print(f'{k + 1} of {len(list(range(0, nFrames - maxStep, nLagSteps)))} frames', end='\r')

        for kFrame2 in range(numSteps):
            kdt = int(Steps[kFrame2])
            Ndt[kFrame2] += 1

            x0 = disp[kFrame, :, 0]
            y0 = disp[kFrame, :, 1]
            z0 = disp[kFrame, :, 2]
            xEnd = disp[kFrame + kdt, :, 0]
            yEnd = disp[kFrame + kdt, :, 1]
            zEnd = disp[kFrame + kdt, :, 2]
            r = np.array([xEnd - x0, yEnd - y0, zEnd - z0])

            for kDir in range(nDirections):
                vec   = vecQ[:, kDir]
                bla   = np.exp(1j * q * np.dot(vec, r))
                delta = np.mean(bla)

                if ~np.isnan(delta):
                    F[kFrame2]  += delta
                    F2[kFrame2] += np.abs(delta) ** 2

    # Final normalization of the quantities
    F  = F / Ndt / nDirections
    F2 = F2 / Ndt / nDirections
    X  = disp.shape[1] * (F2 - np.abs(F) ** 2)

    return t, F, X