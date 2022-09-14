import numpy as np


def getExtremePoints(data, typeOfInflexion=None, maxPoints=None):
    """
    https://towardsdatascience.com/modality-tests-and-kernel-density-estimations-3f349bb9e595
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfInflexion = None returns all inflexion points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange == 1)[0]

    if len(idx) == 0:
        return None

    try:
        if typeOfInflexion == 'max' and data[idx[0]] < data[idx[1]]:
            idx = idx[1:][::2]

        elif typeOfInflexion == 'min' and data[idx[0]] > data[idx[1]]:
            idx = idx[1:][::2]
        elif typeOfInflexion is not None:
            idx = idx[::2]
    except:
        print('error')

    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data) - 1) in idx:
        idx = np.delete(idx, len(data) - 1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx = idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data) // (maxPoints + 1))

    return idx

data = np.random.rand(100)
getExtremePoints(data)
print()