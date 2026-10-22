import numpy as np




def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


def RemoveNoiseDBSCAN(df, colum = "cluster"):
    noise =(df[colum] == -1)
    df_only_noise = df[noise]
    df_not_noise = df.drop(df_only_noise.index)
    return df_not_noise





def interpcurve(N, pX, pY):
    # Number of points to be uniformly interpolated
    nt = N
    
    # Number of points on the curve
    n = len(pX)
    
    # Concatenate the x and y coordinates of the known points
    pxy = np.column_stack((pX, pY))
    
    # Calculate the chordal arclength of each segment
    chordlen = np.linalg.norm(np.diff(pxy, axis=0), axis=1)
    
    # Normalize the arclengths to a unit total
    chordlen /= np.sum(chordlen)
    
    # Calculate the cumulative arclength
    cumarc = np.insert(np.cumsum(chordlen), 0, 0)
    
    # Calculate the parameter values at which to interpolate
    t = np.linspace(0, 1, nt)
    
    # Initialize the interpolated points array
    pt = np.zeros((nt, 2))
    
    # Interpolate the points
    for i, ti in enumerate(t):
        # Find the segment to interpolate based on the parameter value
        seg_idx = np.searchsorted(cumarc, ti) - 1
        
        # Check if the segment index is out of bounds
        if seg_idx >= n - 1:
            seg_idx = n - 2
        
        # Find the interpolation parameter within the segment
        s = (ti - cumarc[seg_idx]) / chordlen[seg_idx]
        
        # Interpolate the coordinates using linear interpolation
        pt[i] = pxy[seg_idx] + s * (pxy[seg_idx + 1] - pxy[seg_idx])
    
    return pt