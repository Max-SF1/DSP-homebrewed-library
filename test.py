import numpy as np

def Threshold(x: np.ndarray, th: float) -> np.ndarray:
    # Ensure input is a NumPy array
    x = np.array(x, ndmin=1)  # Convert scalars to 1D arrays if needed
    y = np.zeros_like(x, dtype=x.dtype)  # Ensure output matches input dtype
    
    if x.ndim == 1:  # For 1D arrays
        for i in range(len(x)):
            if np.abs(x[i]) >= th:  # Use magnitude for complex numbers
                y[i] = x[i]
    elif x.ndim == 2:  # For 2D matrices
        for i in range(x.shape[0]):  
            for j in range(x.shape[1]):
                if np.abs(x[i, j]) >= th:  # Use magnitude for complex numbers
                    y[i, j] = x[i, j]
    else:
        raise ValueError("Input array must be 1D or 2D.")
    
    return y
print(Threshold([5,3,3,1,8,0,-2],4))