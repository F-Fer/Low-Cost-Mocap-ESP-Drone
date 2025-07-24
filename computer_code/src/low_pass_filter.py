import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

class LowPassFilter:
    """
    Butterworth low-pass filter for smoothing noisy signals.
    Uses proper online filtering with state preservation.
    """

    def __init__(self, cutoff_frequency: float, sampling_frequency: float, dims: int, order: int = 5, buffer_size: int = 300):
        """
        Initialize the low-pass filter.
        
        Args:
            cutoff_frequency: Cutoff frequency in Hz
            sampling_frequency: Sampling frequency in Hz  
            dims: Number of dimensions to filter
            order: Filter order (higher = steeper rolloff)
            buffer_size: Size of internal data buffer (for debugging/monitoring)
        """
        self.sampling_frequency = sampling_frequency
        self.cutoff_frequency = cutoff_frequency
        self.order = order
        self.dims = dims
        self.buffer_size = buffer_size
        
        # Design Butterworth filter
        self.b, self.a = butter(self.order, self.cutoff_frequency / (self.sampling_frequency / 2), btype='low')
        
        # Initialize filter state for online filtering (one per dimension)
        self.zi = []
        for _ in range(dims):
            zi_single = lfilter_zi(self.b, self.a)
            self.zi.append(zi_single)
        
        # Optional: Keep recent data for monitoring
        self.recent_data = np.empty((0, dims))
        self.is_initialized = False

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply low-pass filter to input data using online filtering.
        
        Args:
            data: Input data to filter, shape should match self.dims
            
        Returns:
            Filtered data with same shape as input
        """
        # Convert input to numpy array if it isn't already
        data = np.asarray(data)
        
        # Handle different input formats
        if data.ndim == 0:  # Scalar input
            if self.dims != 1:
                raise ValueError(f"Scalar input provided but filter expects {self.dims} dimensions")
            input_values = np.array([data])
        elif data.ndim == 1:  # 1D array input
            if len(data) != self.dims:
                raise ValueError(f"Input has {len(data)} dimensions but filter expects {self.dims}")
            input_values = data
        else:
            raise ValueError(f"Input data has unsupported number of dimensions: {data.ndim}")
        
        # Initialize filter states with first input if not done yet
        if not self.is_initialized:
            for i in range(self.dims):
                # Initialize filter delay line with the first input value
                self.zi[i] = self.zi[i] * input_values[i]
            self.is_initialized = True
        
        # Apply filter to each dimension separately using online filtering
        filtered_values = np.zeros(self.dims)
        for i in range(self.dims):
            # Apply filter to single sample with state preservation
            filtered_val, self.zi[i] = lfilter(self.b, self.a, [input_values[i]], zi=self.zi[i])
            filtered_values[i] = filtered_val[0]
        
        # Optional: Keep recent data for monitoring
        if self.recent_data.shape[0] >= self.buffer_size:
            self.recent_data = self.recent_data[-self.buffer_size//2:]
        self.recent_data = np.vstack([self.recent_data, input_values.reshape(1, -1)])
        
        # Return result in appropriate format
        if self.dims == 1:
            return filtered_values[0]  # Return scalar for 1D filters
        else:
            return filtered_values  # Return 1D array for multi-dimensional filters

    def reset(self):
        """Reset the filter state."""
        # Reset filter states
        self.zi = []
        for _ in range(self.dims):
            zi_single = lfilter_zi(self.b, self.a)
            self.zi.append(zi_single)
        
        # Reset monitoring data
        self.recent_data = np.empty((0, self.dims))
        self.is_initialized = False 