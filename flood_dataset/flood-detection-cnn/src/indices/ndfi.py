def compute_ndfi(vh_image, vv_image, scaling_value=1.0):
    """
    Compute the Normalized Difference Flood Index (NDFI) for flood detection.
    
    Parameters:
    - vh_image: The VH band image (numpy array).
    - vv_image: The VV band image (numpy array).
    - scaling_value: A value to scale the images (default is 1.0).
    
    Returns:
    - ndfi: The computed NDFI (numpy array).
    """
    # Scale the images
    vh_image = vh_image / scaling_value
    vv_image = vv_image / scaling_value
    
    # Calculate NDFI
    ndfi = (vh_image - vv_image) / (vh_image + vv_image + 1e-10)  # Adding a small constant to avoid division by zero
    
    return ndfi

def load_and_compute_ndfi(vh_path, vv_path, scaling_value=1.0):
    """
    Load VH and VV images and compute the NDFI.
    
    Parameters:
    - vh_path: File path for the VH band image.
    - vv_path: File path for the VV band image.
    - scaling_value: A value to scale the images (default is 1.0).
    
    Returns:
    - ndfi: The computed NDFI (numpy array).
    """
    vh_image = load_and_preprocess_sar(vh_path, scaling_value=scaling_value)
    vv_image = load_and_preprocess_sar(vv_path, scaling_value=scaling_value)
    
    ndfi = compute_ndfi(vh_image, vv_image, scaling_value)
    
    return ndfi