"""Utility functions for filtering and managing series from DICOM data."""

import re


def filter_series(series_list, keywords=None, include=None, exclude=None, order=True):
    """
    Filter a list of series based on keywords and inclusion/exclusion patterns.
    
    Parameters
    ----------
    series_list : list
        List of series names to filter
    keywords : list, optional
        List of keywords that ALL must be present in the series name.
        Example: ['IVIM', 'DWI'] will match series containing both words
    include : list, optional
        List of patterns where AT LEAST ONE must be present in the series name.
        Example: ['_RR_'] will only match series containing '_RR_'
    exclude : list, optional
        List of patterns where NONE should be present in the series name.
        Example: ['_RR_'] will exclude series containing '_RR_'
    order : bool, optional
        If True, sort the filtered series by the numeric part before _MR.
        Default is True.
    
    Returns
    -------
    filtered_series : list
        List of series matching all criteria, optionally sorted by series number
    
    Examples
    --------
    >>> all_series = ['IVIM_DWI_RR_101', 'IVIM_DWI_102']

    # Get IVIM_DWI with RR
    >>> filter_series(all_series, keywords=['IVIM', 'DWI'], include=['_RR_'])
    ['IVIM_DWI_RR_101']

    # Get IVIM_DWI without RR
    >>> filter_series(all_series, keywords=['IVIM', 'DWI'], exclude=['_RR_'])
    ['IVIM_DWI_102']

    # Get all IVIM series
    >>> filter_series(all_series, keywords=['IVIM'])
    ['IVIM_DWI_RR_101', 'IVIM_DWI_102']
    """
    
    filtered = []
    
    for series in series_list:
        # Check keywords (ALL must be present)
        if keywords:
            if not all(keyword in series for keyword in keywords):
                continue
        
        # Check include patterns (AT LEAST ONE must be present)
        if include:
            if not any(pattern in series for pattern in include):
                continue
        
        # Check exclude patterns (NONE should be present)
        if exclude:
            if any(pattern in series for pattern in exclude):
                continue
        
        filtered.append(series)

    if order:
        # Extract and sort by the numeric part before _MR
        def get_series_number(series_name):
            """Extract the numeric part before _MR (e.g., 29001 from ...29001_MR)"""
            match = re.search(r'_(\d+)_MR', series_name)
            if match:
                return int(match.group(1))
            return float('inf')  # Put series without numbers at the end
        
        filtered.sort(key=get_series_number)
    
    return filtered

def get_single_serie(series_list, keywords=None, include=None, exclude=None):
    """
    Get a single serie from the filtered list. Raises an error if not exactly one match.
    
    Parameters
    ----------
    series_list : list
        List of series names to filter
    keywords : list, optional
        List of keywords that ALL must be present in the series name.
    include : list, optional
        List of patterns where AT LEAST ONE must be present in the series name.
    exclude : list, optional
        List of patterns where NONE should be present in the series name.
    
    Returns
    -------
    serie : str
        The single matching series name.
    
    Raises
    ------
    ValueError
        If not exactly one series matches the criteria.
    """
    
    filtered_series = filter_series(
        series_list,
        keywords=keywords,
        include=include,
        exclude=exclude,
        order=False
    )
    
    if len(filtered_series) != 1:
        raise ValueError(f"Expected exactly one series, found {len(filtered_series)}: {filtered_series}")
    
    return filtered_series[0]
