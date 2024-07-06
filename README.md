# PSS
Pattern Similarity Search (PSS) is a time series missing value imputation algorithm.

The Pattern Similarity Search (PSS) compares the shape of the source window to the search space, disregarding the actual values. This approach finds the closest match by calculating the distance between the source and the destination data points in the window. The PSS differs from the value similarity search by applying Z-score normalization to both the source and search space (destination windows), after which the PSS is performed on the normalized data. The best match is then de-normalized using the parameters from the source window. For the de-normalization, a small window is used (for the parameters) and the size of the window is selected dynamically.
