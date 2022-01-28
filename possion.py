import numpy as np
import scipy

def generate_event_histogram(events, shape, samel_window, start_time, end_time):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
    coordinates u and v.
    """
    H, W = shape
    x, y, t, p = events.T
    x = x.astype(np.int)
    y = y.astype(np.int)
    t = (t - t.min())//samel_window
    T = (end_time - start_time)//samel_window

    img_pos = np.zeros((H * W * T,), dtype="float32")
    img_neg = np.zeros((H * W * T,), dtype="float32")

    np.add.at(img_pos, t[p == 1] + T * x[p == 1] + W * T * y[p == 1], 1)
    np.add.at(img_neg, t[p == 1] + T * x[p == 0] + W * T * y[p == 0], 1)

    histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, T, 2))

    return histogram

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = np.nonzero(dense_tensor)
    locations = np.stack(non_zero_indices, axis=-1)

    select_indices = np.array(np.split(locations,range(1,len(locations)))).squeeze(1)
    features = dense_tensor[select_indices[:,0],select_indices[:,1],select_indices[:,2],select_indices[:,3],select_indices[:,4]]

    return locations, features

def possioned_events(events, start_time, end_time, shape, samel_window = 10000, possion_window = 1000):
    time_window = end_time - start_time
    n = samel_window//possion_window
    l = generate_event_histogram(events,shape, samel_window, start_time, end_time)/n
    l = np.repeat(l[:,:,:,None,:],n,3)
    possion_result = np.random.poisson(l)
    locations, ns = denseToSparse(possion_result)
    y, x, t_b, t, p = locations[:,0], locations[:,1], locations[:,2], locations[:,3], locations[:,4]
    ys = []
    xs = []
    ts = []
    ps = []
    for n in np.unique(ns):
        y_n, x_n, t_b, t_n, p_n = y[ns==n], x[ns==n], t_b[ns==n], t[ns==n], p[ns==n]
        y_n = np.repeat(y_n,n)
        x_n = np.repeat(x_n,n)
        t_b_n = np.repeat(t_b_n,n) * samel_window + start_time
        t_n = np.repeat(t_n,n)
        t_n = np.random.uniform(t_n * time_window,(t_n+1) * time_window) + t_b_n
        p_n = np.repeat(p_n,n)
        ys.append(y_n)
        xs.append(x_n)
        ts.append(t_n)
        ps.append(p_n)
    y = np.concatenate(ys)
    x = np.concatenate(xs)
    t = np.concatenate(ts)
    p = np.concatenate(ps)
    events = np.stack([x, y, t, p],axis=-1)
    return events