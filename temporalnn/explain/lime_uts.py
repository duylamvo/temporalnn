def viz_features(x_original, n_features=10, width=32, overlap_rate=0):
    # If plot all segments, it will become multiple time series with same width
    # However, it is hard to see, and also because the index is overlap

    # To plot multiple time series with different index
    # (1) create an empty length with nan values at begining
    # (2) continuously plot the value

    # x ---segment---> x'
    x_segments = segment(x_original, n_features, width, overlap_rate)
    print(x_original.shape, x_segments.shape)

    # visualization
    plt.subplot(211)
    plt.title("Original")
    plt.plot(x_original)

    # plt.subplot(312)
    # plt.title("Segmented")
    # plt.plot(x_segments.transpose())

    plt.subplot(212)
    plt.title("Segmented")
    _len = 0
    _width = int(width * (1 - overlap_rate))
    n_features = x_segments.shape[0]
    for i in range(0, n_features):
        s = np.empty(_len + width)
        s.fill(np.nan)

        _len += _width

        s[-width:] = x_segments[i]
        plt.plot(s)

    plt.show()




def test_main():
    X_STEPS = 32
    STEPS_PER_SEGMENT = 4

    # Loading data for demo
    ts = pd.read_csv("data/ts/climate_demo_stations_44.csv")
    ts.measure_date = pd.to_datetime(ts.measure_date)
    ts = ts.set_index("measure_date").sort_index()

    # Select univariate variable
    temperature = ts["tmk"]
    temperature_till_2016 = temperature[temperature.index.year <= 2016]

    # Select some data for checking predictions
    ts_data = temperature_till_2016[-X_STEPS:]
    model = load_model("models/uts_32_1.h5")

    for i in range(64):
        data = np.asarray(ts_data[-X_STEPS:]) \
            .reshape(1, X_STEPS, 1)
        v = model.predict(data).flatten().item()
        t = pd.Series(v, index=[max(ts_data.index) + timedelta(days=1)])
        ts_data = ts_data.append(t)

    plt.title("Prediction of uts-(32-1) model for temperature")
    plt.plot(temperature_till_2016)
    plt.plot(ts_data)
    plt.show()

    # get an instance x (last 32 steps of end of 2016)
    x_original = np.array(temperature_till_2016[-X_STEPS:])
    x_segments = segment(x_original, N_FEATURES, STEPS_PER_SEGMENT)

    viz_features(x_original, N_FEATURES, STEPS_PER_SEGMENT)

    # Neighbors
    samples_set = neighbors(x_original,
                            n_features=N_FEATURES,
                            n_features_on=int(N_FEATURES / 2),
                            steps_per_segment=STEPS_PER_SEGMENT,
                            sample_size=SAMPLE_SIZE,
                            predict_fn=_predict,
                            model=model,
                            shape=(X_STEPS, 1)
                            )

    # Visualization of Neighboring of z
    choice = np.random.randint(0, len(samples_set))
    sample = samples_set[choice]

    # make a copy to avoid change in sample_set
    z_comma, z_original, f_z = copy.deepcopy(sample)

    plt.subplot(211)
    plt.title("Z' in binary vector ")
    plt.imshow(z_comma.reshape(1, len(z_comma)),
               cmap=plt.cm.gray,
               aspect="auto")

    plt.subplot(212)
    plt.title("Z_original")
    z_original[z_original == 0] = np.nan
    # in case of the feature disabled at the end or begin.
    #   put a value 0 to fix this
    if z_original[-1:] == np.nan:
        z_original = np.insert(z_original, len(z_original), 0)
    if z_original[0] == np.nan:
        z_original = np.insert(z_original, 0, 0)

    plt.plot(z_original)
    plt.show()

    # Define a local explain mode g(z') = w * z'
    linear_xai = Ridge()
    xai_z_commas = [_z_comma for _z_comma, _z_org, _f_z in samples_set]
    xai_target = [_f_z for _z_comma, _z_org, _f_z in samples_set]
    xai_z_original = [_z_org for _z_comma, _z_org, _f_z in samples_set]
    sample_weight = [_pi(x_original, z) for z in xai_z_original]

    linear_xai.fit(xai_z_commas, xai_target, sample_weight)
    xai_weights = linear_xai.coef_
    top_k_idx = get_top_k(xai_weights, TOP_K)
    n_features = min(N_FEATURES, x_segments.shape[0])

    # Visualize xai explains with weights
    plt.subplot(211)
    plt.title("Weights of X' Segments")
    plt.plot(xai_weights, '*', markersize=5)

    high_light = np.zeros(n_features)
    high_light.fill(np.nan)
    high_light[top_k_idx] = 1
    plt.plot(high_light * xai_weights, '*', color="red")

    plt.subplot(212)
    plt.title(f"X Original with top {TOP_K} important segments")
    plt.plot(x_original)
    for i in top_k_idx:
        mask = np.empty(n_features)
        mask.fill(np.nan)
        mask[i] = 1
        xai_segments = x_segments * mask.reshape(n_features, 1)

        alpha = i / max(top_k_idx)
        print(i, alpha)
        plt.plot(xai_segments.ravel(),
                 color="red",
                 alpha=alpha)
    plt.show()
