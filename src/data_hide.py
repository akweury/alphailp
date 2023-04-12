def vertex_normalization(data):
    if len(data.shape) != 3:
        raise ValueError

    ax = 0
    data[:, :, :3] = (data[:, :, :3] - data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[0]) / (
            data[:, :, ax:ax + 1].max(axis=1, keepdims=True)[0] - data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[
        0] + 1e-10)

    ax = 2
    data[:, :, ax] = data[:, :, ax] - data[:, :, ax].min(axis=1, keepdims=True)[0]
    # for i in range(len(data)):
    #     data_plot = np.zeros(shape=(5, 2))
    #     data_plot[:, 0] = data[i, :5, 0]
    #     data_plot[:, 1] = data[i, :5, 2]
    #     chart_utils.plot_scatter_chart(data_plot, config.buffer_path / "hide", show=True, title=f"{i}")
    return data


