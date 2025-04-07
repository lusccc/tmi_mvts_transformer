from nemenyi_plot_tool import nemenyi_plot_multiple


results = {
    "RF": [56.3, 59.7, 64.0, 60.0],
    "KNN": [38.7, 47.3, 46.8, 45.9],
    "MLP": [40.3, 47.7, 54.0, 44.6],
    "DT": [44.5, 49.3, 50.8, 52.4],
    "SVM": [41.0, 48.0, 52.2, 43.7],
    "CNN-based": [81.6, 81.1, 85.6, 69.9],
    # "CNN-based [8]": [81.6, 81.1, 85.6, 69.9],
    "LSTM-based": [80.6, 80.3, 85.6, 69.6],
    # "LSTM-based [12]": [80.6, 80.3, 85.6, 69.6],
    "transformer-based": [82.1, 81, 87.7, 71.4],
    # "transformer-based [37]": [82.1, 81, 87.7, 71.4],
    "Ours": [84.4, 83.2, 91.7, 77.7]
}
nemenyi_plot_multiple([results], [""], row=1, col=1)
