import matplotlib.pyplot as plt

def plot_privacy_performance_tradeoff(privacy_scores, performance_scores, labels=None):
    plt.figure(figsize=(6,4))
    x = performance_scores
    y = privacy_scores
    if labels is None:
        labels = [str(i) for i in range(len(x))]
    for xi, yi, lab in zip(x, y, labels):
        plt.scatter(xi, yi)
        plt.annotate(lab, (xi, yi))
    plt.xlabel('Performance (e.g., 1/Latency)')
    plt.ylabel('Privacy (e.g., PDR↑, SELS↓)')
    plt.tight_layout()
    return plt.gcf()


