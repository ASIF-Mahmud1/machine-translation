import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
#,figsize=(10,15)

def plot_chart_for_all_metrics(table):
    table.plot.bar(x="Dataset Size",figsize=(15,10))
    plt.show(block=True)

def plotBarChart(x, height, title):
    plt.bar(x = x, height = height)
    plt.title(title)
    plt.ylim((0,1))
    plt.show()