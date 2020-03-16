import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


def plot_agreement(list1,
                   list2,
                   title='Agreement beteween method 1 and method 2',
                   ylim1=None,
                   ylim2=None,
                   xlim1=None,
                   xlim2=None):
    """Agreement between the measurements of two methods.

    :param list1: measurements from method 1
    :param list2: measurements from method 2
    :param title: title of plot
    :param ylim1: lower limit for y-axis
    :param ylim2: upper limit for y-axis
    :param xlim1: lower limit for x-axis
    :param xlim2: upper limit for x-axis


    Lists should be indexed accordingly, e.g. list1[0] and list2[0] should be the measurement obtained for the same
    layer using each method, respectively.

    Example usage:
    list1 = [x for x in np.random.rand(100)]
    list2 = [x for x in np.random.rand(100)]
    plot_agreement(list1, list2, 'A title ')
    """

    assert len(list1) == len(list2), "Lenghts of measurements should be equal"

    f, ax = plt.subplots()
    sm.graphics.mean_diff_plot(np.array(list1), np.array(list2), ax=ax)

    if ylim1 is not None and ylim2 is not None and xlim1 is not None and xlim2 is not None:
        plt.ylim([ylim1, ylim2])
        plt.xlim([xlim1, xlim2])
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.title(title, fontweight = 'bold', fontsize=18)
    ax = plt.gca()
    ax.set_xlabel('Means', fontweight = 'bold')
    ax.set_ylabel('Difference', fontweight = 'bold')
    plt.tight_layout()

    plt.show()
