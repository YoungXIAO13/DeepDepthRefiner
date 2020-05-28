import numpy as np
import matplotlib.pyplot as plt


#dataset = 'NYU'
dataset = 'iBims'

#metric = 'acc'
metric = 'comp'

if dataset == 'NYU':
    labels = ['Eigen', 'Laina', 'DORN', 'SharpNet', 'Jiao', 'Yin']

    if metric == 'acc':
        init = [9.926, 4.702, 3.872, 3.041, 8.730, 1.854]
        du = [2.168, 2.372, 3.001, 1.838, 2.410, 1.762]
        xiao = [1.715, 1.976, 2.631, 1.546, 1.985, 1.544]
    else:
        init = [9.993, 8.982, 8.117, 8.692, 9.864, 7.188]
        du = [8.173, 7.041, 7.242, 6.730, 8.230, 6.307]
        xiao = [6.048, 6.423, 6.507, 5.988, 6.990, 5.453]

else:
    labels = ['Eigen', 'Laina', 'Liu', 'Li', 'PlaneNet', 'SharpNet']

    if metric == 'acc':
        init = [9.97, 6.19, 2.42, 3.90, 4.84, 3.69]
        du = [4.83, 3.32, 2.36, 3.43, 2.78, 2.13]
        xiao = [2.46, 2.56, 2.37, 2.07, 2.75, 2.16]
    else:
        init = [9.99, 9.17, 7.11, 8.17, 8.86, 7.82]
        du = [8.78, 7.15, 7., 7.19, 7.65, 6.33]
        xiao = [5.74, 6.20, 5.91, 5.26, 6.4, 5.82]


# generate figure
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.2 * width, init, width, label='init', zorder=3)
rects2 = ax.bar(x, du, width, label='du', zorder=3)
rects3 = ax.bar(x + 1.2 * width, xiao, width, label='ours', zorder=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('err_acc')
ax.grid(zorder=0, axis='y')

#ax.set_title('Occlusion boundary accuracy obtained by different methods')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(None,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()

fig.savefig('{}_err_{}.eps'.format(dataset, metric))
plt.close(fig)
