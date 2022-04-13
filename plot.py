import glob
import matplotlib.pyplot as plt

lst = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 1e-7, 1e-8]
for lr in lst:
    filename = 'result_lr_{}'.format(lr)
    iters = []
    costs = []
    with open(filename, 'rt') as f:
        for i, line in enumerate(f.readlines(), 1):
            tokens = line.split()
            iters.append(i)
            costs.append(float(tokens[8]))

        plt.plot(iters, costs, label=r'$\alpha$={}'.format(filename[10:]))

plt.xlabel('updates')
plt.ylabel('cost')
plt.legend(loc='best')
plt.show()

