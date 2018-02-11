import os

# for i in range(20, 220, 20):
	# os.mkdir('dev/logs/budget={}'.format(i))


# x = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
fig=plt.figure()
x= list(range(1,8,1))
plt.clf()
plt.errorbar(x, pomcp_mean, yerr=pomcp_err, fmt='-o',label='POMCP')
plt.errorbar(x, bpomcp_mean, yerr=bpomcp_err, fmt='-o',label='BPOMCP')
plt.legend(loc='upper left')
plt.xlabel('budget')
plt.ylabel('mean total reward')
plt.show()