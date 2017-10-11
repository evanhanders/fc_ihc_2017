import numpy as np
import matplotlib.pyplot as plt

data_r = np.genfromtxt('data_files/onsets_v_r.csv', delimiter=',', skip_header=1)
data_eps = np.genfromtxt('data_files/onsets_v_epsilon.csv', delimiter=',', skip_header=1)
data_n_rho = np.genfromtxt('data_files/onsets_v_n_rho.csv', delimiter=',', skip_header=1)

fig = plt.figure(figsize=(8, 9))

#### Variation of r
ax = fig.add_subplot(3,2,1)
bx = fig.add_subplot(3,2,2)

r, ra_crit, kx_crit, n_rho = data_r[:,1], data_r[:,4], data_r[:,5], data_r[:,3]

ax.plot(r[n_rho == 1], ra_crit[n_rho == 1], label=r'$n_\rho = 1$', color='indigo', lw=2)
bx.plot(r[n_rho == 1], kx_crit[n_rho == 1], label=r'$n_\rho = 1$', color='indigo', lw=2)
ax.plot(r[n_rho == 3], ra_crit[n_rho == 3], label=r'$n_\rho = 3$', dashes=(5,1), color='green', lw=2)
bx.plot(r[n_rho == 3], kx_crit[n_rho == 3], label=r'$n_\rho = 3$', dashes=(5,1), color='green', lw=2)
ax.plot(r[n_rho == 5], ra_crit[n_rho == 5], label=r'$n_\rho = 5$', dashes=(5,1,2,1,2,1), color='tomato', lw=2)
bx.plot(r[n_rho == 5], kx_crit[n_rho == 5], label=r'$n_\rho = 5$', dashes=(5,1,2,1,2,1), color='tomato', lw=2)

plt.axes(ax)
plt.legend(loc='best', frameon=False, fontsize=10, labelspacing=0.1, handlelength=2.4)
plt.axes(bx)
plt.legend(loc='best', frameon=False, fontsize=10, labelspacing=0.1, handlelength=2.4)


ax.annotate('(a)', xy=(0.98*ax.get_xlim()[0] + 0.02*ax.get_xlim()[1], 0.9*ax.get_ylim()[1]+ 0.1*ax.get_ylim()[0]))
bx.annotate('(d)', xy=(0.92*bx.get_xlim()[1] + 0.08*bx.get_xlim()[0], 0.96*bx.get_ylim()[0] + 0.04*bx.get_ylim()[1]))
ax.set_xlabel(r'$r$')
bx.set_xlabel(r'$r$')
ax.set_ylabel(r'$\mathrm{Ra}_{\mathrm{crit}}$')
bx.set_ylabel(r'$\frac{L_z}{2\pi}\mathrm{k}_{\mathrm{crit}}$')


#### Variation of epsilon
ax = fig.add_subplot(3,2,3)
bx = fig.add_subplot(3,2,4)

eps, ra_crit, kx_crit, n_rho, r = data_eps[:,0], data_eps[:,4], data_eps[:,5], data_eps[:,1], data_eps[:,2]

ax.plot(eps[(n_rho == 1)*(r == 0)], ra_crit[(n_rho == 1)*(r==0)], label=r'$r = 0;\,n_\rho=1 $', color='darkorchid', lw=2)
bx.plot(eps[(n_rho == 1)*(r == 0)], kx_crit[(n_rho == 1)*(r==0)], label=r'$r = 0;\,n_\rho=1 $', color='darkorchid', lw=2)
ax.plot(eps[(n_rho == 1)*(r == 0.3)], ra_crit[(n_rho == 1)*(r==0.3)], label=r'$r = 0.3;\,n_\rho=1 $', dashes=(5,1), color='olive', lw=2)
bx.plot(eps[(n_rho == 1)*(r == 0.3)], kx_crit[(n_rho == 1)*(r==0.3)], label=r'$r = 0.3;\,n_\rho=1 $', dashes=(5,1), color='olive', lw=2)
ax.plot(eps[(n_rho == 3)*(r == 0)], ra_crit[(n_rho == 3)*(r==0)], label=r'$r = 0;\,n_\rho=3 $', dashes=(5,1,2,1,2,1), color='peru', lw=2)
bx.plot(eps[(n_rho == 3)*(r == 0)], kx_crit[(n_rho == 3)*(r==0)], label=r'$r = 0;\,n_\rho=3 $', dashes=(5,1,2,1,2,1), color='peru', lw=2)

ax.set_xscale('log')
bx.set_xscale('log')
plt.axes(ax)
plt.legend(loc='upper left', frameon=False, fontsize=10, labelspacing=0.1, handlelength=2.4)
plt.axes(bx)
plt.legend(loc='upper left', frameon=False, fontsize=10, labelspacing=0.1, handlelength=2.4)

ax.set_xlim(np.min(eps), np.max(eps))
bx.set_xlim(np.min(eps), np.max(eps))

bx.set_yticks((0.29, 0.30))

ax.annotate('(b)', xy=(10**(0.92*np.log10(ax.get_xlim()[1]) + 0.08*np.log10(ax.get_xlim()[0])), 0.9*ax.get_ylim()[1] + 0.1*ax.get_ylim()[0]))
bx.annotate('(e)', xy=(10**(0.92*np.log10(ax.get_xlim()[1]) + 0.08*np.log10(ax.get_xlim()[0])), 0.9*bx.get_ylim()[1] + 0.1*bx.get_ylim()[0]))

ax.set_xlabel(r'$\epsilon$')
bx.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$\mathrm{Ra}_{\mathrm{crit}}$')
bx.set_ylabel(r'$\frac{L_z}{2\pi}\mathrm{k}_{\mathrm{crit}}$', labelpad=-2)

#### Variation of n_rho
ax = fig.add_subplot(3,2,5)
bx = fig.add_subplot(3,2,6)

n_rho, ra_crit, kx_crit, r = data_n_rho[:,0], data_n_rho[:,4], data_n_rho[:,5], data_n_rho[:,3]

ax.plot(np.exp(n_rho[r == 0])-1, ra_crit[r == 0], label=r'$r = 0$', color='royalblue', lw=2)
bx.plot(np.exp(n_rho[r == 0])-1, kx_crit[r == 0], label=r'$r = 0$', color='royalblue', lw=2)
ax.plot(np.exp(n_rho[r == 0.3])-1, ra_crit[r == 0.3], label=r'$r = 0.3$', color='darkorange', dashes=(5,1), lw=2)
bx.plot(np.exp(n_rho[r == 0.3])-1, kx_crit[r == 0.3], label=r'$r = 0.3$', color='darkorange', dashes=(5,1), lw=2)
ax.plot(np.exp(n_rho[r == 2])-1, ra_crit[r == 2], label=r'$r = 2$', color='darkseagreen', dashes=(5,1,2,1,2,1), lw=2)
bx.plot(np.exp(n_rho[r == 2])-1, kx_crit[r == 2], label=r'$r = 2$', color='darkseagreen', dashes=(5,1,2,1,2,1), lw=2)

ax.set_xlabel(r'$e^{n_\rho}-1$')
bx.set_xlabel(r'$e^{n_\rho}-1$')
ax.set_ylabel(r'$\mathrm{Ra}_{\mathrm{crit}}$')
bx.set_ylabel(r'$\frac{L_z}{2\pi}\mathrm{k}_{\mathrm{crit}}$')

ax.set_xlim(np.min(np.exp(n_rho)-1), np.max(np.exp(n_rho)-1))
bx.set_xlim(np.min(np.exp(n_rho)-1), np.max(np.exp(n_rho)-1))


ax.set_xscale('log')
bx.set_xscale('log')

ax.annotate('(c)', xy=(10**(0.92*np.log10(ax.get_xlim()[1]) + 0.08*np.log10(ax.get_xlim()[0])), 0.9*ax.get_ylim()[1]+ 0.1*ax.get_ylim()[0]))
bx.annotate('(f)', xy=(10**(0.97*np.log10(ax.get_xlim()[0]) + 0.03*np.log10(ax.get_xlim()[1])), 0.9*bx.get_ylim()[1] + 0.1*bx.get_ylim()[0]))

plt.axes(ax)
plt.legend(loc='best', frameon=False, fontsize=10, labelspacing=0.1, handlelength=2.4)
plt.axes(bx)
plt.legend(loc='center right', frameon=False, fontsize=10, labelspacing=0.1, handlelength=2.4)

plt.subplots_adjust(left=0, right=0.95, top=0.95, bottom=0)
plt.savefig('figs/onset_figure.png', bbox_inches='tight', dpi=300)
