from diffusion.sampling import *


class BoltzmannResampler:
    def __init__(self, args, model=None):
        self.model = model
        self.args = args
        self.temp = args.temp

    def try_resample(self, data):
        try:
            times_seen = data.times_seen
        except:
            self.resample(data)
        if data.times_seen > max(5, int(data.ess)):
            self.resample(data)
        elif data.times_seen == max(5, int(data.ess)):
            if np.random.rand() < 0.5:
                self.resample(data)
        data.times_seen += 1

    def resample(self, data, temperature=None):
        T = temperature if temperature else self.temp
        kT = 1.38e-23 * 6.022e23 * T / 4148
        model, args = self.model, self.args
        model.eval()
        data.pos = data.pos[0]

        samples = []
        for i in range(args.boltzmann_confs):
            data_new = copy.deepcopy(data)
            samples.append(data_new)
        samples = perturb_seeds(samples)
        samples = sample(samples, model, steps=args.boltzmann_steps, ode=True,
                         sigma_max=args.sigma_max, sigma_min=args.sigma_min, likelihood=args.likelihood)

        data.pos = []
        logweights = []

        data.mol.RemoveAllConformers()
        for i, data_conf in enumerate(samples):
            mol = pyg_to_mol(data.mol, data_conf, mmff=False, rmsd=False)
            populate_likelihood(mol, data_conf, water=False, xtb=None)
            data.pos.append(data_conf.pos)
            energy = mol.mmff_energy
            logweights.append(-energy / kT - mol.euclidean_dlogp)

        weights = np.exp(logweights - np.max(logweights))
        data.weights = weights / weights.sum()
        data.ess = 1 / np.sum(data.weights ** 2)
        data.times_seen = 0
        model.train()
        return data.ess


class BaselineResampler:
    def __init__(self, ais_steps, temp, mcmc_sigma, n_samples):
        self.ais_steps = ais_steps
        self.temp = temp
        self.mcmc_sigma = mcmc_sigma
        self.n_samples = n_samples

    def logp_func(self, data):
        def logp(i, xi):
            kT = 1.38e-23 * 6.022e23 * self.temp / 4148
            data.pos = xi
            mol = pyg_to_mol(data.mol, data, rmsd=False, copy=False)
            populate_likelihood(mol, data, water=False, xtb=None)
            energy = mol.mmff_energy
            logp_start = mol.euclidean_dlogp
            logp_end = -energy / kT
            frac = i / (self.ais_steps + 1)
            logp_ = (1 - frac) * logp_start + frac * logp_end
            return {
                'logp': logp_,
                'E': energy,
                'jac': logp_start
            }

        return logp

    def transition_func(self, data):
        logp = self.logp_func(data)

        def transition(i, xi):
            oldlogp = logp(i, xi)
            torsion_updates = np.random.normal(loc=0, scale=self.mcmc_sigma, size=data.edge_mask.sum())
            xi_prop = modify_conformer(copy.deepcopy(xi), data.edge_index.T[data.edge_mask],
                                       data.mask_rotate, torsion_updates, as_numpy=True)
            newlogp = logp(i, xi_prop)
            loga = newlogp['logp'] - oldlogp['logp'] + oldlogp['jac'] - newlogp['jac']
            if np.random.rand() < np.exp(min(0, loga)):
                xi = xi_prop
            return xi

        return transition

    def single_sample(self, data):
        torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=data.edge_mask.sum())
        xi = modify_conformer(data.pos, data.edge_index.T[data.edge_mask],
                              data.mask_rotate, torsion_updates, as_numpy=True)

        logp = self.logp_func(data)
        transition = self.transition_func(data)
        logweight = logp(1, xi)['logp'] - logp(0, xi)['logp']
        for i in range(1, self.ais_steps + 1):
            xi = transition(i, xi)
            logweight += logp(i + 1, xi)['logp'] - logp(i, xi)['logp']

        return xi, logweight

    def resample(self, data):
        data.pos = data.pos[0]
        samples = []
        logweights = []
        for _ in range(self.n_samples):
            data_new = copy.deepcopy(data)
            data_new, logweight = self.single_sample(data_new)
            samples.append(data_new)
            logweights.append(logweight)
        data.pos = samples
        weights = np.exp(logweights - np.max(logweights))
        data.weights = weights / weights.sum()
        data.ess = 1 / np.sum(data.weights ** 2)
        return data.ess
