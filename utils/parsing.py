from argparse import ArgumentParser


def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./test_run', help='Folder in which to save model and logs')
    parser.add_argument('--restart_dir', type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--cache', type=str, default='data/DRUGS/cache', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--data_dir', type=str, default='data/DRUGS/drugs/', help='Folder containing original conformers')
    parser.add_argument('--std_pickles', type=str, default='data/DRUGS/standardized_pickles', help='Folder in which the pickle are put after standardisation/matching')
    parser.add_argument('--split_path', type=str, default='data/DRUGS/split.npy', help='Path of file defining the split')
    parser.add_argument('--dataset', type=str, default='drugs', help='drugs or qm9')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=250, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--optimizer', type=str, default='adam', help='Adam optimiser only one supported')
    parser.add_argument('--scheduler', type=str, default='plateau', help='LR scehduler: plateau or none')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of plateau scheduler')
    parser.add_argument('--sigma_min', type=float, default=0.01*3.14, help='Minimum sigma used for training')
    parser.add_argument('--sigma_max', type=float, default=3.14, help='Maximum sigma used for training')
    parser.add_argument('--limit_train_mols', type=int, default=0, help='Limit to the number of molecules in dataset, 0 uses them all')
    parser.add_argument('--boltzmann_weight', action='store_true', default=False, help='Whether to sample conformers based on B.w.')

    # Feature arguments
    parser.add_argument('--in_node_features', type=int, default=74, help='Dimension of node features: 74 for drugs and xl, 44 for qm9')
    parser.add_argument('--in_edge_features', type=int, default=4, help='Dimension of edge feature (do not change)')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Dimension of sinusoidal embedding of sigma')
    parser.add_argument('--radius_embed_dim', type=int, default=50, help='Dimension of embedding of distances')
    
    # Model arguments
    parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=32, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=8, help='Number of hidden features per node of orser >0')
    parser.add_argument('--no_residual', action='store_true', default=False, help='If set, it removes residual connection')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')

    # Boltzmann training arguments
    parser.add_argument('--boltzmann_training', action='store_true', default=False, help='Set to true for torsional Boltzmann training')
    parser.add_argument('--boltzmann_confs', type=int, default=32, help='Number of conformers to generate at each resampling step')
    parser.add_argument('--boltzmann_steps', type=int, default=5, help='Number of inference steps used by the resampler')
    parser.add_argument('--likelihood', type=str, default='full', help='Method to evaluate likelihood: full (default) or hutch')
    parser.add_argument('--temp', type=int, default=300, help='Temperature used for Boltzmann weight')
    parser.add_argument('--adjust_temp', action='store_true', default=False, help='Whether to perform the temperature annealing during training')

    args = parser.parse_args()
    return args
