import argparse
import datetime
import importlib
import os
import random
import uuid

import numpy as np
import torch

from utils.iteration_diagnostics import (
    aggregate_online_diagnostics,
    annotate_iteration_summary,
    save_prediction_results,
)
from utils.run_config import (
    annotate_run_config,
    save_run_config,
    validate_online_feedback_protocol,
)


# from exp.exp_online import Exp_TS2VecSupervised


def init_dl_program(
        args,
        seed=None,
        use_cudnn=True,
        deterministic=False,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    device_name = args.gpu
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        if importlib.util.find_spec("mkl") is not None:
            import mkl
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    if args.use_gpu:
        devices = []
        for t in reversed(device_name):
            t_device = torch.device(t)
            devices.append(t_device)
            if t_device.type == 'cuda':
                assert torch.cuda.is_available()
                torch.cuda.set_device(t_device)
        devices.reverse()
        torch.backends.cudnn.enabled = use_cudnn
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = use_tf32
            torch.backends.cuda.matmul.allow_tf32 = use_tf32

        return devices if len(devices) > 1 else devices[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser.add_argument('--data', type=str, default='ETTh2', help='data')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=0, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--skip_test', action='store_true', default=False, help='skip test stage after training')
    parser.add_argument(
        '--pretrain_mode',
        type=str,
        default='retrain',
        choices=['retrain', 'load', 'none'],
        help=(
            'retrain before test, load an existing checkpoint, or use '
            'random initialization with online adaptation only'
        ),
    )
    parser.add_argument('--pretrained_checkpoint', type=str, default='',
                        help='checkpoint path used when --pretrain_mode load')
    parser.add_argument('--checkpoint_tag', type=str, default='',
                        help='optional tag used to isolate incompatible checkpoint variants')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--seed', type=int, default=0,
                        help='base seed; iteration ii uses seed + ii')
    parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='optimizer learning rate')
    parser.add_argument('--learning_rate_expert', type=float, default=None,
                        help='optimizer learning rate for expert parameters')
    parser.add_argument('--learning_rate_router', type=float, default=None,
                        help='optimizer learning rate for router parameters')
    parser.add_argument('--online_lr_expert', type=float, default=1e-4,
                        help='base expert learning rate during online testing')
    parser.add_argument('--online_lr_router', type=float, default=1e-5,
                        help='base router learning rate during online testing')
    parser.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--method', type=str, default='onenet_fsnet')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--tcn_output_dim', type=int, default=320, help='decomposition-kernel')
    parser.add_argument('--tcn_layer', type=int, default=2, help='decomposition-kernel')
    parser.add_argument('--tcn_hidden', type=int, default=160, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')

    parser.add_argument('--teacher_forcing', action='store_true', help='use teacher forcing during forecasting',
                        default=False)
    parser.add_argument('--online_learning', type=str, default='full')
    parser.add_argument('--opt', type=str, default='adam')

    parser.add_argument('--test_bsz', type=int, default=1)
    parser.add_argument('--n_inner', type=int, default=1)
    parser.add_argument('--num_experts', type=int, default=4, help='number of experts for multi-expert OneNet variants')
    parser.add_argument('--expert_composition', type=str, default='mixed',
                        choices=['mixed', 'fsnet', 'fsnet_time'],
                        help='Expert architecture composition; mixed preserves legacy allocation')
    parser.add_argument('--top_k', type=int, default=4, help='top-k experts selected by MoE router')
    parser.add_argument('--lambda_div', type=float, default=0.0, help='weight of expert diversity loss during pretraining')
    parser.add_argument('--tsb_alpha', type=float, default=0.5, help='EMA factor for online TSB smoothing')
    parser.add_argument('--tsb_eps', type=float, default=1e-8, help='epsilon for online TSB projection')
    parser.add_argument('--tsb_buffer_size', type=int, default=8, help='buffer size for batched online TSB reference gradient')
    parser.add_argument('--disable_tsb', action='store_true', default=False,
                        help='disable all TSB reference gradients, smoothing, and conflict filtering')
    parser.add_argument('--disable_tsb_smoothing', action='store_true', default=False,
                        help='disable TSB gradient smoothing while retaining optional conflict filtering')
    parser.add_argument('--disable_tsb_conflict_filter', action='store_true', default=False,
                        help='disable TSB conflict projection while retaining optional smoothing')
    parser.add_argument('--adaptive_controller', type=str, default='dynamic',
                        choices=['fixed', 'dynamic'],
                        help='fixed or error-adaptive online Expert/Router learning rates')
    parser.add_argument('--expert_grad_clip', type=float, default=1.0)
    parser.add_argument('--router_grad_clip', type=float, default=0.5)
    parser.add_argument('--router_temperature', type=float, default=2.0)
    parser.add_argument('--router_entropy_weight', type=float, default=1e-3)
    parser.add_argument('--progressive_fb', action='store_true', default=False,
                        help='release one newly matured timestamp per rolling origin')
    parser.add_argument(
        '--progressive_baseline_fb',
        action='store_true',
        default=False,
        help='enable progressive causal feedback for FSNet/OneNet/DynaME only',
    )
    parser.add_argument('--router_granularity', type=str, default='channel',
                        choices=['channel', 'horizon_channel'],
                        help='router weight granularity; channel preserves legacy checkpoints')
    parser.add_argument('--correction_lr', type=float, default=0.1,
                        help='learning rate for progressive dual routing correction')
    parser.add_argument('--correction_decay', type=float, default=0.01,
                        help='per-origin decay applied once to routing correction')
    parser.add_argument('--correction_grad_clip', type=float, default=10.0,
                        help='elementwise clip for centered mixture gradients')
    parser.add_argument('--correction_logit_clip', type=float, default=5.0,
                        help='absolute clip for online routing correction logits')
    parser.add_argument('--local_credit_temperature', type=float, default=1.0)
    parser.add_argument('--sample_credit_temperature', type=float, default=1.0)
    parser.add_argument('--local_credit_weight', type=float, default=0.1)
    parser.add_argument('--min_credit_eps', type=float, default=1e-8)
    parser.add_argument('--capability_sketch_dim', type=int, default=32)
    parser.add_argument('--capability_sketch_seed', type=int, default=2025)
    parser.add_argument('--responsibility_threshold', type=float, default=0.3)
    parser.add_argument('--alignment_threshold', type=float, default=0.8)
    parser.add_argument('--credit_top_k', type=int, default=1)
    parser.add_argument('--stable_buffer_size', type=int, default=32)
    parser.add_argument('--recovery_buffer_size', type=int, default=32)
    parser.add_argument('--buffer_duplicate_threshold', type=float, default=0.98)
    parser.add_argument('--recovery_failure_penalty', type=float, default=0.5)
    parser.add_argument('--max_recovery_attempts', type=int, default=3,
                        help='maximum failed Recovery attempts; evict on the max-th failure')
    parser.add_argument('--recovery_degradation_margin', type=float, default=0.0,
                        help='relative loss margin required for harmful capability drift')
    parser.add_argument('--buffer_storage_dtype', type=str, default='fp16',
                        choices=['fp16', 'fp32'])
    parser.add_argument('--memory_refresh_interval', type=int, default=100)
    parser.add_argument('--promote_alignment_threshold', type=float, default=0.9)
    parser.add_argument('--promote_loss_threshold', type=float, default=1.0)
    parser.add_argument('--recovery_batch_size', type=int, default=2)
    parser.add_argument('--recovery_loss_weight', type=float, default=0.1)
    parser.add_argument('--recovery_sketch_weight', type=float, default=1.0)
    parser.add_argument('--subspace_scope', type=str, default='regressor',
                        choices=['regressor'])
    parser.add_argument('--subspace_rank', type=int, default=0,
                        help='fixed rank; 0 selects rank by energy')
    parser.add_argument('--subspace_max_rank', type=int, default=32)
    parser.add_argument('--subspace_energy_threshold', type=float, default=0.95)
    parser.add_argument('--subspace_refresh_interval', type=int, default=100)
    parser.add_argument('--subspace_min_samples', type=int, default=4)
    parser.add_argument('--subspace_eps', type=float, default=1e-8)
    parser.add_argument('--subspace_lambda', type=float, default=1e4)
    parser.add_argument('--subspace_evidence_mass_scale', type=float, default=8.0,
                        help='saturation scale for normalized Stable evidence mass')
    parser.add_argument('--subspace_gamma_min', type=float, default=0.0)
    parser.add_argument('--subspace_gamma_max', type=float, default=1.0)
    parser.add_argument('--expert_update_strategy', type=str, default='tsb',
                        choices=['plain', 'tsb', 'subspace', 'hybrid'])
    parser.add_argument('--disable_version_awareness', action='store_true',
                        default=False)
    parser.add_argument('--disable_directional_recovery', action='store_true',
                        default=False,
                        help='restore alignment-only Recovery admission and stopping')
    parser.add_argument('--disable_recovery', action='store_true', default=False)
    parser.add_argument('--disable_credit_weighted_subspace', action='store_true',
                        default=False,
                        help='use uniform Stable-sample weights only for subspace covariance geometry; memory admission and evidence-adaptive gamma remain enabled')
    parser.add_argument('--disable_online_correction', action='store_true',
                        default=False)
    parser.add_argument('--disable_expert_online_update', action='store_true',
                        default=False,
                        help='freeze Experts online while retaining Router updates')
    parser.add_argument('--robust_fallback_threshold', type=float, default=0.0,
                        help='deprecated compatibility option; v3 only replaces non-finite predictions')
    parser.add_argument('--online_log_interval', type=int, default=500)
    parser.add_argument('--credit_diagnostic_buffer_size', type=int, default=10000)
    parser.add_argument('--dynamic_comparator', action='store_true', default=False,
                        help='enable offline empirical K-switch comparator evaluation')
    parser.add_argument('--dynamic_comparator_max_switches', type=int, default=1,
                        help='maximum switches allowed by the offline dynamic comparator')
    parser.add_argument('--dynamic_comparator_max_points', type=int, default=64,
                        help='maximum ordered blocks used by K-switch O(T^2) evaluation')
    parser.add_argument('--max_online_steps', type=int, default=-1,
                        help='limit test origins for smoke tests; -1 is unlimited')
    parser.add_argument('--strict_online_checks', action='store_true', default=False,
                        help='enable expensive progressive runtime invariant checks')
    # DynaME baseline
    parser.add_argument('--dyname_period_num', type=int, default=2)
    parser.add_argument('--dyname_krr_lambda', type=float, default=1e-4)
    parser.add_argument('--dyname_krr_train_num', type=int, default=8)
    parser.add_argument('--dyname_temperature', type=float, default=1.0)
    parser.add_argument('--dyname_beta', type=float, default=0.3)
    parser.add_argument('--dyname_delta', type=float, default=0.01)
    parser.add_argument('--dyname_past_num', type=int, default=672)
    parser.add_argument('--dyname_online_lr', type=float, default=1e-3)
    # PatchTST-DGrad baseline
    parser.add_argument('--dgrad_online_lr', type=float, default=1e-3)
    parser.add_argument('--dgrad_grad_clip', type=float, default=1.0)
    # Shared online baselines
    parser.add_argument('--baseline_online_lr', type=float, default=1e-3)
    parser.add_argument('--replay_buffer_size', type=int, default=128)
    parser.add_argument('--replay_batch_size', type=int, default=16)
    parser.add_argument('--dsof_student_width', type=int, default=16)
    parser.add_argument('--dsof_student_lr', type=float, default=1e-3)
    parser.add_argument('--dsof_td_weight', type=float, default=0.2)
    parser.add_argument('--proceed_concept_dim', type=int, default=200)
    parser.add_argument('--proceed_bottleneck_dim', type=int, default=32)
    parser.add_argument('--proceed_online_lr', type=float, default=1e-3)
    parser.add_argument('--channel_cross', type=bool, default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--finetune_model_seed', type=int)

    parser.add_argument('--aug', type=int, default=0, help='Training with augmentation data aug iterations')
    parser.add_argument('--lr_test', type=float, default=1e-3, help='learning rate during test')

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Wavelets',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')

    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--m', type=int, default=24)
    parser.add_argument('--loss_aug', type=float, default=0.5, help='weight for augmentation loss')
    parser.add_argument('--use_adbfgs', action='store_true', help='use the Adbfgs optimizer', default=True)
    parser.add_argument('--period_len', type=int, default=12)
    parser.add_argument('--mlp_depth', type=int, default=3)
    parser.add_argument('--mlp_width', type=int, default=256)
    parser.add_argument('--station_lr', type=float, default=0.0001)

    parser.add_argument('--sleep_interval', type=int, default=1, help='latent dimension of koopman embedding')
    parser.add_argument('--sleep_epochs', type=int, default=1, help='latent dimension of koopman embedding')
    parser.add_argument('--sleep_kl_pre', type=float, default=0, help='latent dimension of koopman embedding')
    parser.add_argument('--delay_fb', action='store_true', default=False, help='use delayed feedback')
    parser.add_argument('--online_adjust', type=float, default=0.0, help='latent dimension of koopman embedding')
    parser.add_argument('--offline_adjust', type=float, default=0.0, help='latent dimension of koopman embedding')
    parser.add_argument('--online_adjust_var', type=float, default=0.0, help='latent dimension of koopman embedding')
    parser.add_argument('--var_weight', type=float, default=0.0, help='latent dimension of koopman embedding')
    parser.add_argument('--alpha_w', type=float, default=0.0001, help='spectrum filter ratio')
    parser.add_argument('--alpha_d', type=float, default=0.003, help='spectrum filter ratio')
    parser.add_argument('--test_lr', type=float, default=0.1, help='spectrum filter ratio')
    args = parser.parse_args()

    # Backward-compatible defaults:
    # if specific lrs are not provided, both expert/router reuse learning_rate.
    if args.learning_rate_expert is None:
        args.learning_rate_expert = args.learning_rate
    if args.learning_rate_router is None:
        args.learning_rate_router = args.learning_rate
    # Keep old training schedule behavior (adjust_learning_rate uses args.learning_rate)
    # by mapping base learning rate to expert learning rate.
    args.learning_rate = args.learning_rate_expert

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.test_bsz = args.batch_size if args.test_bsz == -1 else args.test_bsz
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        'Toy': {'data': 'Toy.csv', 'T': 'Value', 'S': [1, 1, 1]},
        'ToyG': {'data': 'ToyG.csv', 'T': 'Value', 'S': [1, 1, 1]},
        'Exchange': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8, 8]},
        'Illness': {'data': 'national_illness.csv', 'T': 'OT', 'M': [7, 7, 7]},
        'Traffic': {'data': 'traffic.csv', 'T': 'OT', 'M': [862, 862, 862]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args)

    return args


def prepare_experiment_for_run(exp, args, setting):
    """Apply the requested offline initialization policy before testing."""

    if args.pretrain_mode == 'load':
        if not args.pretrained_checkpoint:
            raise ValueError(
                '--pretrained_checkpoint is required when --pretrain_mode load'
            )
        print(
            '>>>>>>>load pretrained checkpoint : '
            '{}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
                args.pretrained_checkpoint
            )
        )
        exp.load_pretrained(args.pretrained_checkpoint)
        return
    if args.pretrain_mode == 'retrain':
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        return
    if args.pretrain_mode == 'none':
        print(
            '>>>>>>>no offline pretraining; random initialization + '
            'online adaptation : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting)
        )
        prepare = getattr(exp, 'prepare_without_pretraining', None)
        if prepare is not None:
            prepare()
        return
    raise ValueError('invalid pretrain_mode: {}'.format(args.pretrain_mode))


if __name__ == '__main__':
    args = parse_args()
    causal_feedback_protocol = validate_online_feedback_protocol(args)
    print("Causal feedback protocol:", causal_feedback_protocol)

    Exp = getattr(
        importlib.import_module('exp.exp_{}'.format(args.method)),
        'Exp_TS2VecSupervised',
    )
    metrics, preds, true, mae, mse = [], [], [], [], []
    diagnostic_summaries = []

    method_name = args.method
    if args.progressive_baseline_fb:
        method_name = '{}_progfb'.format(method_name)
    if args.checkpoint_tag:
        method_name = '{}_{}'.format(method_name, args.checkpoint_tag)
    result_setting = '{}_{}_pl{}_ol{}_opt{}_tb{}'.format(
        method_name,
        args.data,
        args.pred_len,
        args.online_learning,
        args.opt,
        args.test_bsz,
    )
    result_root = './result/'
    os.makedirs(result_root, exist_ok=True)
    next_idx = 1
    for name in os.listdir(result_root):
        if name.startswith('results'):
            index_text = name[len('results'):]
            if index_text.isdigit():
                next_idx = max(next_idx, int(index_text) + 1)
    folder_path = os.path.join(
        result_root, 'results{}'.format(next_idx), result_setting
    )
    os.makedirs(folder_path, exist_ok=True)

    for ii in range(args.itr):
        print('\n ====== Run {} ====='.format(ii))
        uid = uuid.uuid4().hex[:4]
        suffix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
        setting = '{}_{}_pl{}_ol{}_opt{}_tb{}_{}'.format(
            method_name,
            args.data,
            args.pred_len,
            args.online_learning,
            args.opt,
            args.test_bsz,
            suffix,
        )

        iteration_seed = args.seed + ii
        init_dl_program(args, seed=iteration_seed)
        args.finetune_model_seed = iteration_seed
        iteration_folder = os.path.join(folder_path, 'itr_{}'.format(ii))
        run_config_path = save_run_config(
            args, iteration_folder, iteration_index=ii
        )
        print('run config:', run_config_path)
        exp = Exp(args)
        prepare_experiment_for_run(exp, args, setting)
        if args.skip_test:
            print('>>>>>>>testing skipped : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            iteration_metrics = [np.nan] * 6
            metrics.append(iteration_metrics)
            mae.append(np.nan)
            mse.append(np.nan)
            save_prediction_results(
                iteration_folder,
                iteration_metrics,
                np.asarray([]),
                np.asarray([]),
                np.asarray(np.nan),
                np.asarray(np.nan),
            )
            torch.cuda.empty_cache()
            continue

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        iteration_metrics, mae_curve, mse_curve, prediction, target = exp.test(
            setting
        )
        metrics.append(iteration_metrics)
        preds.append(prediction)
        true.append(target)
        mae.append(mae_curve)
        mse.append(mse_curve)
        save_prediction_results(
            iteration_folder,
            iteration_metrics,
            prediction,
            target,
            mae_curve,
            mse_curve,
        )
        if (
            args.progressive_baseline_fb
            and hasattr(exp, 'save_progressive_baseline_diagnostics')
        ):
            protocol_path = exp.save_progressive_baseline_diagnostics(
                iteration_folder
            )
            protocol_summary = exp.progressive_protocol_diagnostics.as_dict()
            annotate_run_config(
                run_config_path,
                {
                    'seed': iteration_seed,
                    'optimizer_step_count': protocol_summary[
                        'optimizer_step_count'
                    ],
                    'released_event_count': protocol_summary[
                        'released_event_count'
                    ],
                    'pending_records_at_end': protocol_summary[
                        'pending_records_at_end'
                    ],
                },
            )
            print('progressive baseline diagnostics:', protocol_path)
        if hasattr(exp, 'save_online_diagnostics') and args.progressive_fb:
            npz_path, json_path = exp.save_online_diagnostics(
                iteration_folder
            )
            metadata = {
                'index': ii,
                'seed': iteration_seed,
                'dataset': args.data,
                'pred_len': args.pred_len,
                'expert_update_strategy': args.expert_update_strategy,
                'processed_origins': getattr(
                    exp, 'processed_online_origins', None
                ),
                'completed_record_count': getattr(
                    exp, 'completed_record_count', None
                ),
                'early_ended': getattr(
                    exp, 'online_test_early_ended', None
                ),
                'strict_checks_enabled': args.strict_online_checks,
                'strict_checks_passed': (
                    getattr(exp.online_checker, 'failure_count', 0) == 0
                ),
                'strict_check_failure_count': getattr(
                    exp.online_checker, 'failure_count', 0
                ),
                'stable_buffer_capacity': args.stable_buffer_size,
                'recovery_buffer_capacity': (
                    0 if args.disable_recovery
                    else args.recovery_buffer_size
                ),
                'subspace_max_rank': args.subspace_max_rank,
            }
            annotate_iteration_summary(json_path, metadata)
            diagnostic_summaries.append(json_path)
            print('online diagnostics:', npz_path, json_path)
        torch.cuda.empty_cache()

    metrics_array = np.asarray(metrics)
    np.save(os.path.join(folder_path, 'metrics.npy'), metrics_array)
    np.save(os.path.join(folder_path, 'preds.npy'), np.asarray(preds))
    np.save(os.path.join(folder_path, 'trues.npy'), np.asarray(true))
    np.save(os.path.join(folder_path, 'mae.npy'), np.asarray(mae))
    np.save(os.path.join(folder_path, 'mse.npy'), np.asarray(mse))
    np.savez_compressed(
        os.path.join(folder_path, 'aggregate_metrics.npz'),
        metrics_mean=np.nanmean(metrics_array, axis=0),
        metrics_std=np.nanstd(metrics_array, axis=0),
        metrics=metrics_array,
    )
    if diagnostic_summaries:
        aggregate_online_diagnostics(
            diagnostic_summaries,
            os.path.join(
                folder_path, 'aggregate_diagnostics_summary.json'
            ),
        )
    print('RESULT_DIR: {}'.format(os.path.abspath(folder_path)))
