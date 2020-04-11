"""
Jinsung Yoon (9/13/2018)
PATE-GAN: Synthetic Data Generation 
"""
import numpy as np
from PATE_GAN import PATE_GAN
import argparse
import os
import time
import pandas as pd
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


# gpu_frac = 0.5
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
# set_session(tf.Session(config=config))


def df_split_random(df, train_rate=0.8):
    idx = np.random.permutation(len(df))

    train_idx = idx[:int(train_rate * len(df))]
    test_idx = idx[int(train_rate * len(df)):]
        
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx, :]
    return df_train, df_test


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--otrain")
    parser.add_argument("--otest")
    parser.add_argument("--itrain")
    parser.add_argument("--itest")
    parser.add_argument("--iter", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=1)
    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--teachers", type=int, default=5)
    parser.add_argument(
        "--target",
        help='name of response var when using csv as input')
    parser.add_argument(
        "--separator",
        default=',',
        help="separator csv file")
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()
    target = args.target.split(',')
    sep = args.separator

    fn_i_train = args.itrain
    fn_i_test = args.itest
    fn_o_train = args.otrain
    fn_o_test = args.otest

    num_teachers = args.teachers
    niter = args.iter
    epsilon = args.epsilon
    delta = args.delta

    assert fn_i_train is not None
    assert fn_i_test is not None
    assert fn_o_train is not None
    assert fn_o_test is not None
    assert target is not None

    logger = utilmlab.init_logger(os.path.dirname(fn_o_train))

    logger.info('loading {} target:{} sep:{}'.format(
        (fn_i_train, fn_i_test),
        target,
        sep))

    df_train = pd.read_csv(fn_i_train, sep=sep)
    df_test = pd.read_csv(fn_i_test, sep=sep)

    features = list(df_train.columns)
    for lbl in target:
        assert lbl in features
        features.remove(lbl)

    logger.info('features:{} {}'.format(len(features), features))
    logger.info('target:{}'.format(target))
    logger.info('epsilon:{} delta:{}'.format(epsilon, delta))

    time_exe_start = time.time()

    x_train_new, y_train_new, x_test_new, y_test_new = PATE_GAN(
        df_train[features].values,
        df_train[target].values,
        df_test[features].values,
        df_test[target].values,
        epsilon,
        delta,
        niter,
        num_teachers)

    cols = features
    cols.append(target[0])

    df_train_new = pd.DataFrame(
        np.hstack(
            [x_train_new,
             y_train_new.reshape(len(y_train_new), -1)]),
        columns=cols)

    df_test_new = pd.DataFrame(
        np.hstack(
            [x_test_new,
             y_test_new.reshape(len(y_test_new), -1)]),
        columns=cols)

    df_train_new.to_csv(
        fn_o_train,
        index=False,
        compression=utilmlab.get_df_compression(fn_o_train))

    df_test_new.to_csv(
        fn_o_test,
        index=False,
        compression=utilmlab.get_df_compression(fn_o_test))    
