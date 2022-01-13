import pandas as pd
import numpy as np
import argparse
import os
import random
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from _utils import DPDataset, EarlyStopping, dp_predictor_loss
from Predictor import DPPredictor
from Metrics import print_metrics_binary
from OurModel import StageEncoder_C, StageEncoder_A, StageEncoder_CA


RANDOM_SEED = 2022
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)


def parse_arguments(parser):
    parser.add_argument('mimic_dataset', type=int, default=3, help='MIMIC-3 or 4 dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Training epochs')
    parser.add_argument('--patience', type=int, default=20, help='EarlyStopping patience')
    parser.add_argument('--input_dim', type=int, default=76, help='Dimension of data')
    parser.add_argument('--core_hidden_dim', type=int, default=128, help='Dimension of Core_model_hidden_units')
    parser.add_argument('--core_output_dim', type=int, default=128, help='Dimension of Core_model_output_units')
    parser.add_argument('--label_dim', type=int, default=1, help='Dimension of predictor_output_units')
    parser.add_argument('--reseample_times', type=int, default=100, help='Resample the test set <reseample_times> times')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dropconnect_rate', type=float, default=0.5, help='Dropout rate in RNN')
    parser.add_argument('--dropres_rate', type=float, default=0.5, help='Dropout rate in residue connection')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    baseline_dict = {
        'OurModel-C': StageEncoder_C,
        'OurModel-A': StageEncoder_A,
        'OurModel-CA': StageEncoder_CA,
    }
    baseline_names = ["OurModel-C", 'OurModel-A', 'OurModel-CA']

    all_baseline_results = []

    train_data_file = ''
    val_data_file = ''
    test_data_file = ''
    saved_model_file = ''

    logging_file = f'~/out_log/dp_mimic{args.mimic_dataset:d}_ourmodel_results_log.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='[%Y/%m/%d %H:%M:%S]',
                        filename=logging_file,
                        filemode='w+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '[%Y/%m/%d %H:%M:%S]')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    for baseline_name in baseline_names:
        print(baseline_names)
        logging.info("Current: " + baseline_name)
        logging.info('Setting:mimic_dataset={};batch_size={};epochs={},lr={},patience={},reseample_times={}'.format(args.mimic_dataset, args.batch_size,
                                                                                    args.epochs, args.lr, args.patience, args.reseample_times))

        if args.mimic_dataset == 3:
            train_data_file = '~/dp_mimic3_train_data.pkl'
            val_data_file = '~/dp_mimic3_val_data.pkl'
            test_data_file = '~/dp_mimic3_test_data.pkl'
            saved_model_file = f'~/{baseline_name:s}-MIMIC-III'

            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        else:
            train_data_file = '~/dp_mimic4_train_data.pkl'
            val_data_file = '~/dp_mimic4_val_data.pkl'
            test_data_file = '~/dp_mimic4_test_data.pkl'
            saved_model_file = f'~/{baseline_name:s}-MIMIC-IV'
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        ''' Prepare training data'''

        logging.info('Preparing training data ... ')

        train_data_loader = DataLoader(DPDataset(train_data_file), batch_size=args.batch_size, shuffle=True)
        val_data_loader = DataLoader(DPDataset(val_data_file), batch_size=args.batch_size, shuffle=True)
        logging.info('Num Train: '+str(DPDataset(train_data_file).__len__())+'\tNum Val: '+str(DPDataset(val_data_file).__len__()))

        '''Model structure'''

        logging.info('Constructing model ... ')

        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        logging.info("Available GPU: {}".format(device))
        core_model = baseline_dict[baseline_name](input_dim=args.input_dim, hidden_dim=args.core_hidden_dim, output_dim=args.core_output_dim, device=device, n_steps=168, n_stages=7, n_layers=6).to(device)
        model = DPPredictor(core_model=core_model, label_dim=args.label_dim).to(device)
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')

        # optimizer = torch.optim.Adamax(params=model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        '''Train phase'''
        logging.info('Start training ... ')

        train_loss = []
        val_loss = []
        batch_loss = []
        max_auprc = 0

        early_stopping = EarlyStopping(args.patience, verbose=True, path=saved_model_file)

        for n_epoch in range(args.epochs):
            cur_batch_loss = []
            model.train()
            for i_batch, batch in enumerate(train_data_loader):
                train_x = torch.tensor(batch[0], dtype=torch.float32).to(device)
                train_mask = torch.tensor(batch[1], dtype=torch.float32).to(device)
                train_cur_mask = torch.tensor(batch[2], dtype=torch.float32).to(device)
                train_interval = torch.tensor(batch[3], dtype=torch.float32).to(device)
                train_y = torch.tensor(batch[4], dtype=torch.float32).to(device)

                inputs = {'X': train_x, 'M': train_mask, 'cur_M': train_cur_mask, 'T': train_interval}
                optimizer.zero_grad()
                train_y_hat = model(inputs)
                loss = dp_predictor_loss(y_true=train_y, y_pred=train_y_hat, mask_=train_mask)
                # loss = torch.nn.BCELoss()(input=train_y_hat, target=train_y)
                cur_batch_loss.append(loss.cpu().detach().numpy())
                loss.backward()
                optimizer.step()
            train_loss.append(np.mean(np.array(cur_batch_loss)))
            # logging.info('N_epoch %d, Train Loss = %.8f' % (n_epoch, np.mean(np.array(cur_batch_loss))))

            # logging.info("\n==>Predicting on validation")
            with torch.no_grad():
                model.eval()
                cur_val_loss = []
                val_true = []
                val_pred = []
                for i_batch, batch in enumerate(val_data_loader):
                    val_x = torch.tensor(batch[0], dtype=torch.float32).to(device)
                    val_mask = torch.tensor(batch[1], dtype=torch.float32).to(device)
                    val_cur_mask = torch.tensor(batch[2], dtype=torch.float32).to(device)
                    val_interval = torch.tensor(batch[3], dtype=torch.float32).to(device)
                    val_y = torch.tensor(batch[4], dtype=torch.float32).to(device)

                    inputs = {'X': val_x, 'M': val_mask, 'cur_M': val_cur_mask, 'T': val_interval}

                    val_y_hat = model(inputs)
                    valid_loss = dp_predictor_loss(y_true=val_y, y_pred=val_y_hat, mask_=val_mask)
                    # valid_loss = torch.nn.BCELoss()(input=val_y_hat, target=val_y)
                    cur_val_loss.append(valid_loss.cpu().detach().numpy())

                    for m, t, p in zip(val_mask.cpu().numpy().flatten(), val_y.cpu().numpy().flatten(),
                                       val_y_hat.cpu().detach().numpy().flatten()):
                        if np.equal(int(m), 1):
                            val_true.append(t)
                            val_pred.append(p)

                val_ret_auprc = (print_metrics_binary(val_true, val_pred))['AUPRC']
                early_stopping(-1.0*val_ret_auprc, n_epoch, model)
                val_loss.append(np.mean(np.array(cur_val_loss)))

                if early_stopping.early_stop:
                    logging.info(f"Early stopping at epoch:{n_epoch + 1:d}")
                    break

        '''Evaluate phase'''
        logging.info('Testing model ... ')
        model.load_state_dict(torch.load(saved_model_file))
        model.eval()

        dp_test = DPDataset(test_data_file)
        all_test = dp_test.__len__()
        test_data_loader = DataLoader(dp_test, batch_size=args.batch_size, shuffle=True)
        end_n_batch = int((all_test * 0.8) / args.batch_size)
        # print(all_test, 0.8*all_test, end_n_batch)

        with torch.no_grad():
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)

            all_test_time_results = []
            for re_time in tqdm(range(args.reseample_times), desc='Testing'):
                cur_test_loss = []
                test_true = []
                test_pred = []

                for i_batch, batch in enumerate(test_data_loader):
                    test_x = torch.tensor(batch[0], dtype=torch.float32).to(device)
                    test_mask = torch.tensor(batch[1], dtype=torch.float32).to(device)
                    test_cur_mask = torch.tensor(batch[2], dtype=torch.float32).to(device)
                    test_interval = torch.tensor(batch[3], dtype=torch.float32).to(device)
                    test_y = torch.tensor(batch[4], dtype=torch.float32).to(device)

                    inputs = {'X': test_x, 'M': test_mask, 'cur_M': test_cur_mask, 'T': test_interval}

                    test_y_hat = model(inputs)
                    test_loss = dp_predictor_loss(y_true=test_y, y_pred=test_y_hat, mask_=test_mask)
                    # test_loss = torch.nn.BCELoss()(input=test_y_hat, target=test_y)
                    cur_test_loss.append(test_loss.cpu().detach().numpy())

                    for m, t, p in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(),
                                       test_y_hat.cpu().detach().numpy().flatten()):
                        if np.equal(int(m), 1):
                            test_true.append(t)
                            test_pred.append(p)
                    if i_batch + 1 == end_n_batch:
                        break
                # logging.info('Test loss = %.4f' % (np.mean(np.array(cur_test_loss))))
                test_ret = print_metrics_binary(test_true, test_pred)
                all_test_time_results.append([test_ret['F1'],
                                              test_ret['AUROC'],
                                              test_ret['AUPRC'],
                                              test_ret['Min(Se, P+)']])
            avg_all_test_time_results = np.mean(np.array(all_test_time_results), axis=0)
            std_all_test_time_results = np.std(np.array(all_test_time_results), axis=0)
            # print(list(all_test_time_results))
            logging.info(baseline_name + ' Avg'
                         f"\tAUROC = {avg_all_test_time_results[1]:.6f}" +
                         f"\tAUPRC = {avg_all_test_time_results[2]:.6f}" +
                         f"\tMin(Se, P+) = {avg_all_test_time_results[3]:.6f}"
                         )
            logging.info(baseline_name + ' Std'
                         f"\tAUROC = {std_all_test_time_results[1]:.6f}" +
                         f"\tAUPRC = {std_all_test_time_results[2]:.6f}" +
                         f"\tMin(Se, P+) = {std_all_test_time_results[3]:.6f}"
                         )
            logging.info('')
            all_baseline_results.append([
                baseline_name,
                str(round(avg_all_test_time_results[0], 3)) + '±' + str(round(std_all_test_time_results[0], 4)),
                str(round(avg_all_test_time_results[1], 3)) + '±' + str(round(std_all_test_time_results[1], 4)),
                str(round(avg_all_test_time_results[2], 3)) + '±' + str(round(std_all_test_time_results[2], 4)),
                str(round(avg_all_test_time_results[3], 3)) + '±' + str(round(std_all_test_time_results[3], 4))
            ])
    pd.DataFrame(
        data=np.array(all_baseline_results, dtype=np.str),
        columns=['Model', 'F1', 'AUROC', 'AUPRC', 'Min(Se, P+)'],
        index=None).to_csv(f'~/out_log/dp_mimic{args.mimic_dataset:d}_ourmodel_results.csv', index=None)
