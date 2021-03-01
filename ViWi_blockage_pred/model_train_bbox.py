'''
Author: Gouranga Charan
Nov. 2021
'''
import torch
import torch.nn as nn
import torch.optim as optimizer
# from torch.utils.data import DataLoader
import numpy as np
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import os


def modelTrain(net,trn_loader,val_loader,options_dict):
    """

    :param net:
    :param data_samples:
    :param options_dict:
    :return:
    """
    # Optimizer:
    # ----------
    if options_dict['solver'] == 'Adam':
        opt = optimizer.Adam(net.parameters(),
                             lr=options_dict['lr'],
                             weight_decay=options_dict['wd'],
                             amsgrad=True)
    else:
        ValueError('Not recognized solver')

    scheduler = optimizer.lr_scheduler.MultiStepLR(opt,
                                                   milestones=options_dict['lr_sch'],
                                                   gamma=options_dict['lr_drop_factor'])

    # Define training loss:
    # ---------------------
    nSamples = [2100, 1900]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).cuda()   
    criterion = nn.CrossEntropyLoss(weight=normedWeights) 
    start_epoch = 0


    # Initialize training hyper-parameters:
    # -------------------------------------
    itr = 0
    embed = nn.Embedding(options_dict['cb_size'], options_dict['embed_dim'])
    running_train_loss = []
    running_trn_top_1 = []
    running_val_top_1 = []
    train_loss_ind = []
    val_acc_ind = []
    train_acc = []
    val_acc = []

    def save_checkpoint(state, filename='checkpoint/bbox_blockage_pred_img_beam_3_Nov_only_vehicles.pth.tar'):
        torch.save(state, filename)

    if not os.path.exists('checkpoint'):
      os.makedirs('checkpoint')

    print('------------------------------- Commence Training ---------------------------------')
    t_start = time.time()
    for epoch in range(start_epoch, start_epoch + options_dict['num_epochs']):

        net.train()
        h = net.initHidden(options_dict['batch_size'])
        h = h.cuda()

        # Training:
        # ---------
        for batch, (bbox, label) in enumerate(trn_loader):
            itr += 1   
            bbox = bbox[:, :options_dict['inp_seq']].float()           
            bbox = bbox.cuda()
            batch_size = label.shape[0]
            targ = label[:, options_dict['inp_seq']:options_dict['inp_seq']+options_dict['out_seq']].type(torch.LongTensor)
            targ = targ.view(-1)
            targ = targ.cuda()
            init_beams = label[:, :options_dict['inp_seq']].type(torch.LongTensor)
            inp_beams = embed(init_beams)
            inp_beams = inp_beams.cuda()
            batch_size = label.shape[0]          
            h = h.data[:,:batch_size,:].contiguous().cuda()
            opt.zero_grad()
            out, h = net.forward(bbox, inp_beams, h)
            #print("out", out)
            out = out.view(-1,out.shape[-1])
            #print("out after reshaping", out)
            train_loss = criterion(out, targ)  # (pred, target)
            train_loss.backward()
            opt.step()        
            out = out.view(batch_size,options_dict['out_seq'],options_dict['out_dim'])
            pred_beams = torch.argmax(out,dim=2)
            targ = targ.view(batch_size,options_dict['out_seq'])
            top_1_acc = torch.sum( torch.prod(pred_beams == targ, dim=1, dtype=torch.float) ) / targ.shape[0]
            #batch_score += torch.sum( torch.prod( pred_beams == targ, dim=1, dtype=torch.float ) )
            if np.mod(itr, options_dict['coll_cycle']) == 0:  # Data collection cycle
                running_train_loss.append(train_loss.item())
                running_trn_top_1.append(top_1_acc.cpu().numpy())
                train_loss_ind.append(itr)
                train_acc.append(top_1_acc.item())
                
            if np.mod(itr, options_dict['display_freq']) == 0:  # Display frequency
                print(
                    'Epoch No. {0}--Iteration No. {1}-- Mini-batch loss = {2:10.9f} and Top-1 accuracy = {3:5.4f}'.format(
                    epoch + 1,
                    itr,
                    train_loss.item(),
                    top_1_acc.item())
                    )

            # Validation:
            # -----------
            #print("Validation")
            test_list = []
            pred_list = []
            pred_seq_correct = []
            pred_seq_wrong = []
            out_lst = []
            if np.mod(itr, options_dict['val_freq']) == 0:  # or epoch + 1 == 
                net.eval()
                batch_score = 0
                _acc_score = 0
                _recall_score = 0
                _precision_score = 0
                _f1_score = 0
                
                with torch.no_grad():
                    for v_batch, (bbox, label) in enumerate(val_loader):
                        blockage_seq = []
                        init_beams = label[:, :options_dict['inp_seq']].type(torch.LongTensor)
                        lst_init_beams = (init_beams.float()).detach().cpu().numpy()
                        blockage_seq.append(lst_init_beams.tolist())
                        inp_beams = embed(init_beams)
                        inp_beams = inp_beams.cuda()
                        bbox = bbox[:, :options_dict['inp_seq']].float()            
                        bbox = bbox.cuda()
                        batch_size = label.shape[0]
                        targ = label[:,options_dict['inp_seq']:options_dict['inp_seq']+options_dict['out_seq']]\
                               .type(torch.LongTensor)
                        targ = targ.view(batch_size,options_dict['out_seq'])
                        targ = targ.cuda()
                        h_val = net.initHidden(batch_size).cuda()

                        out, h_val = net.forward(bbox, inp_beams, h_val)

                        pred_beams = torch.argmax(out, dim=2)
                        test_list.append(targ.detach().cpu().numpy())
                        pred_list.append(pred_beams.detach().cpu().numpy())
                        _targ = targ.float()
                        _targ = _targ.detach().cpu().numpy()
                        _pred_beams = pred_beams.float()
                        _pred_beams = _pred_beams.detach().cpu().numpy()
                        batch_score += torch.sum( torch.prod( pred_beams == targ, dim=1, dtype=torch.float ) )
                        tmp_list = [blockage_seq, targ.detach().cpu().numpy(), pred_beams.detach().cpu().numpy()[0][0] ]
                        out_lst.append(tmp_list)
                    import csv
                    with open("./analysis/cam2_img_beam_test_dataset_out_bbox_sequence_only_vehicles_0_0_%s.csv"%epoch, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(out_lst)
                    
                       
                    running_val_top_1.append(batch_score.cpu().numpy() / options_dict['test_size'])
                    val_acc_ind.append(itr)
                    print('Validation-- Top-1 accuracy = {0:5.4f}'.format(
                        running_val_top_1[-1])
                    )
                    val_acc.append(running_val_top_1)                  
                    #import csv
                    #with open("./analysis/cam2_img_beam_test_dataset.csv", "w", newline="") as f:
                    #    writer = csv.writer(f)
                    #    writer.writerows(out_lst)

                    save_checkpoint({
                         'epoch': epoch + 1,
                         'gru_state_dict': net.state_dict(),
                         'optimizer': opt.state_dict(),
                         'embedding': embed,
                    })
            net.train()

        current_lr = scheduler.get_lr()[-1]
        scheduler.step()
        new_lr = scheduler.get_lr()[-1]
        if new_lr != current_lr:
            print('Learning rate reduced to {}'.format(new_lr))
            
    with open("./analysis/cam2_block_pred_train_acc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(train_acc))
    with open("./analysis/cam2_block_pred_test_acc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(val_acc))
    
            
    save_checkpoint({
        'epoch': epoch + 1,
        'gru_state_dict': net.state_dict(),
        'optimizer' : opt.state_dict(),
        'embedding': embed,
    })  
    
    
    t_end = time.time()
    train_time = (t_end - t_start)/60
    print('Training lasted {0:6.3f} minutes'.format(train_time))
    print('------------------------ Training Done ------------------------')
    train_info = {'train_loss': running_train_loss,
                  'train_top_1': running_trn_top_1,
                  'val_top_1':running_val_top_1,
                  'train_itr':train_loss_ind,
                  'val_itr':val_acc_ind,
                  'train_time':train_time}


    return [net, options_dict,train_info]
