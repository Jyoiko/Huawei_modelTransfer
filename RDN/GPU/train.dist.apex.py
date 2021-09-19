import argparse
import os
import copy
import sys
import time
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import datetime
from models import RDN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, denormalize
import torch.distributed as dist
import torch.multiprocessing as mp
import apex
from apex import amp
from apex.parallel import DistributedDataParallel as apexDDP



from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 

def flush_print(func):
    def new_print(*args,**kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
print=flush_print(print)
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=800)
    parser.add_argument('--num-workers', type=int, default=8) #modified by zhangdy
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus',type=int,default=1,help='num of gpus of per node')
    parser.add_argument('--nr',type=int,default=0,help='raking within the nodes')



    args = parser.parse_args()


    args.world_size = args.gpus*args.nodes
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='23456'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"#notice

    

    mp.spawn(train,nprocs=args.gpus, args=(args,))

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)   



def train(gpu,args):
    rank = args.nr*args.gpus + gpu
    torch.distributed.init_process_group(backend="nccl",
                                        init_method='env://',
                                        world_size=args.world_size,
                                        rank =rank ) 
    local_rank=torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda",local_rank)



    # cudnn.benchmark = True
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed+local_rank)

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = nn.L1Loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #####################################apex-start
    model,optimizer = amp.initialize(model,optimizer,opt_level='O2',loss_scale=128)
    model = apexDDP(model)
    #######################################apex-end

    #model = DDP(model,device_ids=[local_rank],output_device=local_rank)

    train_dataset = TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)

    train_sampler = DistributedSampler(train_dataset, rank=local_rank)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  sampler=train_sampler)
    
    
    eval_dataset = EvalDataset(args.eval_file)
    # eval_datasampler=DistributedSampler(eval_dataset,rank=local_rank)
    eval_dataloader = DataLoader(dataset=eval_dataset,batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    start_time = time.time()
    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()
        batchtime = AverageMeter(start_count_index=10)
        end = time.time()
        for i,data in enumerate(train_dataloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            ###########################apex-start
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
        ############################apex-end
            #loss.backward()
            optimizer.step()

            # t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            # t.update(len(inputs))
            batchsize = inputs.shape[0]
            batchtime.update(time.time() - end)
            end = time.time()
            if local_rank==0:
                print("Epoch {} step {},loss :{},img/s :{},time :{}".format(epoch,i,loss,batchsize/batchtime.val,batchtime.val))
            
        print('epoch:{} FPS: {:.3f}'.format(epoch,args.gpus*args.batch_size/batchtime.avg))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        if local_rank==0:
            model.eval()
            epoch_psnr = AverageMeter()

            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs)

                preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
                labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

                preds = preds[args.scale:-args.scale, args.scale:-args.scale]
                labels = labels[args.scale:-args.scale, args.scale:-args.scale]

                
                epoch_psnr.update(calc_psnr(preds,labels),len(inputs))
                
                print('eval psnr[{}/{}] : {:.2f}'.format(epoch+1,args.num_epochs,epoch_psnr.val))

            print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if local_rank == 0:
        print("Total time {}".format(total_time_str))
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))


if __name__ == '__main__':
    main()
