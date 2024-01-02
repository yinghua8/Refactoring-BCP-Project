import torch
from torchvision import transforms
from networks.net_factory import net_factory
from dataloaders.dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from tensorboardX import SummaryWriter
import logging
from utils.BCP_utils import context_mask, mix_loss, update_ema_variables, get_cut_mask
from utils import test_3d_patch
import torch.nn.functional as F
from utils import ramps
from utils import losses
import save_load_net 

saveLoad = save_load_net.save_load_net()

device = torch.device("cpu")

class LA_train:
    def __init__(self, args, num_classes, patch_size):
        self.args = args
        self.train_data_path = args.root_path
        self.self_max_iterations = args.self_max_iteration
        self.base_lr = args.base_lr
        self.model = net_factory(net_type=args.model, in_chns=1, class_num = num_classes, mode="train")
        self.db_train = LAHeart(base_dir = self.train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))
        
        self.labeled_idxs = list(range(args.labelnum))
        self.unlabeled_idxs = list(range(args.labelnum, args.max_samples))
        self.pre_max_iterations = args.pre_max_iteration
        self.sub_bs = int(args.labeled_bs/2)
    
    def pre_train(self, snapshot_path):
        batch_sampler = TwoStreamBatchSampler(self.labeled_idxs, self.unlabeled_idxs, self.args.batch_size, self.args.batch_size - self.args.labeled_bs)

        trainloader = DataLoader(self.db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn = self._worker_init_fn)
        optimizer = optim.SGD(self.model.parameters(), lr = self.base_lr, momentum=0.9, weight_decay=0.0001)
        DICE = losses.mask_DiceLoss(n_classes=2)

        self.model.train()
        writer = SummaryWriter(snapshot_path+'/log')
        logging.info("{} iterations per epoch".format(len(trainloader)))
        iter_num = 0
        best_dice = 0
        max_epoch = self.pre_max_iterations // len(trainloader) + 1
        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch_num in iterator:
            for _, sampled_batch in enumerate(trainloader):
                
                img_a, img_b, lab_a, lab_b, _, _ = self._cal_patch(sampled_batch, True)

                with torch.no_grad():
                    img_mask, loss_mask = context_mask(img_a, self.args.mask_ratio)

                volume_batch, label_batch = self._MixInput(img_a, img_b, lab_a, lab_b, img_mask)

                outputs, _ = self.model(volume_batch)
                loss_ce = F.cross_entropy(outputs, label_batch)
                loss_dice = DICE(outputs, label_batch)
                loss = (loss_ce + loss_dice) / 2

                iter_num += 1
                writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
                writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
                writer.add_scalar('pre/loss_all', loss, iter_num)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))

                if iter_num % 200 == 0:
                    self.model.eval()
                    self._test_var_all_case(snapshot_path, iter_num, writer)
                    self.model.train()

                if iter_num >= self.pre_max_iterations:
                    break

            if iter_num >= self.pre_max_iterations:
                iterator.close()
                break
        writer.close()

    def self_train(self, pre_snapshot_path, self_snapshot_path):

        ema_model = net_factory(net_type=self.args.model, in_chns=1, class_num = self.num_classes, mode="train")
        for param in ema_model.parameters():
                param.detach_()   # ema_model set
        
        batch_sampler = TwoStreamBatchSampler(self.labeled_idxs, self.unlabeled_idxs, self.args.batch_size, self.args.batch_size - self.args.labeled_bs)
            
        trainloader = DataLoader(self.db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn = self._worker_init_fn)
        optimizer = optim.SGD(self.model.parameters(), lr = self.base_lr, momentum=0.9, weight_decay=0.0001)
        pretrained_model = os.path.join(pre_snapshot_path, f'{self.args.model}_best_model.pth')

        saveLoad.load_net(self.model, pretrained_model)
        saveLoad.load_net(ema_model, pretrained_model)
        self.model.train()
        ema_model.train()

        writer = SummaryWriter(self_snapshot_path+'/log')
        logging.info("{} iterations per epoch".format(len(trainloader)))
        iter_num = 0
        best_dice = 0
        max_epoch = self.self_max_iterations // len(trainloader) + 1
        lr_ = self.base_lr
        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch in iterator:
            for _, sampled_batch in enumerate(trainloader):
                
                img_a, img_b, lab_a, lab_b, unimg_a, unimg_b = self._cal_patch(sampled_batch, False)

                with torch.no_grad():
                    # extract model training process
                    unoutput_a, _ = ema_model(unimg_a)
                    unoutput_b, _ = ema_model(unimg_b)
                    plab_a = get_cut_mask(unoutput_a, nms=1)
                    plab_b = get_cut_mask(unoutput_b, nms=1)
                    img_mask, loss_mask = context_mask(img_a, self.args.mask_ratio)
                consistency_weight = self._get_current_consistency_weight(self.args, iter_num // 150)

                mixl_img, mixl_lab = self._MixInput(img_a, unimg_a, lab_a, plab_a, img_mask)
                mixu_img, mixu_lab = self._MixInput(unimg_b, img_b, plab_b, lab_b, img_mask)

                outputs_l, _ = self.model(mixl_img)
                outputs_u, _ = self.model(mixu_img)
                loss_l = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=self.args.u_weight)
                loss_u = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=self.args.u_weight, unlab=True)

                loss = loss_l + loss_u

                iter_num += 1

                writer.add_scalar('Self/consistency', consistency_weight, iter_num)
                writer.add_scalar('Self/loss_l', loss_l, iter_num)
                writer.add_scalar('Self/loss_u', loss_u, iter_num)
                writer.add_scalar('Self/loss_all', loss, iter_num)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f'%(iter_num, loss, loss_l, loss_u))

                update_ema_variables(self.model, ema_model, 0.99)

                # change lr
                if iter_num % 2500 == 0:
                    lr_ = self.args.base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                if iter_num % 200 == 0:
                    self.model.eval()
                    self._test_var_all_case(self_snapshot_path, iter_num, writer)
                    self.model.train()
        
                if iter_num % 200 == 1:
                    ins_width = 2
                    B,C,H,W,D = outputs_l.size()

                    snapshot_img = self._init_snapshot(H, W, D, ins_width)

                    self._create_snapshot_img(outputs_l, mixl_lab, mixl_img, H, W, snapshot_img, ins_width)
                    writer.add_images('Epoch_%d_Iter_%d_labeled'% (epoch, iter_num), snapshot_img)
                    self._create_snapshot_img(outputs_u, mixu_lab, mixu_img, H, W, snapshot_img, ins_width)
                    writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch, iter_num), snapshot_img)

                if iter_num >= self.self_max_iterations:
                    break

            if iter_num >= self.self_max_iterations:
                iterator.close()
                break
        writer.close()
    
    def _test_var_all_case(self, self_snapshot_path, iter_num, writer):
        dice_sample = test_3d_patch.var_all_case_LA(self.model, num_classes = self.num_classes, patch_size = self.patch_size, stride_xy=18, stride_z=4)
        if dice_sample > best_dice:
            best_dice = round(dice_sample, 4)
            saveLoad.save_best_model(self_snapshot_path, iter_num, best_dice, self.model)
        writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
        writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)

    def _cal_patch(self, sampled_batch, labeled_bs):
        volume_batch, label_batch = sampled_batch['image'][:self.args.labeled_bs], sampled_batch['label'][:self.args.labeled_bs]
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

        if labeled_bs == True:
            end = len(volume_batch)
        else:
            end = self.args.labeled_bs

        img_a, img_b = volume_batch[:self.sub_bs], volume_batch[self.sub_bs:end]
        lab_a, lab_b = label_batch[:self.sub_bs], label_batch[self.sub_bs:end]
        unimg_a, unimg_b = volume_batch[self.args.labeled_bs:self.args.labeled_bs + self.sub_bs], volume_batch[self.args.labeled_bs + self.sub_bs:]
        
        return img_a, img_b, lab_a, lab_b, unimg_a, unimg_b

    
    def _MixInput(self, img_a, img_b, lab_a, lab_b, img_mask):
        mix_img = img_a * img_mask + img_b * (1 - img_mask)
        mix_label = lab_a * img_mask + lab_b * (1 - img_mask)

        return mix_img, mix_label

    def _worker_init_fn(self, worker_id):
        random.seed(self.args.seed + worker_id)

    def _init_snapshot(self, H, W, D, ins_width):
        snapshot_img = torch.zeros(size = (D, 3, 3 * H + 3 * ins_width, W + ins_width), dtype = torch.float32)
        snapshot_img[:,:, H:H + ins_width,:] = 1
        snapshot_img[:,:, 2 * H + ins_width:2 * H + 2 * ins_width,:] = 1
        snapshot_img[:,:, 3 * H + 2 * ins_width:3 * H + 3 * ins_width,:] = 1
        snapshot_img[:,:, :, W:W + ins_width] = 1
        return snapshot_img

    def _create_snapshot_img(self, outputs, mix_lab, mix_img, H, W, snapshot_img, ins_width):

        outputs_soft = F.softmax(outputs, dim=1)
        seg_out = outputs_soft[0,1,...].permute(2,0,1) # y
        target =  mix_lab[0,...].permute(2,0,1)
        train_img = mix_img[0,0,...].permute(2,0,1)

        for i in range(3):
            snapshot_img[:, i,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
            snapshot_img[:, i, H + ins_width:2 * H + ins_width,:W] = target
            snapshot_img[:, i, 2 * H + 2 * ins_width:3 * H + 2 * ins_width,:W] = seg_out

    def _get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)
    

    