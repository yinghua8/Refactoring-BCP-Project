import torch
import os
import logging

class save_load_net:
    def save_net_opt(net, optimizer, path):
        state = {
            'net': net.state_dict(),
            'opt': optimizer.state_dict(),
        }
        torch.save(state, str(path))

    def load_net(net, optimizer, path):
        state = torch.load(str(path))
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['opt'])

    def load_net(net, path):
        state = torch.load(str(path))
        net.load_state_dict(state['net'])

    def save_best_model(self_snapshot_path, iter_num, best_dice, model, format_mdl):
        save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
        save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(format_mdl))
        # save_net_opt(model, optimizer, save_mode_path)
        # save_net_opt(model, optimizer, save_best_path)
        torch.save(model.state_dict(), save_mode_path)
        torch.save(model.state_dict(), save_best_path)
        logging.info("save best model to {}".format(save_mode_path))