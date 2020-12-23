
import torch
import numpy as np
import bit_pytorch.models as models
from bit_pytorch.dataloader import GetLoader

def evaluate(model, train_loader):

    model.eval()
    num_ones = 0
    num_zeros = 0
    pred_vec = []
    with torch.no_grad():
        for b, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True, dtype=torch.float)
            y = y.to(device, non_blocking=True, dtype=torch.long)

            # Compute output, measure accuracy
            logits = model(x)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            pred_vec.extend(preds)
            num_ones += np.sum(preds == 1)
            num_zeros += np.sum(preds == 0)
            print("GT:", y)
            print("Pred:", pred_vec)

    return pred_vec

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = models.KNOWN_MODELS['FCCustomize'](head_size=2, grid_num=10)
    checkpoint = torch.load('./output/website/bit.pth.tar', map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    train_set = GetLoader(img_folder='./data/first_round_3k3k/all_imgs',
                          annot_path='./data/first_round_3k3k/all_coords.txt')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, drop_last=False, shuffle=False)

    pred_vec = evaluate(model, train_loader)
    print(pred_vec)
