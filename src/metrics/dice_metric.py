import torch

class DiceMetric(object):
    def __init__(self, 
                force_binary = False, 
                per_image = False, 
                organ_order = [], 
                drop_background = False):
        self.force_binary = force_binary
        self.per_image = per_image
        self.organ_order = organ_order
        self.drop_background = drop_background

    def preprocess(self, output):
        y, yhat = output
        if isinstance(yhat, tuple):
            yhat = yhat[0]
        y = y.detach().cpu().float()
        yhat = yhat.detach().cpu().float()
        if len(y.shape) != len(yhat.shape) and yhat.shape[1] != 1:
            num_classes = yhat.shape[1]
            yhat = yhat.argmax(1)
            inte_and_card = []
            for c in range(num_classes):
                inte = ((y == c).long() * (yhat == c).long()).sum((1,2))
                card = ((y == c).long() + (yhat == c).long()).sum((1,2))
                inte_and_card.append(torch.stack([inte, card], 1))
            return torch.stack(inte_and_card, 2)
        else:
            if yhat.shape[1] == 1: yhat = yhat.squeeze(1)
            yhat = yhat.sigmoid().round()
            inte = ((y == 1).long() * (yhat == 1).long()).sum((1,2))
            card = ((y == 1).long() + (yhat == 1).long()).sum((1,2))
            return torch.stack([inte, card], 1)

    def __call__(self, outputs):
        dice = torch.cat(outputs).numpy()
        
        if not self.per_image:
            dice = dice.sum(0)
            dice = 2 * dice[0] / (dice[1] + 1e-6)
        else:
            dice = 2 * dice[:,0] / (dice[:,1] + 1e-6)
            dice = dice.mean(0)

        if not self.organ_order:
            return {"val_dice": dice}
        else:
            if not self.drop_background:
                met = {"val_dice": dice.mean()}
                met["val_dice_bg"] = dice[0]
            else:
                met = {"val_dice": dice[1:].mean()}
            for i, organ in enumerate(self.organ_order):
                met["val_dice_" + organ] = dice[i + 1]
            return met

