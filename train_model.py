import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils as utils
from torch.utils.data import DataLoader
import random
seed_num = 223
torch.manual_seed(seed_num)
random.seed(seed_num)


def train(train_iter, dev_iter, model, args):
    train_dataloader = DataLoader(dataset=train_iter, batch_size=32, num_workers=2, shuffle=True)
    valid_dataloader = DataLoader(dataset=dev_iter, batch_size=32, num_workers=2, shuffle=False)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay, momentum=args.momentum_value)
    best_accuracy = 0
    corrects_ = 0
    loss_ = 0
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to("cuda")
    for epoch in range(args.epochs):
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        model.train()
        for i, batch in enumerate(train_dataloader):
            feature, target = batch
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = model(feature)
            optimizer.zero_grad()
            loss = criterion(logit, target)
            _, predicted = torch.max(logit.data, 1)
            corrects_ += (predicted == target).sum().item()
            loss.backward()
            loss_ += loss
            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm_(model.parameters(), max_norm=args.init_clip_max_norm)
            optimizer.step()

        size = len(train_iter)
        accuracy = float(corrects_) / size
        loss_ = float(loss_) / size

        print("Training {} / {}, loss:{}, acc:{}%".format(epoch, args.epochs, loss_, 100 * accuracy))
        loss_ = 0
        corrects_ = 0

        print("#### Eval ####")
        total = 0
        correct = 0
        model.eval()
        for i, data in enumerate(valid_dataloader):
            data, label = data
            if args.cuda:
                data, label = data.cuda(), label.cuda()

            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        eval_acc = 100 * correct / total
        print('Eval {} / {}, acc:{}%'.format(epoch, args.epochs, eval_acc))

        if best_accuracy < eval_acc:
            best_accuracy = eval_acc
            torch.save(model.state_dict(), "best.pth")