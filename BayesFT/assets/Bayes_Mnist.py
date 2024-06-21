from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import numpy as np
import torch
from utilis import add_noise_to_weights, evaluate, select_Data, fit_one_cycle_Bayes
from utilis_gpu import to_device, get_default_device

device = get_default_device()

train_dl, valid_dl = select_Data("Mnist")

curr_iter = 0

def run(model_name, n_iter):
    if model_name == 'LeNet':
        step_LeNet(n_iter)

# def fit_one_cycle_Bayes(epochs, max_lr, model, train_loader, val_loader,
#                         weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
#     torch.cuda.empty_cache()
#     history = []

#     # Set up cutom optimizer with weight decay
#     optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
#     # Set up one-cycle learning rate scheduler
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
#                                                 steps_per_epoch=len(train_loader))

#     for epoch in range(epochs):
#         # Training Phase 
#         model.train()
#         train_losses = []
#         lrs = []
#         for batch in train_loader:
#             loss = training_step(model, batch)
#             train_losses.append(loss)
#             loss.backward()

#             # Gradient clipping
#             if grad_clip:
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)

#             optimizer.step()
#             optimizer.zero_grad()

#             # Record & update learning rate
#             lrs.append(get_lr(optimizer))
#             sched.step()

#         # Validation phase
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         result['lrs'] = lrs
#         # if (epoch % 9) == 0:
#         epoch_end(epoch, result)
#         history.append(result)
#     return history

def bbf_LeNet(p1, p2, p3, p4, p5):
    global curr_iter
    
    from model import LeNet
    model = LeNet.LeNet_Bayes(p1, p2, p3, p4, p5)
    to_device(model, device)
    history = fit_one_cycle_Bayes(10, 0.01, model, train_dl, valid_dl,
                                  grad_clip=1e-1, weight_decay=1e-4, opt_func=torch.optim.Adam)
    weight_path = './results/Bayes/LeNet/LeNet-{:0.2f}-{:0.2f}-{:0.2f}-{:0.2f}-{:0.2f}.pth'.format(p1, p2, p3, p4, p5)
    torch.save(model.state_dict(), weight_path)
    # AVERAGE RUN 30 TIMES
    num = 30
    std = 1.#0.5 #1.0
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load(weight_path))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    curr_iter += 1
    print('Current iter: {}'.format(curr_iter))
    print('Std_dev of noise: {:0.2f}, accu: {:0.2f}'.format(std, accu))
    print('Dropout rates: {:0.2f}-{:0.2f}-{:0.2f}-{:0.2f}-{:0.2f}'.format(p1, p2, p3, p4, p5))
    print('_______________________________')
    # if accu > 0.4:
    #     torch.save(model.state_dict(), './results/Bayes/LeNet/LeNet-{:0.2f}@{:0.2f}.pth'.format(accu, std))
    
    with open('./results/Bayes/logs.txt', "a+") as f:
        # print(model, file=f)
        if curr_iter == 1:
            print('Iter\t Std\t Accu\t p (generalized DoRates)', file=f)
        print('{}\t {:0.2f}\t {:0.2f}\t {:0.2f}-{:0.2f}-{:0.2f}-{:0.2f}-{:0.2f}'.format(curr_iter, std, accu, p1, p2, p3, p4, p5), file=f)
        # print('The dropout rates: ', file=f)
    return accu


def step_LeNet(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0.2, 1.), 'p2': (0.2, 1.), 'p3': (0.2, 1.), 'p4': (0.2, 1.), 'p5': (0.2, 1)}

    optimizer = BayesianOptimization(
        f=bbf_LeNet,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': .2, 'p2': .2, 'p3': .2, 'p4': .2, 'p5': .5},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Bayes/log/LeNet/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )
