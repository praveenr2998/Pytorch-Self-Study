from torch import optim


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,                  # Optimizer here could be anythong
                                    factor=0.5,patience=0,                   # factor = 0.5 is the factor with which the learning rate is reduced
                                    threshold = 0.001,                       # threshold is the difference between succesive n epochs or (patience epochs) here patience=0 so each succesive epoch
                                    verbose=True,                            # min_lr is the minimum value upto which the lr could be reduced to
                                    min_lr = 1e-5,                           
                                    threshold_mode = 'abs')



train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(30):
    #print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, \
                                 loss_fn)
        train_epoch_losses.append(batch_loss) 
    train_epoch_loss = np.array(train_epoch_losses).mean()

    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        validation_loss = val_loss(x, y, model)
        scheduler.step(validation_loss)
    val_epoch_accuracy = np.mean(val_is_correct)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)                    