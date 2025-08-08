
def on_train_batch_start(trainer):
    print('on train batch start')
    # breakpoint()


def on_train_batch_end(trainer):
    print('on train batch end')
    # trainer.train_loader.update_dataset()
    # breakpoint()



def on_train_epoch_end(trainer):
    print('on train epoch end')
    # breakpoint()