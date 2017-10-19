from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from tqdm import tqdm
import visual
import utils
import splits


def train(model, train_dataset, test_dataset=None, model_dir='models',
          lr=1e-04, lr_decay=.1, lr_decay_epochs=None, weight_decay=1e-04,
          gamma1=1., gamma2=1., gamma3=10.,
          batch_size=32, test_size=256, epochs=5,
          eval_log_interval=30,
          loss_log_interval=30,
          weight_log_interval=500,
          checkpoint_interval=500,
          resume_best=False,
          resume_latest=False,
          cuda=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay
    )
    scheduler = MultiStepLR(optimizer, lr_decay_epochs, gamma=lr_decay)

    # prepare the model and statistics.
    model.train()
    epoch_start = 1
    best_precision = 0

    # load checkpoint if needed.
    if resume_latest or resume_best:
        epoch_start, best_precision = utils.load_checkpoint(
            model, model_dir, best=resume_best
        )

    for epoch in range(epoch_start, epochs + 1):
        # adjust learning rate if needed.
        scheduler.step(epoch-1)

        # prepare a data stream for the epoch.
        data_loader = utils.get_data_loader(
            train_dataset, batch_size, cuda=cuda
        )
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (data, labels) in data_stream:
            # where are we?
            data_size = len(data)
            dataset_size = len(data_loader.dataset)
            dataset_batches = len(data_loader)
            iteration = (
                (epoch-1)*(len(data_loader.dataset) // batch_size) +
                batch_index + 1
            )

            # clear the gradients.
            optimizer.zero_grad()

            # run the network.
            x = Variable(data).cuda() if cuda else Variable(data)
            labels = Variable(labels).cuda() if cuda else Variable(labels)
            scores = model(x)
            _, predicted = scores.max(1)
            precision = (labels == predicted).sum().data[0] / data_size

            # update the network.
            cross_entropy_loss = criterion(scores, labels)
            overlap_loss, uniform_loss, split_loss = model.reg_loss()
            overlap_loss *= gamma1
            uniform_loss *= gamma3
            split_loss *= gamma2
            reg_loss = overlap_loss + uniform_loss + split_loss

            total_loss = cross_entropy_loss + reg_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # update & display statistics.
            data_stream.set_description((
                'epoch: {epoch}/{epochs} | '
                'it: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'prec: {prec:.3} | '
                'loss => '
                'ce: {ce_loss:.4} / '
                'reg: {reg_loss:.4} / '
                'total: {total_loss:.4}'
            ).format(
                epoch=epoch,
                epochs=epochs,
                iteration=iteration,
                trained=(batch_index+1)*batch_size,
                total=dataset_size,
                progress=(100.*(batch_index+1)/dataset_batches),
                prec=precision,
                ce_loss=(cross_entropy_loss.data[0] / data_size),
                reg_loss=(reg_loss.data[0] / data_size),
                total_loss=(total_loss.data[0] / data_size),
            ))

            # Send test precision to the visdom server.
            if iteration % eval_log_interval == 0:
                visual.visualize_scalar(utils.validate(
                    model, test_dataset,
                    test_size=test_size, cuda=cuda, verbose=False
                ), 'precision', iteration, env=model.name)

            # Send losses to the visdom server.
            if iteration % loss_log_interval == 0:
                reg_losses_and_names = ([
                    overlap_loss.data / data_size,
                    uniform_loss.data / data_size,
                    split_loss.data / data_size,
                    reg_loss.data / data_size,
                ], ['overlap', 'uniform', 'split', 'total'])

                visual.visualize_scalar(
                    overlap_loss.data / data_size,
                    'overlap loss', iteration, env=model.name
                )
                visual.visualize_scalar(
                    uniform_loss.data / data_size,
                    'uniform loss', iteration, env=model.name
                )
                visual.visualize_scalar(
                    split_loss.data / data_size,
                    'split loss', iteration, env=model.name
                )
                visual.visualize_scalars(
                    *reg_losses_and_names,
                    'regulaization losses', iteration, env=model.name
                )

                model_losses_and_names = ([
                    cross_entropy_loss.data / data_size,
                    reg_loss.data / data_size,
                    total_loss.data / data_size,
                ], ['cross entropy', 'regularization', 'total'])

                visual.visualize_scalar(
                    cross_entropy_loss.data / data_size,
                    'cross entropy loss', iteration, env=model.name
                )

                visual.visualize_scalar(
                    reg_loss.data / data_size,
                    'regularization loss', iteration, env=model.name
                )

                visual.visualize_scalars(
                    *model_losses_and_names,
                    'model losses', iteration, env=model.name
                )

            if iteration % weight_log_interval == 0:
                # Send visualized weights to the visdom server.
                weights = [
                    (w.data, p, q) for
                    i, g in enumerate(model.residual_block_groups) for
                    b in g.residual_blocks for
                    w, p, q in (
                        (b.w1, b.p(), b.r()),
                        (b.w2, b.r(), b.q()),
                        (b.w3, b.p(), b.q()),
                    ) if i+1 > (len(model.residual_block_groups) -
                                (len(model.split_sizes)-1)) and w is not None
                ] + [(
                    model.fc.linear.weight.data,
                    model.fc.p(),
                    model.fc.q()
                )]

                names = [
                    'g{i}-b{j}-w{k}'.format(i=i+1, j=j+1, k=k+1) for
                    i, g in enumerate(model.residual_block_groups) for
                    j, b in enumerate(g.residual_blocks) for
                    k, w in enumerate((b.w1, b.w2, b.w3)) if
                    i+1 > (len(model.residual_block_groups) -
                           (len(model.split_sizes)-1)) and w is not None
                ] + ['fc-w']

                for (w, p, q), name in zip(weights, names):
                    visual.visualize_kernel(
                        splits.block_diagonalize_kernel(w, p, q), name,
                        label='epoch{}-{}'.format(epoch, batch_index+1),
                        update_window_without_label=True,
                        env=model.name,
                    )

                # Send visualized split indicators to the visdom server.
                indicators = [
                    q.data for
                    i, g in enumerate(model.residual_block_groups) for
                    j, b in enumerate(g.residual_blocks) for
                    k, q in enumerate((b.p(), b.r())) if q is not None
                ] + [model.fc.p().data, model.fc.q().data]

                names = [
                    'g{i}-b{j}-{indicator}'
                    .format(i=i+1, j=j+1, indicator=ind) for
                    i, g in enumerate(model.residual_block_groups) for
                    j, b in enumerate(g.residual_blocks) for
                    ind, q in zip(('p', 'r'), (b.p(), b.r())) if
                    q is not None
                ] + ['fc-p', 'fc-q']

                for q, name in zip(indicators, names):
                    # Stretch the split indicators before visualization.
                    q_diagonalized = splits.block_diagonalize_indacator(q)
                    q_diagonalized_expanded = q_diagonalized\
                        .view(*q.size(), 1)\
                        .repeat(1, 20, 1)\
                        .view(-1, q.size()[1])

                    visual.visualize_kernel(
                        q_diagonalized_expanded, name,
                        label='epoch{}-{}'.format(epoch, batch_index+1),
                        update_window_without_label=True,
                        env=model.name, w=100, h=100
                    )

            if iteration % checkpoint_interval == 0:
                # notify that we've reached to a new checkpoint.
                print()
                print()
                print('#############')
                print('# checkpoint!')
                print('#############')
                print()

                # test the model.
                model_precision = utils.validate(
                    model, test_dataset or train_dataset,
                    test_size=test_size, cuda=cuda, verbose=True
                )

                # update best precision if needed.
                is_best = model_precision > best_precision
                best_precision = max(model_precision, best_precision)

                # save the checkpoint.
                utils.save_checkpoint(
                    model, model_dir, epoch,
                    model_precision, best=is_best
                )
                print()
