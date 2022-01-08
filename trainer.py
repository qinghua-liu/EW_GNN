import torch


def train(epoch, data_loader, model, optimizer, device, recorder):
    recorder.info('------starting {} epoch training------'.format(epoch))
    model.train()
    total = 0
    corrects = 0
    train_loss = 0.0

    for i, batch in enumerate(data_loader, 1):
        optimizer.zero_grad()
        batch.to(device=device)
        loss = model(batch)
        corrects += model.corrects
        loss.backward()
        optimizer.step()
        total += batch.y.size()[0]
        train_loss += loss.cpu().item()
    log = "train epoch[{}] end! train loss: {:.4f} train accuarcy: {:.2f}%".format(
        epoch, train_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return train_loss / i, corrects / total


def valid(epoch, data_loader, model, device, recorder):
    recorder.info('------starting {} epoch valid------'.format(epoch))
    model.eval()
    total = 0
    corrects = 0
    valid_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss = model(batch)
            corrects += model.corrects
            total += batch.y.size()[0]
            valid_loss += loss.cpu().item()

    log = "valid epoch[{}] end! valid loss: {:.4f} valid accuarcy: {:.2f}%".format(
        epoch, valid_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return valid_loss / i, corrects / total


def test(data_loader, model, device, recorder):
    recorder.info('------starting test------')
    model.eval()
    total = 0
    corrects = 0
    test_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss = model(batch)
            corrects += model.corrects
            total += batch.y.size()[0]
            test_loss += loss.cpu().item()
    log = "test end! test loss: {:.4f} test accuarcy: {:.2f}%".format(
        test_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return test_loss / i, corrects / total
