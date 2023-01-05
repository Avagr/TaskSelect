def basic_unpack(data_obj, device, model, criterion):
    pic, task, arg, res = data_obj
    pic, task, arg, res = pic.to(device), task.to(device), arg.to(device), res.to(device)
    prediction = model(pic, task, arg)
    loss = criterion(prediction, res)
    return loss


def two_task_unpack(data_obj, device, model, criterion):
    pic, task1, arg1, res1, task2, arg2, res2 = data_obj
    pic, task1, arg1, res1, task2, arg2, res2 = pic.to(device), task1.to(device), arg1.to(device), res1.to(
        device), task2.to(device), arg2.to(device), res2.to(device)
    prediction1, prediction2 = model(pic, task1, arg1, task2, arg2)
    loss1 = criterion(prediction1, res1)
    loss2 = criterion(prediction2, res1)
    return (loss1 + loss2) / 2
