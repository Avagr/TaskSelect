class BasicUnpack:
    def __init__(self, device, criterion):
        self.device = device
        self.criterion = criterion
        self.main_key = "loss"

    def __call__(self, data_obj, model):
        pic, task, arg, res = data_obj
        pic, task, arg, res = pic.to(self.device), task.to(self.device), arg.to(self.device), res.to(self.device)
        prediction = model(pic, task, arg)
        loss = self.criterion(prediction, res)
        return loss, {"loss": loss}


class TwoLossUnpack:
    def __init__(self, device, criterion1, criterion2, fraction: float, acc1, acc2, task_names=None, main_key=None):
        if task_names is None:
            task_names = ["first", "second"]

        self.task_names = task_names
        self.device = device
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.acc1 = acc1
        self.acc2 = acc2
        if not 0 <= fraction <= 1:
            raise ValueError(f"Fraction should be between 0 and 1, got {fraction}")
        self.fraction = fraction
        if main_key is None:
            main_key = f"{self.task_names[0]}_loss"
        self.main_key = main_key

    def __call__(self, data_obj, model):
        pic, task, arg, res1, res2 = data_obj
        pic, task, arg, res1, res2 = pic.to(self.device), task.to(self.device), arg.to(self.device), res1.to(
            self.device), res2.to(self.device)
        prediction1, prediction2 = model(pic, task, arg)
        loss1 = self.criterion1(prediction1, res1)
        loss2 = self.criterion2(prediction2, res2)
        return self.fraction * loss1 + (1 - self.fraction) * loss2, {
            f"{self.task_names[0]}_loss": loss1,
            f"{self.task_names[1]}_loss": loss2
        }

    def evaluate(self, data_obj, model):
        pic, task, arg, res1, res2 = data_obj
        pic, task, arg, res1, res2 = pic.to(self.device), task.to(self.device), arg.to(self.device), res1.to(
            self.device), res2.to(self.device)
        prediction1, prediction2 = model(pic, task, arg)
        loss1 = self.criterion1(prediction1, res1)
        loss2 = self.criterion2(prediction2, res2)
        accuracy1 = self.acc1(prediction1, res1)
        accuracy2 = self.acc2(prediction2, res2)

        return {
            f"{self.task_names[0]}_loss": loss1,
            f"{self.task_names[1]}_loss": loss2,
            f"{self.task_names[0]}_acc": accuracy1,
            f"{self.task_names[1]}_acc": accuracy2
        }


def two_task_unpack(data_obj, device, model, criterion):
    pic, task1, arg1, res1, task2, arg2, res2 = data_obj
    pic, task1, arg1, res1, task2, arg2, res2 = pic.to(device), task1.to(device), arg1.to(device), res1.to(
        device), task2.to(device), arg2.to(device), res2.to(device)
    prediction1, prediction2 = model(pic, task1, arg1, task2, arg2)
    loss1 = criterion(prediction1, res1)
    loss2 = criterion(prediction2, res1)
    return (loss1 + loss2) / 2
