import abc

import torch


class Unpack(abc.ABC):
    main_key: str

    def __call__(self, data_obj, model) -> (torch.Tensor, dict[str, torch.Tensor]):
        pass

    def evaluate(self, data_obj, model) -> dict[str, torch.Tensor]:
        pass


class BasicUnpack(Unpack):
    def __init__(self, device, criterion, accuracy_metric):
        self.device = device
        self.criterion = criterion
        self.accuracy_metric = accuracy_metric
        self.main_key = "loss"

    def __call__(self, data_obj, model):
        pic, task, arg, gt = data_obj
        pic, task, arg, gt = pic.to(self.device), task.to(self.device), arg.to(self.device), gt.to(self.device)
        prediction = model(pic, task, arg).main_logits.squeeze()
        loss = self.criterion(prediction, gt)
        return loss, {"loss": loss}

    def evaluate(self, data_obj, model):
        pic, task, arg, gt = data_obj
        pic, task, arg, gt = pic.to(self.device), task.to(self.device), arg.to(self.device), gt.to(self.device)
        prediction = model(pic, task, arg).main_logits.squeeze()
        loss = self.criterion(prediction, gt)
        acc = self.accuracy_metric(prediction, gt)
        return {"loss": loss, "acc": acc}


class TwoLossUnpack(Unpack):
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
        pic, task, arg, gt1, gt2 = data_obj
        pic, task, arg, gt1, gt2 = pic.to(self.device), task.to(self.device), arg.to(self.device), gt1.to(
            self.device), gt2.to(self.device)
        res = model(pic, task, arg)
        prediction1 = res.main_logits.squeeze()
        prediction2 = res.aux_logits.squeeze()
        loss1 = self.criterion1(prediction1, gt1)
        loss2 = self.criterion2(prediction2, gt2)
        return self.fraction * loss1 + (1 - self.fraction) * loss2, {
            f"{self.task_names[0]}_loss": loss1,
            f"{self.task_names[1]}_loss": loss2
        }

    def evaluate(self, data_obj, model):
        pic, task, arg, gt1, gt2 = data_obj
        pic, task, arg, gt1, gt2 = pic.to(self.device), task.to(self.device), arg.to(self.device), gt1.to(
            self.device), gt2.to(self.device)
        res = model(pic, task, arg)
        prediction1 = res.main_logits.squeeze()
        prediction2 = res.aux_logits.squeeze()
        loss1 = self.criterion1(prediction1, gt1)
        loss2 = self.criterion2(prediction2, gt2)
        accuracy1 = self.acc1(prediction1, gt1)
        accuracy2 = self.acc2(prediction2, gt2)
        return {
            f"{self.task_names[0]}_loss": loss1,
            f"{self.task_names[1]}_loss": loss2,
            f"{self.task_names[0]}_acc": accuracy1,
            f"{self.task_names[1]}_acc": accuracy2
        }


class KLUnpack(Unpack):
    def __init__(self, device, criterion, accuracy_metric, kl_weight=0.5, return_embeddings=False):
        self.device = device
        self.criterion = criterion
        self.kl_weight = kl_weight
        self.accuracy_metric = accuracy_metric
        self.main_key = "loss"
        self.return_embeddings = return_embeddings

    def __call__(self, data_obj, model):
        pic, task, arg, gt = data_obj
        pic, task, arg, gt = pic.to(self.device), task.to(self.device), arg.to(self.device), gt.to(self.device)
        res = model(pic, task, arg)
        prediction = res.main_logits.squeeze()
        dist_tuple = res.distribution_vectors
        main_loss = self.criterion(prediction, gt)
        kl_loss = 0
        for params in dist_tuple:
            if not params:
                continue
            kl_loss = kl_loss + torch.mean(
                -0.5 * torch.sum(1 + params['log_var'] - params['mean'].pow(2) - params['log_var'].exp(), dim=1), dim=0)
        kl_loss = kl_loss / len(dist_tuple)
        return main_loss + self.kl_weight * kl_loss, {"loss": main_loss, "kl_divergence": kl_loss}

    def evaluate(self, data_obj, model):
        pic, task, arg, gt = data_obj
        pic, task, arg, gt = pic.to(self.device), task.to(self.device), arg.to(self.device), gt.to(self.device)
        res = model(pic, task, arg)
        prediction = res.main_logits.squeeze()
        dist_tuple = res.distribution_vectors
        main_loss = self.criterion(prediction, gt)
        kl_loss = 0
        for params in dist_tuple:
            if not params:
                continue
            kl_loss = kl_loss + torch.mean(
                -0.5 * torch.sum(1 + params['log_var'] - params['mean'].pow(2) - params['log_var'].exp(), dim=1), dim=0)
        kl_loss = kl_loss / len(dist_tuple)
        accuracy = self.accuracy_metric(prediction, gt)
        metrics = {"loss": main_loss, "kl_divergence": kl_loss, "acc": accuracy}
        if self.return_embeddings:
            metrics["embeddings"] = dist_tuple
        return metrics


class VQACollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[(str, torch.Tensor, str, str)]):
        ids = []
        images = []
        questions = []
        answers = []
        for i, image, question, answer in batch:
            ids.append(i)
            images.append(image)
            questions.append(question)
            answers.append(answer)
        inputs = self.processor(text=questions, images=images, return_tensors='pt', padding=True)
        return inputs, questions, answers, ids


class VQAUnpack(Unpack):

    def __init__(self, device, criterion, generation_params: dict, processor):
        self.main_key = "loss"
        self.device = device
        self.criterion = criterion
        self.generation_params = generation_params
        self.processor = processor
        self.mistakes = []
        self.log_errors = False

    def set_logging(self, log_errors):
        if not self.log_errors and log_errors:
            self.mistakes = []
        self.log_errors = log_errors

    def __call__(self, data_obj, model):
        raise NotImplementedError  # TODO: implement

    def evaluate(self, data_obj, model):
        inputs, questions, answers, ids = data_obj
        inputs = inputs.to(device=self.device)
        outputs = model.generate(**inputs, **self.generation_params)
        generated_answers = self.processor.batch_decode(outputs, skip_special_tokens=True)
        hits = []
        for pred, question, gt, img_id in zip(generated_answers, questions, answers, ids):
            pred, gt = pred.strip().lower(), gt.strip().lower()
            if pred == gt:
                hits.append(1.)
            else:
                if self.log_errors:
                    self.mistakes.append((img_id, question, pred, gt))
                hits.append(0.)
        # TODO check LAVIS postprocessing
        return {"acc": torch.tensor(hits)}


def two_task_unpack(data_obj, device, model, criterion):
    pic, task1, arg1, res1, task2, arg2, res2 = data_obj
    pic, task1, arg1, res1, task2, arg2, res2 = pic.to(device), task1.to(device), arg1.to(device), res1.to(
        device), task2.to(device), arg2.to(device), res2.to(device)
    prediction1, prediction2 = model(pic, task1, arg1, task2, arg2)
    loss1 = criterion(prediction1, res1)
    loss2 = criterion(prediction2, res1)
    return (loss1 + loss2) / 2
