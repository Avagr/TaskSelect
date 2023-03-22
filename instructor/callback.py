from stable_baselines3.common.callbacks import BaseCallback


class IoUCallback(BaseCallback):

    def __init__(self, verbose=0, arr=None):
        super().__init__(verbose)
        self.arr = arr
        self._log_freq = 5

    # def _on_training_start(self):
    #     print(self.locals)

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            ious = 0
            count = 0
            for info in self.locals['infos']:
                if 'iou' in info:
                    ious += info['iou']
                    count += 1
            if count > 0:
                self.logger.record("IoU", ious / count)
                if self.arr is not None:
                    self.arr.append(ious / count)
        return True
