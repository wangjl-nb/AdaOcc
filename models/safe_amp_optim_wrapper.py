import torch
from mmengine.optim import AmpOptimWrapper
from mmengine.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class SafeAmpOptimWrapper(AmpOptimWrapper):
    """AMP optimizer wrapper with optional gradient sanitization.

    This wrapper keeps the default AmpOptimWrapper behavior unless
    ``sanitize_nonfinite_grads=True`` is set in config.
    """

    def __init__(self,
                 sanitize_nonfinite_grads=False,
                 sanitize_nan_value=0.0,
                 sanitize_posinf_value=0.0,
                 sanitize_neginf_value=0.0,
                 sanitize_grad_max_abs=None,
                 log_nonfinite_stats=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.sanitize_nonfinite_grads = sanitize_nonfinite_grads
        self.sanitize_nan_value = float(sanitize_nan_value)
        self.sanitize_posinf_value = float(sanitize_posinf_value)
        self.sanitize_neginf_value = float(sanitize_neginf_value)
        self.sanitize_grad_max_abs = (
            None if sanitize_grad_max_abs is None else float(sanitize_grad_max_abs))
        self.log_nonfinite_stats = bool(log_nonfinite_stats)

    def _iter_trainable_params(self):
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad and param.grad is not None:
                    yield param

    def _sanitize_gradients(self):
        nonfinite_param_count = 0
        nonfinite_value_count = 0

        for param in self._iter_trainable_params():
            grad = param.grad
            if grad.is_sparse:
                grad_values = grad._values()
                finite_mask = torch.isfinite(grad_values)
                if not finite_mask.all():
                    nonfinite_param_count += 1
                    nonfinite_value_count += int((~finite_mask).sum().item())
                    cleaned = torch.nan_to_num(
                        grad_values,
                        nan=self.sanitize_nan_value,
                        posinf=self.sanitize_posinf_value,
                        neginf=self.sanitize_neginf_value)
                    grad_values.copy_(cleaned)
                if self.sanitize_grad_max_abs is not None:
                    grad_values.clamp_(
                        min=-self.sanitize_grad_max_abs,
                        max=self.sanitize_grad_max_abs)
                continue

            finite_mask = torch.isfinite(grad)
            if not finite_mask.all():
                nonfinite_param_count += 1
                nonfinite_value_count += int((~finite_mask).sum().item())
                cleaned = torch.nan_to_num(
                    grad,
                    nan=self.sanitize_nan_value,
                    posinf=self.sanitize_posinf_value,
                    neginf=self.sanitize_neginf_value)
                grad.copy_(cleaned)
            if self.sanitize_grad_max_abs is not None:
                grad.clamp_(
                    min=-self.sanitize_grad_max_abs,
                    max=self.sanitize_grad_max_abs)

        return nonfinite_param_count, nonfinite_value_count

    def step(self, **kwargs):
        need_unscale = self.clip_grad_kwargs or self.sanitize_nonfinite_grads
        if need_unscale:
            self.loss_scaler.unscale_(self.optimizer)

        if self.sanitize_nonfinite_grads:
            nonfinite_param_count, nonfinite_value_count = self._sanitize_gradients()
            if self.log_nonfinite_stats:
                self.message_hub.update_scalar(
                    'train/nonfinite_grad_params', float(nonfinite_param_count))
                self.message_hub.update_scalar(
                    'train/nonfinite_grad_values', float(nonfinite_value_count))

        if self.clip_grad_kwargs:
            self._clip_grad()

        self.loss_scaler.step(self.optimizer, **kwargs)
        self.loss_scaler.update(self._scale_update_param)
