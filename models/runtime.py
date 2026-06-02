import time
import datetime
from collections import OrderedDict

from mmengine.hooks import IterTimerHook
from mmengine.registry import HOOKS, LOG_PROCESSORS, RUNNERS
from mmengine.runner import Runner
from mmengine.runner.log_processor import LogProcessor
from mmengine.utils.dl_utils import collect_env


@HOOKS.register_module()
class AdaOccIterTimerHook(IterTimerHook):
    """Iter timer hook with wall-clock elapsed time logging."""

    def before_run(self, runner) -> None:
        self.run_start_time = time.time()
        runner.message_hub.update_info('elapsed', 0.0)

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch=None,
                    outputs=None,
                    mode: str = 'train') -> None:
        super()._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode=mode)
        runner.message_hub.update_info(
            'elapsed', time.time() - self.run_start_time)


@LOG_PROCESSORS.register_module()
class AdaOccLogProcessor(LogProcessor):
    """Log processor that appends elapsed wall time to iter logs."""

    def get_log_after_iter(self, runner, batch_idx: int, mode: str):
        tag, log_str = super().get_log_after_iter(runner, batch_idx, mode)
        elapsed = runner.message_hub.get_info('elapsed', None)
        if elapsed is None:
            return tag, log_str

        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        if 'memory:' in log_str:
            log_str = log_str.replace(
                'memory:',
                f'elapsed: {elapsed_str}  memory:',
                1)
        else:
            log_str = f'{log_str}  elapsed: {elapsed_str}'
        return tag, log_str


@RUNNERS.register_module()
class AdaOccRunner(Runner):
    """Runner variant that keeps env logging but skips pretty config dump."""

    def _log_env(self, env_cfg: dict) -> None:
        env = collect_env()
        runtime_env = OrderedDict()
        runtime_env.update(env_cfg)
        runtime_env.update(self._randomness_cfg)
        runtime_env['seed'] = self._seed
        runtime_env['Distributed launcher'] = self._launcher
        runtime_env['Distributed training'] = self._distributed
        runtime_env['GPU number'] = self._world_size

        env_info = '\n    ' + '\n    '.join(f'{k}: {v}' for k, v in env.items())
        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        self.logger.info('\n' + dash_line + '\nSystem environment:' +
                         env_info + '\n'
                         '\nRuntime environment:' + runtime_env_info + '\n' +
                         dash_line + '\n')
