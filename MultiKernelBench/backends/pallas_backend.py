import torch
from backends.backend_registry import register_backend, Backend
import os
import gc
import torch_xla.core.xla_model as xm
from utils.correctness import set_seed
from config import seed_num, num_correct_trials, num_perf_trials, num_warmup
import torch_xla
import torch_xla.debug.metrics as met

@register_backend('pallas')
class PallasBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()

    def get_device(self):
        return xm.xla_device()

    def get_hardware_name(self):
        return 'v2-8'

    def compile(self, generated_code, op):
        try:
            compile(generated_code, "<string>", "exec")
            exec(generated_code, self.context)
            return True, None
        except Exception as e:
            return False, str(e)

    def correctness_execution(self, ref_src):
        exec(ref_src, self.context)
        correctness = True
        correctness_information = ''
        get_inputs = self.context['get_inputs']
        get_init_inputs = self.context['get_init_inputs']
        Model = self.context['Model']
        ModelNew = self.context['ModelNew']
            
        try:
            init_inputs = get_init_inputs()
            init_inputs = [
                x.to(device=self.device) if isinstance(x, torch.Tensor) else x for x in init_inputs
            ]
            with torch.no_grad():
                set_seed(seed_num)  # set seed for reproducible weights
                original_model = Model(*init_inputs).to(self.device)
                torch_xla.sync(wait=True)
                set_seed(seed_num)
                custom_model = ModelNew(*init_inputs).to(self.device)
                torch_xla.sync(wait=True)
            with torch.no_grad():
                for trial in range(num_correct_trials):
                    inputs = get_inputs()
                    inputs = [
                        x.to(self.device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    ]
                    torch_xla.sync(wait=True)
                    ref_output = original_model(*inputs)       
                    torch_xla.sync(wait=True)
                    new_output = custom_model(*inputs)
                    torch_xla.sync(wait=True)
                    feedback = None
                    if ref_output.shape != new_output.shape:
                        feedback = f"[FAIL] Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                    elif not torch.allclose(ref_output, new_output, atol=1e-02, rtol=1e-02):
                        feedback = f"[FAIL] Output mismatch"
                    if feedback is not None:
                        correctness = False
                        correctness_information = feedback
                        break
        except Exception as e:
            print('[FAIL] runtime error when evaluating correctness')
            correctness = False
            correctness_information = f"[FAIL] {str(e)}"
            return correctness, correctness_information

        return correctness, correctness_information

    def time_execution(self, eval_target='ModelNew'):
        get_inputs = self.context['get_inputs']
        get_init_inputs = self.context['get_init_inputs']
        ModelNew = self.context[eval_target]
        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=self.device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            custom_model = ModelNew(*init_inputs).to(self.device)
        ExecuteTime_list = self.profile_op(get_inputs, custom_model)
        return ExecuteTime_list
    
    def profile_op(self, get_inputs, method):
        met.clear_counters()
        device = xm.xla_device()
        # warmup
        for _ in range(10):
            inputs = get_inputs()
            inputs = [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            output = method(*inputs)
            xm.mark_step()  # 确保所有操作完成
        res_time = []

        inputs = get_inputs()
        inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        met.clear_counters()
        met.clear_all()
        for _ in range(num_perf_trials + 1):
            met.clear_counters()
            met.clear_all()
            # 同步设备，确保计算完成
            output = method(*inputs)
            torch_xla.sync(wait=True)
            xm.mark_step()
            xm.wait_device_ops()
            # 获取ExecuteTime指标的数据
            # execute_time_ns = met.metric_data('ExecuteTime')[1]
            execute_time_ns = met.metric_data('ExecuteTime')[2][-1][1]
            execute_time_sec = execute_time_ns / 1e9
            res_time.append(execute_time_sec)
        return res_time[1:]
    
    def cleanup(self):
        del self.context
        gc.collect()
        xm.mark_step()
        xm.wait_device_ops()
