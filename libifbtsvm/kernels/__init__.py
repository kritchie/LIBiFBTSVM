from enum import Enum

# For now we use only standard sklearn kernels, custom kernels could
# live in their own files, e.g. my_custom_kernel.py could define the
# implementation of a custom kernel and then could be imported with:
# > from my_custom_kernel import CustomKernel.
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    RBFSampler
)

# Kernel constants
ACHI2_SAMPLE_STEPS = 10
ACHI2_SAMPLE_INTERVAL = 1

RBF_GAMMA = 1
RBF_RANDOM_STATE = 1
RBF_N_COMPONENTS = 100


class Kernel(Enum):
    ACHI2 = AdditiveChi2Sampler(sample_steps=ACHI2_SAMPLE_STEPS, sample_interval=ACHI2_SAMPLE_INTERVAL)
    RBF = RBFSampler(gamma=RBF_GAMMA, random_state=RBF_RANDOM_STATE)

    def fit_transform(self, *args, **kwargs):
        return self.value.fit_transform(*args, **kwargs)
