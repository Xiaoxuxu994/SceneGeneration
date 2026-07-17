import os 
import shutil 
from setuptools import setup, Extension
from torch.utils import cpp_extension

# # some global variables 
ascend_op_extension_names = ['ascendc_kernels_npu_pcd_dist']
ascend_toolkit_install_path='/usr/local/Ascend/ascend-toolkit/latest'

pytorch_extension_name = 'kernel_pcd_dist'
pytorch_interface_cpp_files = ['kernel_pcd_dist.cpp']


# build pytorch extension 
import torch 
import torch_npu
torch_npu_path = torch_npu.__file__
torch_npu_path = '/'.join(torch_npu_path.split('/')[:-1])
extra_link_args = [
    f'-Wl,-rpath={torch_npu_path}/lib,-rpath={os.getcwd()}'
]
setup(name=pytorch_extension_name,
      ext_modules=[cpp_extension.CppExtension(pytorch_extension_name, pytorch_interface_cpp_files,
                                              include_dirs=[
                                                  '/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/torch_npu/include',
                                                  ],
                                              library_dirs=[
                                                  '/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/torch_npu/lib',
                                                  ascend_toolkit_install_path + '/aarch64-linux/lib64',
                                                  os.getcwd(),
                                                  ],
                                              libraries= ascend_op_extension_names + ['torch_npu'],
                                              extra_link_args=extra_link_args,
                                            #   extra_compile_args=['-O2']
                                              )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
