from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 跨平台编译设置
# extra_compile_args = {
#     'cxx': [
#         '-O3',
#         '-march=native',        # 启用本地CPU指令集
#         '-fopenmp',             # OpenMP支持
#         '-ffloat-store',        # 防止过度浮点优化
#         '-fno-associative-math',# 禁用破坏结合律的优化
#         '-fno-reciprocal-math', # 禁用不安全的倒数优化
#         '-fno-trapping-math',   # 禁用浮点陷阱
#         '-fno-unsafe-math-optimizations',
#         '-DNDEBUG',             # 禁用调试断言
#     ]
# }

# # Windows特殊处理
# if os.name == 'nt':
#     extra_compile_args['cxx'] = [
#         '/O2',                  # MSVC的优化级别
#         '/openmp',              # MSVC的OpenMP
#         '/fp:precise',          # 精确浮点模式
#         '/arch:AVX2',           # 启用AVX2指令集
#     ]

# # Linux/MacOS额外设置
# if sys.platform != 'win32':
#     extra_compile_args['cxx'].extend([
#         # '-mno-avx512f',         # 避免AVX512频率降频
#         '-fno-fast-math',       # 禁用不安全的数学优化
#     ])

# # 精度调试模式（可通过环境变量启用）
# if os.getenv('FPS_DEBUG_PRECISION'):
#     extra_compile_args['cxx'].extend([
#         '-DFPS_DEBUG=1',
#         '-fno-omit-frame-pointer',
#         '-fsanitize=float-cast-overflow',
#     ])

# setup(
#     name='fps_cpu',
#     version='0.1.0',
#     ext_modules=[
#         CppExtension(
#             name='fps_cpu',
#             sources=['csrc/fps_cpu_multi.cpp'],
#             extra_compile_args=extra_compile_args['cxx'],
#             extra_link_args=extra_compile_args['cxx'],
#             define_macros=[
#                 ('MAX_POINTS', '131072'),  # 默认栈分配阈值
#                 ('FPS_USE_KAHLAN', '1'),   # 启用Kahan求和
#                 ('FPS_STRICT_CHECK', '1'), # 启用严格检查
#             ],
#         ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension.with_options(
#             use_ninja=True,      # 使用ninja构建工具
#             warn_unused=True,    # 显示未使用变量警告
#         )
#     },
#     install_requires=[
#         'torch>=1.8.0',         # 最低版本要求
#     ],
# )

# v0.0.1
# setup(
#     name='fps_cpu',
#     version='0.0.1',
#     ext_modules=[
#         CppExtension(
#             'fps_cpu',
#             ['csrc/fps_cpu.cpp'],
#             extra_compile_args=['-O3']  # 优化标志
#         ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     },
# )

# v0.1.0
setup(
    name='fps_cpu',
    version='0.1.0',
    ext_modules=[
        CppExtension(
            'fps_cpu',
            ['csrc/fps_cpu_v02.cpp'],
            extra_compile_args=["-O3","-shared","-std=c++17","-fPIC", "-fopenmp", "-I/usr/local/include/"]  # 优化标志
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)