SETENV_TEMPLATE = '''#!/bin/bash

if [ $# -lt 1 ]; then
    path=/usr/local/Ascend
else
    path=$1
fi

if [ "${path: -1}" = "/" ]; then
  path=${path%/}
fi

export ASCEND_TOOLKIT_HOME=$path/latest
export ASCEND_HOME_PATH=$path/latest
source $path/latest/bin/setenv.bash
export PATH=$path/latest/bin:$PATH
export LD_LIBRARY_PATH=$path/latest/lib64:$LD_LIBRARY_PATH
msprof_path=$(realpath "$path/latest/tools/msprof/msprof")
ln -snf $path/latest/tools $path/latest/toolkit/tools

# adapt msprof
chmod +w $path/latest/toolkit/tools/profiler/bin
rm -f $path/latest/toolkit/tools/profiler/bin/msprof
cp $msprof_path $path/latest/toolkit/tools/profiler/bin
chmod +w $path/latest/lib64
ln -snf $path/latest/tools/msprof/libmsprofiler.so $path/latest/lib64/libmsprofiler.so
msprof_dir_path=$(realpath "$path/latest/tools/msprof")
cp -r $path/latest/toolkit/tools/profiler $msprof_dir_path/../../

# adapt toolkit/python ascendebug
rm -rf $path/latest/toolkit/python
ln -snf $path/latest/python $path/latest/toolkit/

content=$(cat <<EOF
{
    "Sanitizer": [
      {
        "Ascend910B": [
          "${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-vec.a",
          "${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-cube.a"
        ],
        "Ascend910_93": [
          "${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-vec.a",
          "${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-cube.a"
        ],
        "Ascend310P": [
          "${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-m200.a",
          "${ASCEND_HOME_PATH}/tools/mssanitizer/lib64/libsanitizer_stub_dav-m200-vec.a"
        ]
      }
    ]
}
EOF
)
chmod +w $ASCEND_HOME_PATH/conf/compile_options_config.json
echo $content > $ASCEND_HOME_PATH/conf/compile_options_config.json
'''