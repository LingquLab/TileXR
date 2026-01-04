source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
echo $LD_LIBRARY_PATH # 确认环境变量是否配置成功
export KERNEL_NAME=LcalTest
bash scripts/compile.sh
# 统一将kernel拷贝至kernel_file目录下
cp -r build/src/CMakeFiles/lccl_op3.dir/base_test.cpp.o kernel_file/base_test.cpp.o

EXECUTABLE="test_template"
ulimit -n 65536
CANN_DIR=$ASCEND_HOME_PATH
echo $EXECUTABLE
echo $CANN_DIR
export CANN_INC="${CANN_DIR}/aarch64-linux/include/"
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH
g++ -std=c++17 -g -I.. -o ${EXECUTABLE} ${EXECUTABLE}.cc -I"$CANN_INC" \
   -L${CANN_DIR}/aarch64-linux/simulator/Ascend910B1/lib \
   -L/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/simulator/Ascend910B1/lib/\
   -L"${CANN_DIR}/aarch64-linux/lib64" \
   -L"${CANN_DIR}/aarch64-linux/lib64/plugin/opskernel" \
   -L"${CANN_DIR}/acllib/lib64/stub"\
   -lruntime_camodel -lnpu_drv -lnpu_drv_camodel \
   /usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B1/lib/libffts_model.so -lstars -lffts -lpem_davinci -lmodel_top -lmcu_wrapper -lmcu_loop \
   -L${CANN_DIR}/lib64/stub -lascendcl

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/simulator/Ascend910B1/lib/:/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B1/lib/:$LD_LIBRARY_PATH
msprof op simulator  --soc-version=Ascend910B1 ./${EXECUTABLE}
# ./${EXECUTABLE}