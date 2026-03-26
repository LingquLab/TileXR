## 1. 移动源码目录

- [x] 1.1 `git mv comm src/comm` — 保留 git 历史
- [x] 1.2 `git mv include src/include` — 保留 git 历史
- [x] 1.3 `git mv mc2 src/mc2` — 保留 git 历史

## 2. 更新 CMakeLists.txt

- [x] 2.1 根 `CMakeLists.txt`：`include_directories` 中 `${CMAKE_CURRENT_LIST_DIR}/include/` 改为 `${CMAKE_CURRENT_LIST_DIR}/src/include/`
- [x] 2.2 根 `CMakeLists.txt`：`add_subdirectory(comm)` 改为 `add_subdirectory(src/comm)`

## 3. 更新 mc2/build.sh 的 TILEXR_INCS

- [x] 3.1 在 `TILEXR_INCS` 赋值前，用 `TILEXR_SRC_INCLUDE=$(realpath "${CURRENT_DIR}/../src/include")` 计算头文件绝对路径
- [x] 3.2 `TILEXR_INCS` 追加 `-I${TILEXR_SRC_INCLUDE} -I${TILEXR_SRC_COMM}`，覆盖 include/ 和 comm/ 两个路径

## 4. 消除深层相对 #include

- [x] 4.1 `src/mc2/all_gather/op_host/op_api/aclnn_all_gather.h`：深层 `include/` 和 `comm/` 路径改为直接文件名
- [x] 4.2 `src/mc2/all_gather/examples/test_aclnn_all_gather.cpp`：深层 `include/` 和 `comm/` 路径改为直接文件名

## 5. 更新 ops_build_run.sh 路径

- [x] 5.1 `cp -rf ${TILEXR_HOME}/mc2/${ops}` → `cp -rf ${TILEXR_HOME}/src/mc2/${ops}`
- [x] 5.2 `cp -rf ${TILEXR_HOME}/mc2/*` → `cp -rf ${TILEXR_HOME}/src/mc2/*`
- [x] 5.3 `cp -rf ${TILEXR_HOME}/include/comm_args.h` → `cp -rf ${TILEXR_HOME}/src/include/comm_args.h`
- [x] 5.4 `cp -rf ${TILEXR_HOME}/include/tilexr_sync.h` → `cp -rf ${TILEXR_HOME}/src/include/tilexr_sync.h`
- [x] 5.5 `cp -f ${TILEXR_HOME}/mc2/build.sh` → `cp -f ${TILEXR_HOME}/src/mc2/build.sh`

## 6. 验证

- [ ] 6.1 `source common_env.sh && cmake . && make -j$(nproc)` 编译通过，产出 `libtile-comm.so`
- [x] 6.2 扫描 `src/mc2/` 确认无超过两层 `../` 的 `#include`（指向 include/ 和 comm/ 的深层引用已清除）
- [x] 6.3 `git status` 确认无意外遗留的旧路径文件
