# tilexr

TileXR — eXtreme Rendezvous for Asynchronous Tile Communication

| 支持的代码仓                                                                  | 版本号       |
| ----------------------------------------------------------------------------- | ------------ |
| [hcomm](https://gitcode.com/cann/hcomm/tree/9.0.0-beta.1)                     | 9.0.0-beta.1 |
| [ops-transformer](https://gitcode.com/cann/ops-transformer/tree/9.0.0-beta.1) | 9.0.0-beta.1 |
| [opbase](https://gitcode.com/cann/opbase/tree/9.0.0-beta.1)                   | 9.0.0-beta.1 |

### 0. 前置依赖

1. 支持的OS: Ubuntu 20.04LTS (安装系统依赖: apt install -y build-essential git git-lfs rdma-core kmod net-tools libssl-dev libz-dev libeigen3-dev python3 python3-pip)
2. 驱动版本 **25.5.0** 以上, 请执行 `npu-smi info` 检查一下驱动的版本号
3. 下载代码仓 `git clone --recursive -b 9.0.0-beta.1 https://gitcode.com/LingquLab/TileXR.git`
4. 请在**root**用户下执行CANN安装和编译操作

### 1. 环境准备

1. **自动下载并安装CANN包**

```shellscript
bash cann_download_install.sh
```

### 2. hcomm 本地编译

```shellscript
bash hcomm_build_install.sh
```

### 3. ops-transformer 编译+执行

```shellscript
bash ops_build_run.sh
```

### 4. ops-transformer 仅执行

```shellscript
bash ops_only_run.sh
```

### 5. opbase 编译安装

```shellscript
bash opbase_build_install.sh
```

### 6. 日志查看工具

```shellscript
bash plog_grep.sh ERROR
```
