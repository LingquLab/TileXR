#!/usr/bin/env bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/../common_env.sh

prepare_3rd() {

echo "" > ${TILEXR_TEMP_HOME}/3rd.log

pkg_name=time
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/bin/time" ]]; then
    warn "install ${pkg_name} begin"

    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/time-1.9.tar.gz --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/

    cd ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time ./configure --prefix=${TILEXR_UTIL_HOME}/${pkg_name}/ >> ${TILEXR_TEMP_HOME}/3rd.log
    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd.log
    make install >> ${TILEXR_TEMP_HOME}/3rd.log

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=pigz
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/pigz" ]]; then
    warn "install ${pkg_name} begin"

    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/pigz-2.8.tar.gz --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/

    cd ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd.log
    mv ${TILEXR_TEMP_HOME}/${pkg_name}/*pigz ${TILEXR_UTIL_HOME}/${pkg_name}/

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=patch
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/bin/patch" ]]; then
    warn "install ${pkg_name} begin"

    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/patch-2.8.tar.gz --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/

    cd ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time ./configure --prefix=${TILEXR_UTIL_HOME}/${pkg_name}/ >> ${TILEXR_TEMP_HOME}/3rd.log
    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd.log
    make install >> ${TILEXR_TEMP_HOME}/3rd.log

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=ccache
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/ccache" ]]; then
    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    colorful_time tar -xJf ${TILEXR_3RD_OPEN_HOME}/ccache-4.12.2-linux-${TILEXR_OS_ARCH}.tar.xz --overwrite --strip-components=1 -C ${TILEXR_UTIL_HOME}/${pkg_name}/

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=cmake
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/bin/cmake" ]]; then
    warn "install ${pkg_name} begin"

    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/cmake-3.31.11.tar.gz --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/

    cd ${TILEXR_TEMP_HOME}/${pkg_name}/

    colorful_time ./bootstrap --prefix=${TILEXR_UTIL_HOME}/${pkg_name}/ --parallel=`nproc` > ${TILEXR_TEMP_HOME}/3rd-${pkg_name}.log
    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd-${pkg_name}.log
    make install >> ${TILEXR_TEMP_HOME}/3rd-${pkg_name}.log

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=ripgrep
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/rg" ]]; then
    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/ripgrep-15.1.0-${TILEXR_OS_ARCH}-unknown-linux-musl.tar.gz --overwrite --strip-components=1 -C ${TILEXR_UTIL_HOME}/${pkg_name}/

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success"
    else
        error "install ${pkg_name} failed"
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=sshpass
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/bin/sshpass" ]]; then
    warn "install ${pkg_name} begin"
    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/
    tar -xzf ${TILEXR_3RD_OPEN_HOME}/sshpass-1.06.tar.gz --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/
    cd ${TILEXR_TEMP_HOME}/${pkg_name}/
    colorful_time ./configure --prefix=${TILEXR_UTIL_HOME}/${pkg_name}/ >> ${TILEXR_TEMP_HOME}/3rd.log
    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd.log
    make install >> ${TILEXR_TEMP_HOME}/3rd.log

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success"
    else
        error "install ${pkg_name} failed"
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=mpich
if [[ ! -x "${MPI_HOME}/bin/mpirun" ]]; then
    warn "install ${pkg_name} begin"
    mkdir -p ${MPI_HOME}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/
    tar -xzf ${TILEXR_3RD_OPEN_HOME}/mpich-4.3.1.tar.gz --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/
    cd ${TILEXR_TEMP_HOME}/${pkg_name}/
    ./configure --prefix=${MPI_HOME}/ --disable-fortran >> ${TILEXR_TEMP_HOME}/3rd.log
    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd.log
    make install >> ${TILEXR_TEMP_HOME}/3rd.log

    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success"
    else
        error "install ${pkg_name} failed"
    fi
else
    warn "${pkg_name} already installed, skip."
fi

warn "install ops-transformer deps begin"
colorful_time python3 -m pip install -r ${TILEXR_OPS_HOME}/requirements.txt

if [ $? -eq 0 ]; then
    success "install ops-transformer deps success"
else
    error "install ops-transformer deps failed"
fi

cd $PWD

}

prepare_3rd
