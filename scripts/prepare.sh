#!/usr/bin/env bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

python_bin=""
for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c 'import sys' >/dev/null 2>&1; then
        python_bin="$(command -v "${candidate}")"
        break
    fi
done
if [[ -z "${python_bin}" ]] || ! "${python_bin}" -m pip --version >/dev/null 2>&1; then
    error "Python with pip is required before running prepare.sh."
    exit 1
fi

warn "install PyPI mermaid-cli begin"
if ! colorful_time "${python_bin}" -m pip install mermaid-cli; then
    error "pip install mermaid-cli failed."
    exit 1
fi

mermaid_scripts_dir="$("${python_bin}" - <<'PY'
import sysconfig

print(sysconfig.get_path("scripts"))
PY
)"
mermaid_cli_bin="${mermaid_scripts_dir}/mmdc"
if [[ ! -x "${mermaid_cli_bin}" ]]; then
    error "PyPI mermaid-cli did not install an executable mmdc at ${mermaid_cli_bin}."
    exit 1
fi
if ! "${mermaid_cli_bin}" --help 2>&1 | grep -q -- '--playwright-config-file'; then
    error "mmdc at ${mermaid_cli_bin} is not the expected PyPI Playwright interface."
    exit 1
fi

warn "install Playwright Chromium begin"
if ! colorful_time "${python_bin}" -m playwright install chromium; then
    error "Playwright Chromium installation failed."
    exit 1
fi
if ! "${python_bin}" - <<'PY'
from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-setuid-sandbox"],
    )
    browser.close()
PY
then
    error "Playwright Chromium validation failed."
    exit 1
fi
success "PyPI mermaid-cli and Playwright Chromium are ready."

bash ${script_path}/download_open_source_deps.sh

echo "" > ${TILEXR_TEMP_HOME}/3rd.log

pkg_name=time
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/bin/time" ]]; then
    _install_autoconf_pkg time time-1.9.tar.gz
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
    cd ${TILEXR_HOME}
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=patch
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/bin/patch" ]]; then
    _install_autoconf_pkg patch patch-2.8.tar.gz
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
cmake_bin="${TILEXR_UTIL_HOME}/${pkg_name}/bin/cmake"
cmake_pkg="cmake-3.22.6.tar.gz"
cmake_expected_prefix="3.22."
need_install_cmake=1

if [[ -x "${cmake_bin}" ]]; then
    cmake_current_version=$(${cmake_bin} --version 2>/dev/null | awk '/version/{print $3; exit}')
    if [[ "${cmake_current_version}" == ${cmake_expected_prefix}* ]]; then
        need_install_cmake=0
    else
        warn "${pkg_name} version mismatch (${cmake_current_version}), reinstall ${cmake_expected_prefix}x"
    fi
fi

if [[ ${need_install_cmake} -eq 1 ]]; then
    warn "install ${pkg_name} begin"
    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    mkdir -p ${TILEXR_TEMP_HOME}/${pkg_name}/
    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/${cmake_pkg} --overwrite --strip-components=1 -C ${TILEXR_TEMP_HOME}/${pkg_name}/
    cd ${TILEXR_TEMP_HOME}/${pkg_name}/
    colorful_time ./bootstrap --prefix=${TILEXR_UTIL_HOME}/${pkg_name}/ --parallel=`nproc` > ${TILEXR_TEMP_HOME}/3rd-${pkg_name}.log
    colorful_time make -j`nproc` >> ${TILEXR_TEMP_HOME}/3rd-${pkg_name}.log
    make install >> ${TILEXR_TEMP_HOME}/3rd-${pkg_name}.log
    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
    fi
    cd ${TILEXR_HOME}
else
    warn "${pkg_name} already installed (${cmake_current_version}), skip."
fi

pkg_name=ripgrep
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/rg" ]]; then
    mkdir -p ${TILEXR_UTIL_HOME}/${pkg_name}/
    case "${TILEXR_OS_ARCH}" in
        x86_64)
            rg_pkg="ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz"
            ;;
        aarch64|arm64)
            rg_pkg="ripgrep-15.1.0-aarch64-unknown-linux-gnu.tar.gz"
            ;;
        *)
            error "unsupported arch for ripgrep: ${TILEXR_OS_ARCH}"
            exit 1
            ;;
    esac
    colorful_time tar -xzf ${TILEXR_3RD_OPEN_HOME}/${rg_pkg} --overwrite --strip-components=1 -C ${TILEXR_UTIL_HOME}/${pkg_name}/
    if [ $? -eq 0 ]; then
        success "install ${pkg_name} success."
    else
        error "install ${pkg_name} failed."
    fi
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=sshpass
if [[ ! -x "${TILEXR_UTIL_HOME}/${pkg_name}/bin/sshpass" ]]; then
    _install_autoconf_pkg sshpass sshpass-1.06.tar.gz
else
    warn "${pkg_name} already installed, skip."
fi

pkg_name=mpich
if [[ ! -x "${MPI_HOME}/bin/mpirun" ]]; then
    _install_autoconf_pkg mpich mpich-4.3.1.tar.gz --disable-fortran
else
    warn "${pkg_name} already installed, skip."
fi

cd ${TILEXR_HOME}
