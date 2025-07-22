#!/usr/bin/env bash
#
# User-only Apptainer installer
# – downloads Go and Apptainer to $HOME
# – builds with --without-suid so root is never required
# -------------------------------------------------------

set -euo pipefail

# ---------- tunables ----------
APPTAINER_VERSION=1.3.3          # latest stable as of 2024-07
GO_VERSION=1.21.11               # Go ≥1.18 is required
INSTALL_DIR="${HOME}/apptainer"  # final location
PARALLEL=$(nproc 2>/dev/null || echo 2)
# ------------------------------

echo "==> Installing Apptainer ${APPTAINER_VERSION} into ${INSTALL_DIR}"

# 1) create working directory
WORKDIR="${HOME}/tmp-apptainer-build"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 2) download and unpack Go to $HOME/go
if [[ ! -x "${HOME}/go/bin/go" ]]; then
  echo "==> Installing Go ${GO_VERSION}"
  GO_TGZ="go${GO_VERSION}.linux-amd64.tar.gz"
  wget -q "https://dl.google.com/go/${GO_TGZ}"
  tar -xf "${GO_TGZ}" -C "${HOME}"
  rm "${GO_TGZ}"
fi
export PATH="${HOME}/go/bin:${PATH}"
export GOPATH="${HOME}/gopath"
export GOROOT="${HOME}/go"

# 3) fetch Apptainer source
if [[ ! -d apptainer ]]; then
  echo "==> Fetching Apptainer source"
  wget -q "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer-${APPTAINER_VERSION}.tar.gz"
  tar -xf "apptainer-${APPTAINER_VERSION}.tar.gz"
  mv "apptainer-${APPTAINER_VERSION}" apptainer
fi
cd apptainer

# 4) configure and build
echo "==> Configuring (no-suid)"
./mconfig \
  --prefix="${INSTALL_DIR}" \
  --without-suid \
  --localstatedir="/tmp/${USER}/apptainer"

echo "==> Building (this may take a while)"
make -C builddir -j${PARALLEL}

echo "==> Installing to ${INSTALL_DIR}"
make -C builddir install

# 5) add to PATH
cat >> "${HOME}/.bashrc" <<EOF

# Apptainer (installed without root)
export PATH="${INSTALL_DIR}/bin:\$PATH"
EOF

# 6) clean-up
cd
rm -rf "${WORKDIR}"

echo
echo "Success!  Open a new shell or run:"
echo "  export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
echo "  apptainer --version"