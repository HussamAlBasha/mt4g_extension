# Building MT4G on MPCDF Viper (MI300A)

## Environment

- **Cluster:** MPCDF Viper (`viper11`)
- **Target GPU:** AMD MI300A (APU) → `gfx942`
- **ROCm:** 7.2.1

## Prerequisites

```bash
module load rocm/7.2
```

## Build

```bash
mkdir -p build && cd build
```

```bash
rm -rf *
```

```bash
cmake .. -DGPU_TARGET_ARCH=gfx942 \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH=/u/halba/local \
         -DCMAKE_INSTALL_PREFIX=/u/halba \
         -DCMAKE_CXX_FLAGS="-I/u/halba/local/include"
```

```bash
make all -j $(nproc)
```

```bash
make install 
```

## Install dependencies locally (no root, no spack)

```bash
mkdir -p /u/halba/local

# nlohmann-json v3.11.3
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git
cd json && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/u/halba/local -DJSON_BuildTests=OFF
make install && cd ../..

# cxxopts v3.2.0
git clone --depth 1 --branch v3.2.0 https://github.com/jarro2783/cxxopts.git
cd cxxopts && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/u/halba/local -DCXXOPTS_BUILD_EXAMPLES=OFF -DCXXOPTS_BUILD_TESTS=OFF
make install && cd ../..

# libdrm headers (workaround: system has /usr/include/drm/ but ROCm expects libdrm/)
ln -s /usr/include/drm /u/halba/local/include/libdrm
```

