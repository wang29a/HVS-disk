export ADDITIONAL_DEFINITIONS=$1

mkdir build
cd build
cmake ..
make -j
