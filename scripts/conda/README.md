---------- Complete build process
git clone http://github.com/sd-ot/pysdot.git
cd pysdot/scripts/conda

conda activate
conda install anaconda anaconda-client conda-build
conda build . --output --python=3.7

anaconda login
anaconda upload -u sdot xxx.tar.bz2


---------- Complete process for windows
download visual studio
download miniconda
launch anaconda prompt
conda install git

pip install mpi4py
pip install petsc petsc4py

---------- To test if it works (e.g. from a docker instance)
# docker run -it debian /bin/bash
cd
apt-get update && apt-get upgrade -y -q && apt-get dist-upgrade -y -q && apt-get -y -q autoclean && apt-get -y -q autoremove
apt-get install -y -q curl bzip2 git
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
. /root/miniconda3/etc/profile.d/conda.sh
conda create --name test_sdot

conda activate test_sdot
conda install -c hugo_lec pysdot


