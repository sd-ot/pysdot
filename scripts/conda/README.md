---------- Complete build process
git clone http://github.com/sd-ot/pysdot.git
cd pysdot/scripts/conda

conda activate
<!-- conda install anaconda-client conda-build -->
<!-- anaconda login -->
<!-- conda build . --output --python=3.7 -->
anaconda upload --force -u sdot `conda build . --output --python=3.6`
anaconda upload --force -u sdot `conda build . --output --python=3.7`

conda build --user hugo_lec . 
 anaconda login --username  --password vrnpv2T92JbSTwB
--------- installation

conda install -c sdot pysdot 

--------- install macosx
VBoxManage registervm 

---------- Complete process for windows
download visual studio 2015
download miniconda
launch anaconda prompt
conda install git

conda build . --output --python=3.7
anaconda login
anaconda upload -u sdot xxx.tar.bz2

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


