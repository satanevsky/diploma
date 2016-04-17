sudo dpkg --configure -a
sudo apt-get -f install
sudo apt-get --fix-missing install
sudo apt-get clean
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libboost-all-dev
sudo apt-get install python-dev
sudo apt-get install python-pandas 
sudo apt-get install unzip 
sudo apt-get clean
sudo apt-get autoremove

rm -f get-pip.py
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
rm get-pip.py

sudo pip install numpy 
sudo pip install cython
sudo pip install scipy 
sudo pip install scikit-learn 
sudo pip install jupyter

rm -f master.zip
rm -f -r Boost.NumPy-master

wget https://github.com/ndarray/Boost.NumPy/archive/master.zip
unzip master.zip
cd Boost.NumPy-master
cmake .
./configure
make 
sudo make install 
cd ../
rm master.zip
rm -r Boost.NumPy-master

rm -f master.zip
rm -f -r sparsehash-master 

wget https://github.com/sparsehash/sparsehash/archive/master.zip
unzip master.zip
cd sparsehash-master 
./configure
./make 
sudo make install 
cd ../
rm master.zip
rm -r sparsehash-master 


