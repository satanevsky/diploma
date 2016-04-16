#wget https://bootstrap.pypa.io/get-pip.py
#sudo python get-pip.py
#rm get-pip.py

sudo apt-get install cmake
sudo apt-get install build-essential
sudo apt-get install python-dev
sudo apt-get install unzip 

sudo pip install numpy 
sudo pip install cython
sudo pip install pandas 
sudo pip install scipy 
sudo pip install scikit-learn 
sudo pip install jupyter 


wget https://github.com/ndarray/Boost.NumPy/archive/master.zip
unzip master.zip
cd 
./configure
./make 
sudo make install 
cd ../
rm master.zip

wget https://github.com/sparsehash/sparsehash/archive/master.zip
unzip master.zip && cd master.zip 
./configure
./make 
sudo make install 
cd ../
rm master.zip


