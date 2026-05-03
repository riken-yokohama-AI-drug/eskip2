conda activate hyena-dna

if [ ! -e datasets ]
then
git clone https://github.com/riken-yokohama-AI-drug/eskip-data.git
mv eskip-data/datasets.zip
unzip datasets.zip
# datasets.zip will be uploaded after the paper is published.
fi

cd CLN3_5
ln -s ../../../hyena-dna/src
ln -s ../../../hyena-dna/csrc
ln -s ../../../hyena-dna/standalone_hyenadna.py
cp    ../datasets/test/CLN3_5.csv .
python predict_CLN3_5.py > predict.log
basename $PWD|tr '\n' ','
grep AUC predict.log
cd ../

cd COL7A1_73
ln -s ../../../hyena-dna/src
ln -s ../../../hyena-dna/csrc
ln -s ../../../hyena-dna/standalone_hyenadna.py
cp    ../datasets/test/COL7A1_73.csv .
python predict_COL7A1_73.py > predict.log
basename $PWD|tr '\n' ','
grep AUC predict.log
cd ../

cd COL7A1_80
ln -s ../../../hyena-dna/src
ln -s ../../../hyena-dna/csrc
ln -s ../../../hyena-dna/standalone_hyenadna.py
cp    ../datasets/test/COL7A1_80.csv .
python predict_COL7A1_80.py > predict.log
basename $PWD|tr '\n' ','
grep AUC predict.log
cd ../

cd NF1_17
ln -s ../../../hyena-dna/src
ln -s ../../../hyena-dna/csrc
ln -s ../../../hyena-dna/standalone_hyenadna.py
cp    ../datasets/test/NF1_17.csv .
python predict_NF1_17.py > predict.log
basename $PWD|tr '\n' ','
grep AUC predict.log
cd ../

cd SCN1A_20N
ln -s ../../../hyena-dna/src
ln -s ../../../hyena-dna/csrc
ln -s ../../../hyena-dna/standalone_hyenadna.py
cp    ../datasets/test/SCN1A_20N.csv .
python predict_SCN1A_20N.py > predict.log
basename $PWD|tr '\n' ','
grep AUC predict.log
cd ../

cd ATM_PE38
ln -s ../../../hyena-dna/src
ln -s ../../../hyena-dna/csrc
ln -s ../../../hyena-dna/standalone_hyenadna.py
cp    ../datasets/test/ATM_PE38_weak.csv .
python predict_ATM_PE38_weak.py > predict.log
basename $PWD|tr '\n' ','
grep AUC predict.log
cd ../
