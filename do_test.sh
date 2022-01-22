rootpath=$1
testCollection=$2
logger_name=$3
overwrite=0
collectionStrt=single

gpu=$4

CUDA_VISIBLE_DEVICES=$gpu python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

