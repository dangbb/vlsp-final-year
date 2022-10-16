# vlsp-final-year

# Installation note 

In project root directory,

## Install SBERT deploy 

```commandline
cd external/sentence_transformer/
gdown https://drive.google.com/a/gm.uit.edu.vn/uc?id=1pXJZ9eHp6DWkQ5MhCzmWYsKyLQEDiodz&export=download
tar xzf /content/vn_sbert_deploy.tar.gz
```

## Install VnCodeNLP

```commandline
mkdir external/sentence_transformer/vncorenlp
cd external/sentence_transformer/
mkdir -p vncorenlp/models/wordsegmenter
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```

## Change env.json to configure a run 