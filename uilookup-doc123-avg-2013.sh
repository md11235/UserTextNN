java -Xmx8g -cp ./bin model.UILookupDocAvg123Main -embeddingLength 200 -embeddingFile /g/glove.6B/glove.6B.200d.txt  -year 2013 -windowSizeWordLookup1 1 -windowSizeWordLookup2 2 -windowSizeWordLookup3 3 -outputLengthWordLookup 50 -embeddingLengthUserLookup 75 -embeddingLengthItemLookup 75 -classNum 5 -inputDir ./dataset/ -roundNum 100 -probThreshold 0.001 -learningRate 0.03 -randomizeBase 0.01 -dumpUserEmbeddingFile user-embedding/2013-doc123-rerun-75dms- -dumpItemEmbeddingFile item-embedding/2013-doc123-rerun-75dms 
