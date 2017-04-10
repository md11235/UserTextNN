package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.omg.CORBA.UserException;

import other.*;
import nnet.*;

public class UILookupDocAvg123Main {

	LookupLinearTanh xseedLLT1;
	LookupLinearTanh xseedLLT2;
	LookupLinearTanh xseedLLT3;
	
	MultiConnectLayer connect;
	AverageLayer average;
	
	LookupLayer userLookup;
	LookupLayer itemLookup;
	MultiConnectLayer connectUserItem;
	
	LinearLayer linearForSoftmax;
	SoftmaxLayer softmax;
	
	HashMap<String, Integer> wordVocab = null;
	
	HashMap<String, Integer> userVocab;
	HashMap<String, Integer> itemVocab;
	
	String unkStr = "unk"; //"<unk>";
	
	public UILookupDocAvg123Main(
				String embeddingFileWord, 
				int embeddingLengthWord,
				int windowSizeWordLookup1,
				int windowSizeWordLookup2,
				int windowSizeWordLookup3,
				int outputLengthWordLookup,
				int embeddingLengthUserLookup,
				int embeddingLengthItemLookup,
				int classNum,
				String trainFile,
				String testFile,
				double randomizeBase) throws Exception
	{
		loadData(trainFile, testFile);
		
		wordVocab = new HashMap<String, Integer>();
		
		int embeddingLineCount = Funcs.lineCounter(embeddingFileWord, "utf8");
		double[][] table = new double[embeddingLineCount][];
		Funcs.loadEmbeddingFile(embeddingFileWord, embeddingLengthWord, "utf8", 
				false, wordVocab, table);
		
		xseedLLT1 = new LookupLinearTanh(windowSizeWordLookup1, wordVocab.size(), outputLengthWordLookup, embeddingLengthWord);
		xseedLLT1.lookup.setEmbeddings(table);
			
		xseedLLT2 = new LookupLinearTanh(windowSizeWordLookup2, wordVocab.size(), outputLengthWordLookup, embeddingLengthWord);
		xseedLLT2.lookup.setEmbeddings(table);
		
		xseedLLT3 = new LookupLinearTanh(windowSizeWordLookup3, wordVocab.size(), outputLengthWordLookup, embeddingLengthWord);
		xseedLLT3.lookup.setEmbeddings(table);
		
		connect = new MultiConnectLayer(
				new int[]{outputLengthWordLookup, outputLengthWordLookup, outputLengthWordLookup});
		
		average = new AverageLayer(connect.outputLength, outputLengthWordLookup);
		connect.link(average);
		
		// user item lookup layers
		userLookup = new LookupLayer(embeddingLengthUserLookup, userVocab.size(), 1);
		itemLookup = new LookupLayer(embeddingLengthItemLookup, itemVocab.size(), 1);
		
		connectUserItem = new MultiConnectLayer(
				new int[]{average.outputLength, userLookup.output.length, itemLookup.output.length});
		
		average.link(connectUserItem, 0);
		userLookup.link(connectUserItem, 1);
		itemLookup.link(connectUserItem, 2);
		
		// linear for softmax
		linearForSoftmax = new LinearLayer(connectUserItem.outputLength, classNum);
		connectUserItem.link(linearForSoftmax);
		
		softmax = new SoftmaxLayer(classNum);
		linearForSoftmax.link(softmax);
		
		Random rnd = new Random(); 
		xseedLLT1.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		xseedLLT2.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		xseedLLT3.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
	}
	
	List<Data> trainDataList;
	List<Data> testDataList;  
	
	public void loadData(
			String trainFile,
			String testFile)
	{
		System.out.println("================ start loading corpus ==============");
		trainDataList = new ArrayList<Data>();  
		
		userVocab = new HashMap<String, Integer>();
		itemVocab = new HashMap<String, Integer>();
		
		Funcs.loadCorpus(trainFile, "utf8", trainDataList);
		
		for(Data data: trainDataList)
		{
			if(!userVocab.containsKey(data.userStr))
			{
				userVocab.put(data.userStr, userVocab.size());
			}
			if(!itemVocab.containsKey(data.productStr))
			{
				itemVocab.put(data.productStr, itemVocab.size());
			}
		}
		
		testDataList = new ArrayList<Data>();  
		Funcs.loadCorpus(testFile, "utf8", testDataList);
		
		System.out.println("training size: " + trainDataList.size());
		System.out.println("testDataList size: " + testDataList.size());
		System.out.println("userVocab.size(): " + userVocab.size());
		System.out.println("itemVocab.size(): " + itemVocab.size());
		System.out.println("================ finsh loading corpus ==============");
	}
	
	public void run(
			int roundNum,
			double probThreshould,
			double learningRate,
			int classNum,
			String dumpUserEmbeddingFile,
			String dumpItemEmbeddingFile
			) throws Exception
	{
		double lossV = 0.0;
		int lossC = 0;
		for(int round = 1; round <= roundNum; round++)
		{
			System.out.println("============== running round: " + round + " ===============");
			Collections.shuffle(trainDataList, new Random());

			for(int idxData = 0; idxData < trainDataList.size(); idxData++)
			{
				Data data = trainDataList.get(idxData);
				
				String[] sentences = data.reviewText.split("<sssss>");
				int[][] wordIdMatrix = Funcs.fillDocument(sentences, wordVocab, unkStr);
				
				DocAverage docAverage1 = new DocAverage(
						xseedLLT1,
						wordIdMatrix, 
						wordVocab.get(unkStr));
				
				DocAverage docAverage2 = new DocAverage(
						xseedLLT2,
						wordIdMatrix, 
						wordVocab.get(unkStr));
				
				DocAverage docAverage3 = new DocAverage(
						xseedLLT3,
						wordIdMatrix, 
						wordVocab.get(unkStr));
				
				if(docAverage1.sentenceConvList.size() == 0 
						|| docAverage2.sentenceConvList.size() == 0
						|| docAverage3.sentenceConvList.size() == 0)
				{
					System.out.println(data.toString() + "docAverage.sentenceConvList.size() == 0");
					continue;
				}
				
				userLookup.input[0] = userVocab.get(data.userStr);
				itemLookup.input[0] = itemVocab.get(data.productStr);
				
				// important
				docAverage1.link(connect, 0);
				docAverage2.link(connect, 1);
				docAverage3.link(connect, 2);
				
				// forward
 				docAverage1.forward();
 				docAverage2.forward();
 				docAverage3.forward();
 				connect.forward();
				average.forward();

				userLookup.forward();
				itemLookup.forward();
				connectUserItem.forward();
				
 				linearForSoftmax.forward();
				softmax.forward();
				
				// set cross-entropy error 
				// we minus 1 because the saved goldRating is in range 1~5, while what we need is in range 0~4
				int goldRating = data.goldRating - 1;
				lossV += -Math.log(softmax.output[goldRating]);
				lossC += 1;
				
				for(int k = 0; k < softmax.outputG.length; k++)
					softmax.outputG[k] = 0.0;
				
				if(softmax.output[goldRating] < probThreshould)
					softmax.outputG[goldRating] =  1.0 / probThreshould;
				else
					softmax.outputG[goldRating] = 1.0 / softmax.output[goldRating];
				
				// backward
				softmax.backward();
				linearForSoftmax.backward();
				
				connectUserItem.backward();
				userLookup.backward();
				itemLookup.backward();
				
				average.backward();
				connect.backward();
				docAverage1.backward();
				docAverage2.backward();
				docAverage3.backward();
				
				// update
				linearForSoftmax.update(learningRate);
				docAverage1.update(learningRate);
				docAverage2.update(learningRate);
				docAverage3.update(learningRate);
				userLookup.update(learningRate);
				itemLookup.update(learningRate);
				
				// regularization ?
//				if(lossC % regularizationFreq == 0)
//				{
//					docAverage.regularization(lambda);
//				}
				
				// clearGrad
				docAverage1.clearGrad();
				docAverage2.clearGrad();
				docAverage3.clearGrad();
				connect.clearGrad();
				average.clearGrad();

				connectUserItem.clearGrad();
				userLookup.clearGrad();
				itemLookup.clearGrad();
				
				linearForSoftmax.clearGrad();
				softmax.clearGrad();
				
				if(idxData % 100 == 0)
				{
					System.out.println("running idxData = " + idxData + "/" + trainDataList.size() + "\t "
							+ "lossV/lossC = " + lossV + "/" + lossC + "\t"
							+ " = " + lossV/lossC
							+ "\t" + new Date().toLocaleString());
				}
			}
			
			Funcs.dumpEmbedFile(dumpUserEmbeddingFile + "-" + round, "utf8", userVocab, userLookup.table, userLookup.embeddingLength);
			Funcs.dumpEmbedFile(dumpItemEmbeddingFile + "-" + round, "utf8", itemVocab, itemLookup.table, itemLookup.embeddingLength);
			
			System.out.println("============= finish training round: " + round + " ==============");
			
			predict(round);
		}
	}
	
	public void predict(int round) throws Exception
	{
		System.out.println("=========== predicting round: " + round + " ===============");
		
		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();
		
		for(int idxData = 0; idxData < testDataList.size(); idxData++)
		{
			Data data = testDataList.get(idxData);
			
			String[] sentences = data.reviewText.split("<sssss>");
			int[][] wordIdMatrix = Funcs.fillDocument(sentences, wordVocab, unkStr);
			
			DocAverage docAverage1 = new DocAverage(
					xseedLLT1,
					wordIdMatrix, 
					wordVocab.get(unkStr));
			
			DocAverage docAverage2 = new DocAverage(
					xseedLLT2,
					wordIdMatrix, 
					wordVocab.get(unkStr));
			
			DocAverage docAverage3 = new DocAverage(
					xseedLLT3,
					wordIdMatrix, 
					wordVocab.get(unkStr));
			
			if(docAverage1.sentenceConvList.size() == 0 
					|| docAverage2.sentenceConvList.size() == 0
					|| docAverage3.sentenceConvList.size() == 0)
			{
				System.out.println(data.toString() + "docAverage.sentenceConvList.size() == 0");
				continue;
			}
			
			userLookup.input[0] = userVocab.get(data.userStr);
			itemLookup.input[0] = itemVocab.get(data.productStr);
			
			// important
			docAverage1.link(connect, 0);
			docAverage2.link(connect, 1);
			docAverage3.link(connect, 2);
			
			// forward
			docAverage1.forward();
			docAverage2.forward();
			docAverage3.forward();
			connect.forward();
			average.forward();

			userLookup.forward();
			itemLookup.forward();
			connectUserItem.forward();
			
			linearForSoftmax.forward();
			softmax.forward();
			
			int predClass = -1;
			double maxPredProb = -1.0;
			for(int ii = 0; ii < softmax.length; ii++)
			{
				if(softmax.output[ii] > maxPredProb)
				{
					maxPredProb = softmax.output[ii];
					predClass = ii;
				}
			}
			
			predList.add(predClass + 1);
			goldList.add(data.goldRating);
		}
		
		Metric.calcMetric(goldList, predList);
		System.out.println("============== finish predicting =================");
	}
	
	public static void main(String[] args) throws Exception
	{
		HashMap<String, String> argsMap = Funcs.parseArgs(args);
		
		System.out.println("==== begin configuration ====");
		for(String key: argsMap.keySet())
		{
			System.out.println(key + "\t\t" + argsMap.get(key));
		}
		System.out.println("==== end configuration ====");
		
		int embeddingLength = Integer.parseInt(argsMap.get("-embeddingLength"));
		String embeddingFile = argsMap.get("-embeddingFile");
		// for yelp14 dataset, windowsize = 1&2&3 works best than other settings. 
		int windowSizeWordLookup1 = Integer.parseInt(argsMap.get("-windowSizeWordLookup1"));
		int windowSizeWordLookup2 = Integer.parseInt(argsMap.get("-windowSizeWordLookup2"));
		int windowSizeWordLookup3 = Integer.parseInt(argsMap.get("-windowSizeWordLookup3"));
		int outputLengthWordLookup = Integer.parseInt(argsMap.get("-outputLengthWordLookup"));
		int classNum = Integer.parseInt(argsMap.get("-classNum"));
		
		int embeddingLengthUserLookup = Integer.parseInt(argsMap.get("-embeddingLengthUserLookup"));
		int embeddingLengthItemLookup = Integer.parseInt(argsMap.get("-embeddingLengthItemLookup"));
		
		String year = argsMap.get("-year");
		String inputDir = argsMap.get("-inputDir");
		int roundNum = Integer.parseInt(argsMap.get("-roundNum"));
		double probThreshold = Double.parseDouble(argsMap.get("-probThreshold"));
		double learningRate = Double.parseDouble(argsMap.get("-learningRate"));
		double randomizeBase = Double.parseDouble(argsMap.get("-randomizeBase"));
		
		String trainFile = inputDir + "/yelp-" + year + "-seg-20-20.train.ss";
		String testFile = inputDir + "/yelp-" + year + "-seg-20-20.test.ss";
		
		if(year.equals("amazon"))
		{
			trainFile = inputDir + "/train.txt.ss.new";
			testFile = inputDir + "/test.txt.ss.new";
		}
		
		String dumpUserEmbeddingFile = argsMap.get("-dumpUserEmbeddingFile");
		String dumpItemEmbeddingFile = argsMap.get("-dumpItemEmbeddingFile");
		
		UILookupDocAvg123Main main = new UILookupDocAvg123Main(
				embeddingFile, 
				embeddingLength, 
				windowSizeWordLookup1,
				windowSizeWordLookup2,
				windowSizeWordLookup3,
				outputLengthWordLookup,
				embeddingLengthUserLookup,
				embeddingLengthItemLookup,
				classNum, 
				trainFile, 
				testFile,
				randomizeBase);
		
		main.run(roundNum, 
				probThreshold, 
				learningRate, 
				classNum,
				dumpUserEmbeddingFile,
				dumpItemEmbeddingFile);
	}
}
