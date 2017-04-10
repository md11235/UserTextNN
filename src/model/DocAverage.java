package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import nnet.*;

public class DocAverage implements NNInterface{

	public List<SentenceConv> sentenceConvList;
	public MultiConnectLayer connectSentenceConv;
	public AverageLayer averageSentenceConv;
	
	public int linkId;
	
	public DocAverage() {
	}
	
	public DocAverage(
			LookupLinearTanh seedLLT,
			int[][] wordIdMatrix,
			int unkIdx) throws Exception
	{
		int hiddenLength = seedLLT.outputLength;
		sentenceConvList = new ArrayList<SentenceConv>();
		
		int windowSizeWord = seedLLT.lookup.inputLength;
		
		for(int i = 0; i < wordIdMatrix.length; i++)
		{
			if(wordIdMatrix[i].length >= windowSizeWord)
			{
				sentenceConvList.add(new SentenceConv(wordIdMatrix[i], seedLLT));
			}
			else
			{
				int[] tmpIds = new int[windowSizeWord];
				for(int k = 0; k < windowSizeWord; k++)
				{
					if(k < wordIdMatrix[i].length)
					{
						tmpIds[k] = wordIdMatrix[i][k];
					}
					else
					{
						tmpIds[k] = unkIdx;
					}
				}
//				sentenceConvList.add(new SentenceConv(tmpIds, seedLLT));
				continue;
			}
		}
		
		if(sentenceConvList.size() == 0)
		{
			return;
		}
		
		int[] sentenceConvLengths = new int[sentenceConvList.size()];
		Arrays.fill(sentenceConvLengths, hiddenLength);
		
		connectSentenceConv = new MultiConnectLayer(sentenceConvLengths);
		
		for(int k = 0; k < sentenceConvLengths.length; k++)
		{
			sentenceConvList.get(k).link(connectSentenceConv, k);
		}
		
		averageSentenceConv = new AverageLayer(connectSentenceConv.outputLength, hiddenLength);
		connectSentenceConv.link(averageSentenceConv);
	}

	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward() {
		for(SentenceConv layer: sentenceConvList)
		{
			layer.forward();
		}
		connectSentenceConv.forward();
		averageSentenceConv.forward();
	}

	@Override
	public void backward() {
		averageSentenceConv.backward();
		connectSentenceConv.backward();
		for(SentenceConv layer: sentenceConvList)
		{
			layer.backward();
		}
	}

	@Override
	public void update(double learningRate) {
		for(SentenceConv layer: sentenceConvList)
		{
			layer.update(learningRate);
		}
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		
	}

	@Override
	public void clearGrad() {
		for(SentenceConv layer: sentenceConvList)
		{
			layer.clearGrad();
		}
		connectSentenceConv.clearGrad();
		averageSentenceConv.clearGrad();
		
		sentenceConvList.clear();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != averageSentenceConv.output.length || nextIG.length != averageSentenceConv.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		averageSentenceConv.output = nextI;
		averageSentenceConv.outputG = nextIG;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return averageSentenceConv.output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return averageSentenceConv.outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		return null;
	}

}
