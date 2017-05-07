package nnet;

import java.util.Random;

public class LinearLayer implements NNInterface{

	public double[][] W;
	public double[] b;
	
	public int inputLength;
	public int outputLength;
	
	public double[] input;
	public double[] output;
	
	public double[] inputG;
	public double[] outputG;
	
	public double[][] WG;
    public double[] bG;

    public double[][] WAdaLR;
    public double[] bAdaLR;
    
    public int linkId;
    
    private LinearLayer()
    {
    }

    public LinearLayer(int xInputLength,
    		int xOutputLength)
    {
    	this(xInputLength, xOutputLength, 0);
    }
    
    public LinearLayer(int xInputLength,
    		int xOutputLength,
    		int xLinkId)
    {
    	inputLength = xInputLength;
    	outputLength = xOutputLength;
    	linkId = xLinkId;
    	
    	W = new double[outputLength][];
    	WG = new double[outputLength][];
    	WAdaLR = new double[outputLength][];
    	
    	for(int i = 0; i < W.length; i++)
    	{
    		W[i] = new double[inputLength];
    		WG[i] = new double[inputLength];
    		WAdaLR[i] = new double[inputLength];
    	}
    	
    	b = new double[outputLength];
    	bG = new double[outputLength];
    	bAdaLR = new double[outputLength];
    	
    	input = new double[inputLength];
    	inputG = new double[inputLength];
    	output = new double[outputLength];
    	outputG = new double[outputLength];
    }
    
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				W[i][j] = r.nextFloat() * (max - min) + min;
			}
		}
		for(int i = 0; i < b.length; i++)
		{
			b[i] = r.nextFloat() * (max - min) + min;
		}
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		MathOp.A_times_x_plus_b(W, input, b, output);
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		MathOp.transA_times_x(outputG, W, inputG); // to be used by the previous layer
		
		MathOp.A_add_trans_x_times_y(outputG, input, WG);
		
		for (int i = 0; i < bG.length; ++i)
        {
            bG[i] += outputG[i];
        }
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		for(int i = 0; i < b.length; i++)
		{
			b[i] += learningRate * bG[i];
		}
		
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				W[i][j] += learningRate * WG[i][j];
			}
		}
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		for(int i = 0; i < b.length; i++)
		{
			bAdaLR[i] += (bG[i] / batchsize) * (bG[i] / batchsize);
			b[i] += learningRate / batchsize * bG[i] / Math.sqrt(bAdaLR[i]);
		}
		
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				WAdaLR[i][j] += (WG[i][j] / batchsize) * (WG[i][j] / batchsize);
				W[i][j] += (learningRate / batchsize) * WG[i][j] / Math.sqrt(WAdaLR[i][j]);
			}
		}
	}
	
	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				WG[i][j] = 0;
			}
		}
		
		for(int i = 0; i < b.length; i++)
		{
			bG[i] = 0;
		}
		
		for(int i = 0; i < outputG.length; i++)
		{
			outputG[i] = 0;
		}
		
		for(int i = 0; i < inputG.length; i++)
		{
			inputG[i] = 0;
		}
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != output.length || nextIG.length != outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		output = nextI;
		outputG = nextIG;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return input;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		
		LinearLayer clone = new LinearLayer();
		clone.linkId = linkId;
        clone.inputLength = inputLength;
        clone.outputLength = outputLength;
        
		clone.W = W;
        clone.b = b;
        clone.WG = new double[WG.length][];
        for(int i = 0; i < WG.length; i++)
        {
        	clone.WG[i] = new double[inputLength];
        }
        clone.bG = new double[bG.length];
        clone.bAdaLR = bAdaLR;
        clone.WAdaLR = WAdaLR;
        
        clone.input = new double[input.length];
        clone.inputG = new double[input.length];
        clone.output = new double[output.length];
        clone.outputG = new double[output.length];
        
		return clone;
	}

	public void regularizationLinear(double lambda) {
		for(int i = 0; i < b.length; i++)
		{
			b[i] -= lambda * b[i];
		}
		
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				W[i][j] -= lambda * W[i][j];
			}
		}
	}
}
