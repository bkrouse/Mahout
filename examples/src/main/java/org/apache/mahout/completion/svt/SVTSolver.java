/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.completion.svt;


import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.decomposer.lanczos.LanczosSolver;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDSolver;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.io.Closeables;

/**
 * <p>SVT Solver</p>
 * <p>Code converted from <a href="http://svt.caltech.edu/">Matlab version </a> by Emmanuel Candes and Stephen Becker.</p>
 * <p>Uses SVT algorithm to complete a low-rank matrix from a sampling of it's elements.</p>
 * <p>More precisely,
 * <ul>
 * 		<li> Finds the minimum of   tau ||X||_* + .5 || X ||_F^2
 * 		<li> subject to P_Omega(X) = P_Omega(M)
 * 		<li> using linear Bregman iterations
 * </ul>
 * </p>
 * <p>Inputs:
 * <ul>
 * 		<li>numRows - number of rows of matrix M (to be completed)
 * 		<li>numColumns - number of columns of matrix M (to be completed)
 *		<li>Omega - Vector containing list of the observed entries (i.e. positions of M)
 *		<li>b - Vector containing list of the values of the observed entries (i.e. values of M)
 *		<li>tau - parameter defining the objective functional
 *		<li>delta - step size.  Choose delta less than 2 to be safe but conservative (i.e. slower);
 *				choose delta closer to numRows*numColumns/Omega.size() to be riskier (i.e. algorithm may diverge)
 *		<li>maxiter - maximum number of iterations
 *		<li>tol - stopping criteria 
 * </ul>
 * </p>
 * <p>Output written to output path: matrix X stored in SVD format X = U*diag(S)*V'
 * <ul>
 * 		<li>U - numRows x rank left singular vectors
 * 		<li>S - rank x 1 singular values
 * 		<li>V - numColumns x rank right singular vectors
 * 		<li>output log with data from each iteration
 * 		<li>timing log 
 * </ul>
 * </p>
*/

public class SVTSolver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(SVTSolver.class);

	private static final double DIVERGENCE = 1.0e5;

	private static double CALC_DEFAULT_STEPSIZE = 0.05; //not the actual default -- trigger to calculate a recommended value
	private static double DEFAULT_TOLERANCE = 1.0e-4;
	private static double CALC_DEFAULT_THRESHOLD = -1; //not the actual default -- trigger to calculate a recommended value
	private static int DEFAULT_INCREMENT = 4; //make this larger for more accuracy
	private static int DEFAULT_MAXITER = 500;

	private static String U_THRESHOLDED_REL_PATH = "UThresholded";
	private static String V_THRESHOLDED_REL_PATH = "VThresholded";
	private static String S_THRESHOLDED_REL_PATH = "DiagS";
		
  @Override
  public int run(String[] args) throws Exception {
  	
    addInputOption();
    addOutputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    addOption("stepSize", "ss", "Step size - delta", Double.toString(CALC_DEFAULT_STEPSIZE)); 
    addOption("tolerance", "err", "Tolerance - epsilon", Double.toString(DEFAULT_TOLERANCE)); 
    addOption("threshold", "tao", "Threshold - tao", Double.toString(CALC_DEFAULT_THRESHOLD)); 
    addOption("increment", "incr", "Increment - l", Integer.toString(DEFAULT_INCREMENT)); 
    addOption("maxIter", "iter", "Maximum iteration count - k_max", Integer.toString(DEFAULT_MAXITER)); 
    addOption(DefaultOptionCreator.overwriteOption().create());

    Map<String, List<String>> pargs = parseArguments(args);
    if (pargs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));
    double stepSize = Double.parseDouble(getOption("stepSize"));
    stepSize = (stepSize==-1) ? calcDefaultStepSize(numRows, numCols) : stepSize;
    double tolerance = Double.parseDouble(getOption("tolerance"));
    double threshold = Double.parseDouble(getOption("threshold"));
    threshold = (threshold==-1) ? calcDefaultThreshold(numRows, numCols) : threshold;
    int increment = Integer.parseInt(getOption("increment"));
    int maxIter = Integer.parseInt(getOption("maxIter"));
    boolean overwrite =
      pargs.containsKey(keyFor(DefaultOptionCreator.OVERWRITE_OPTION));

    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Path tempPath = getTempPath();


    solve(conf,
                     inputPath,
                     outputPath,
                     new Path(tempPath, "svt-working"),
                     numRows,
                     numCols,
                     stepSize,
                     tolerance,
                     threshold,
                     increment,
                     maxIter,
                     overwrite);

 

    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SVTSolver(), args);
  }

  
  public void solve(Configuration conf,
	      Path inputPath,
	      Path outputPath,
	      Path workingPath,
	      int numRows,
	      int numCols,
	      boolean overwrite) throws IOException
	  {  	
  		solve(conf,
          inputPath, outputPath, workingPath, numRows, numCols, calcDefaultStepSize(numRows, numCols), DEFAULT_TOLERANCE, calcDefaultThreshold(numRows, numCols),
          DEFAULT_INCREMENT, DEFAULT_MAXITER, overwrite);

	  }
  
  /**
   * Run the solver to complete the matrix using SVT algorithm.  
   * 
   * @param inputPath the Path to the input matrix stored as a sequence file
   * @param outputPath the Path to the output matrix stored as a sequence file
   * @param workingPath a Path to a temporary working directory
   * @param numRows the int number of rows 
   * @param stepSize the delta in SVT algorithm
   * @param tolerance the tolerance epsilon in SVT algorithm
   * @param threshold the thresholding tao in SVT algorithm
   * @param increment the increment l in SVT algorithm
   * @param maxIter the maximum iterations k_max in SVT algorithm
   * @param overwrite whether to overwrite the contents of the output directory or not
   * @param increment 
   */
  
  public void solve(Configuration conf,
      Path inputPath,
      Path outputPath,
      Path workingPath,
      int numRows,
      int numCols,
      double stepSize,
      double tolerance,
      double threshold,
      int increment,
      int maxIter,
      boolean overwrite) throws IOException
  {  	
  	log.info("SVTSolver: start");
  	
    FileSystem fs = outputPath.getFileSystem(conf);
        
	  DistributedRowMatrix matrixtmp = new DistributedRowMatrix(new Path(workingPath.getParent(), "test/sampled-m-repartitioned-6137"), workingPath, numRows, numCols);
	  matrixtmp.setConf(conf);
	  DistributedRowMatrix XminusMtemp = new DistributedRowMatrix(new Path(workingPath.getParent(), "test/XminusM-repartitioned-61697"), workingPath, numRows, numCols);
	  XminusMtemp.setConf(conf);
	  DistributedRowMatrix XminusMonOmegatmp = XminusMtemp.projection(new Path(workingPath.getParent(), "test/XminusMonOmega"), matrixtmp);

	  if(1==1)
	  	return;
    
    //cleanup outputPath and workingPath is overwrite is true, otherwise bail
    if(overwrite) {
  		fs.delete(outputPath, true);
  		fs.delete(workingPath, true);
  	}
  	else {
  		if(fs.exists(outputPath))
  			throw new IOException("outputPath has contents and overwrite=false: " + outputPath.toString());
  		
  		if(fs.exists(workingPath))
  			throw new IOException("workingPath has contents and overwrite=false: " + workingPath.toString());
  	}
  	
    
    
    long timingStart = 0, timingEnd = 0;
    
  	//run algorithm to complete the matrix
    DistributedRowMatrix matrix = new DistributedRowMatrix(inputPath, workingPath, numRows, numCols);
    matrix.setConf(conf);
    
    //kicking step
  	timingStart = System.currentTimeMillis();
    double norm2 = norm2(matrix);
  	int k0 = (int)Math.ceil(threshold / (stepSize*norm2) );  	
  	DistributedRowMatrix Y = matrix.times(new Path(workingPath, "Y0"), k0*stepSize);  
  	timingEnd = System.currentTimeMillis();
  	writeTimingResults(0, "kickingStep", timingEnd - timingStart);

  	//calculate matrixFrobNorm -- will use in each iteration below
  	timingStart = System.currentTimeMillis();
  	double matrixFrobNorm = matrix.frobeniusNorm();
  	timingEnd = System.currentTimeMillis();
  	writeTimingResults(0, "frobeniusNorm", timingEnd - timingStart);
  	
    //main SVT loop
    int r=0, s=0;
    Path iterationWorkingPath = null;
    for (int k=1; k<=maxIter; k++) {
    	long iterationStart = System.currentTimeMillis();
    	iterationWorkingPath = new Path(workingPath, Integer.toString(k));
    	s = r+1;
    	
    	//get the first s singular values/vectors of the matrix Y
    	//check if the last singular value is <= tao
    	//if its not, s = s + increment -- and do it again
    	//TODO: currently hard-coding s (based on my input matrix) -- need to make this incremental.
    	//			this will involve modifying Lanczos to (a) handle asym matrices, and (b) handle iterating up to a threshold
    	//			(which internally means letting it save work between iterations)
    	//			For now, I'll use SSVD (since it handles asym), but hard code s (since SSVD doesn't
    	//			handle restarts.  
    	//			I may need to interact with the community on this one.  But first, I want to get 
    	//			through a skeleton of the entire SVT, and show some results to Dr Ye.
    	//		  I may also need to do some studies into the accuracy of results, based on the rank
    	//			that I am trying to achieve and based on how many internal iterations of Lanczos.
    	//			Maybe while I'm at it, I can also try to make Lanczos faster.
    	//			Note - I'll probably want to use DistributedLanczosSolver -- because it can be restarted on the HdfsLanczosState, and
    	//			also because it ties to the EigenVerificationJob (i.e. cleansvd) at the end if it's run if I send in those params.
    	//			I think I'll need to do this after settling on the right rank in each iteration -- or at least understand why it's needed --
    	//			Since I need not only the singular values, but the vectors as well.  That too might need adjustment to support asymmetric?
    	
    	timingStart = System.currentTimeMillis();
    	SVD svd;
    	while (true) {
    		 s = 14;  //hard code this for now -- will ultimately increment this in steps, and pass in to computeSVD my saved work
    		 svd = computeSVD(conf, iterationWorkingPath, Y, s);
    		 break;
    	}
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "computeSVD", timingEnd - timingStart);

    	//set r to the index of the smallest singular value greater than tao
    	while(r < svd.S.size()) {
    		if(svd.S.get(r) <= threshold) {
    			if (r > 0)  { r--; }  //TODO: what to do if threshold is greater than even the first singular value?  right now, just making sure I don't go out of bounds
    			break;
    		}
    		r++;
    	}
    	 

    	//truncate U and V, up to r+1th (i.e. vector whose index is r) vectors each (all rows, up to r+1 columns)
    	//truncate S up to r+1 values, and subtract threshold from each value
    	//TODO: make the truncate and threshold more efficient -- currently makes a copy of both matrices
    	//SSVD returns U and V as distributed matrix format -- so it's not as easy to take the first r+1 vectors.
    	//Lanczos returns the singular vectors as vectors -- so it's easy peasy to take the first r+1.
    	//ultimately, I know I will need to use Lanczos -- for the iterative approach
    	//but in my current workaround that uses SSVD, I need to just quickly pull out the first r+1 vectors
    	//so until I switch to Lanczos...we'll just hack something together that works, even if it's a bit expensive
    	timingStart = System.currentTimeMillis();
    	svd.truncateAndThreshold(conf, r, threshold);    	
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "truncateAndThreshold", timingEnd - timingStart);

    	//calculate X = U'*S*V -- with truncated singular values
    	//currently assuming I can just hard-code 1 partition coming back....that may change with larger data sets?  Dunno
    	//TODO: maybe I dynamically discover the "natural" partition size, and use that?
    	//exactly what SSVD will return in those cases...
    	//not sure why it matters...but I was getting weird results until I setConf(conf) on each matrix.
    	timingStart = System.currentTimeMillis();
    	DistributedRowMatrix DiagS = svd.getDiagS(conf);
      DiagS.setConf(conf);    	
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "getDiagS", timingEnd - timingStart);

    	timingStart = System.currentTimeMillis();
    	DistributedRowMatrix Utrans = svd.U.transpose(new Path(iterationWorkingPath,"Utrans")); 
    	Utrans.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "Utrans", timingEnd - timingStart);

    	timingStart = System.currentTimeMillis();
  		DistributedRowMatrix Vtrans = svd.V.transpose(new Path(iterationWorkingPath,"Vtrans"));  
    	Vtrans.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "Vtrans", timingEnd - timingStart);
    	
    	timingStart = System.currentTimeMillis();
    	DistributedRowMatrix SV = DiagS.times(Vtrans, new Path(iterationWorkingPath,"SV")); 
    	SV.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "SV=DiagS.times(Vtrans)", timingEnd - timingStart);

    	timingStart = System.currentTimeMillis();
    	DistributedRowMatrix X = Utrans.times(SV, new Path(iterationWorkingPath,"X")); 
    	X.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "X=Utrans.times(SV)", timingEnd - timingStart);
    	  	

    	//checking stopping conditions
    	timingStart = System.currentTimeMillis();
    	DistributedRowMatrix XminusM = X.plus(new Path(iterationWorkingPath, "XminusM"), matrix, -1);
    	XminusM.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "XMinusM", timingEnd - timingStart);

    	timingStart = System.currentTimeMillis();
    	DistributedRowMatrix XminusMonOmega = XminusM.projection(new Path(iterationWorkingPath, "XminusMonOmega"), matrix);
    	XminusMonOmega.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "XminusM.projection(Omega)", timingEnd - timingStart);

    	
    	timingStart = System.currentTimeMillis();
    	double relRes = XminusMonOmega.frobeniusNorm() / matrixFrobNorm; 
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "XminusMonOmega.frobeniusNorm() / matrixFrobNorm", timingEnd - timingStart);
    	
    	
    	//log and exit if we're within goal tolerance -- or bail when it diverges
    	if (relRes <= tolerance)
    	{
      	writeIterationResults(k, r, relRes, System.currentTimeMillis() - iterationStart);
    		break;
    	}
    	else if (relRes > DIVERGENCE) {
    		throw new IOException("relRes diverged in iteration " + Integer.toString(k));
    	}
    	
    	
    	//calculate Y for next iteration
    	timingStart = System.currentTimeMillis();
    	DistributedRowMatrix YplusStep = Y.plus(new Path(iterationWorkingPath, "YplusStep"), XminusM, -1*stepSize); //this is same as Y + delta*Projection(M-X) -- just re-using XminusM 
    	YplusStep.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "YplusStep", timingEnd - timingStart);

    	timingStart = System.currentTimeMillis();
    	Y = YplusStep.projection(new Path(iterationWorkingPath,"Y"), matrix);  
    	Y.setConf(conf);
    	timingEnd = System.currentTimeMillis();
    	writeTimingResults(k, "YplusStep.projection(Omega)", timingEnd - timingStart);
    		    	
    	//TODO - add the option to clean up last iteration's working directory?
    	
    	this.writeIterationResults(k, r, relRes, System.currentTimeMillis() - iterationStart);
    }
    
    //move U, S and V to the outputPath
    if(iterationWorkingPath!=null) {
	    FileUtil.copy(fs, new Path(iterationWorkingPath,U_THRESHOLDED_REL_PATH), fs, new Path(outputPath,U_THRESHOLDED_REL_PATH), false, conf);
	    FileUtil.copy(fs, new Path(iterationWorkingPath,S_THRESHOLDED_REL_PATH), fs, new Path(outputPath,S_THRESHOLDED_REL_PATH), false, conf);
	    FileUtil.copy(fs, new Path(iterationWorkingPath,V_THRESHOLDED_REL_PATH), fs, new Path(outputPath,V_THRESHOLDED_REL_PATH), false, conf);
    }    
    
     
  	return;
  }

  
  private SVD computeSVD(Configuration conf, Path workingPath, DistributedRowMatrix Y, int desiredRank) throws IOException {
  	//use SSVD to compute for now -- eventually switch to Lanczos when it can (a) handle asym, (b) iterate up to a threshold
	  int r = 10000;
	  int k = desiredRank;
	  int p = 15;
	  int reduceTasks = 10; //check shell scripts for better default
	  SSVDSolver solver = 
	    new SSVDSolver(conf,
	                   new Path[] {Y.getRowPath()},
	                   workingPath,
	                   r,
	                   k,
	                   p,
	                   reduceTasks);
	
	  solver.setMinSplitSize(-1);
	  solver.setComputeU(true);
	  solver.setComputeV(true);
	  solver.setcUHalfSigma(false);
	  solver.setcVHalfSigma(false);
	  solver.setcUSigma(false);
	  solver.setOuterBlockHeight(30000);
	  solver.setAbtBlockHeight(200000);
	  solver.setQ(2);
	  solver.setBroadcast(false);  //BK: setting this to true seemed to be causing an error in BtJob
	  solver.setOverwrite(true);
	  solver.run();

	  SVD svd = new SVD();
	  svd.U = new DistributedRowMatrix(new Path(solver.getUPath()), workingPath, Y.numRows(), k);
    svd.U.setConf(conf);
    svd.S = solver.getSingularValues().viewPart(0, k);
	  svd.V = new DistributedRowMatrix(new Path(solver.getVPath()), workingPath, Y.numCols(), k);
    svd.V.setConf(conf);

    
    
  	return svd;
  }
  
  
  /*
   * norm2 calculates the 2-norm, which is the largest singular value
   */
	private double norm2(DistributedRowMatrix matrix) throws IOException {
		
		//TODO: come back and understand if I should be using DistributedLanczosSolver instead
	  //compute Lanczos SVD of matrix^2 (recommended rank is 2k+1 for k vectors) -- and get sqrt of top singular value
	  int desiredRank = 3; //should be 2k + 1....or maybe a convergence?  But in my manual testing, I found I needed about 10 for 0.01 accuracy
	  DistributedRowMatrix matrixSquared = matrix.times(matrix);
	  //TODO: can I do better with this initial vector?  See the unclear comments at the start of LanczosSolver class...
	  Vector initialVector = new DenseVector(matrixSquared.numRows());
	  initialVector.assign(1.0 / Math.sqrt(matrixSquared.numRows()));
	  LanczosState state = new LanczosState(matrixSquared, desiredRank, initialVector);
	  LanczosSolver solver = new LanczosSolver();
	  solver.solve(state, desiredRank, true);
	  double topSingular = state.getSingularValue(desiredRank - 1);	  //compute SVD of m -- and get top singular value		
	  double norm2 = Math.sqrt(topSingular);
	  
		return norm2;
		
//		//One way to do it...that is MIGHTY slow!  Oh well....
//		double NORM2EST_TOLERANCE = 0.01;
//    double norm2est = matrix.norm2est(NORM2EST_TOLERANCE);
//
//    return norm2est;
	}


/* Comments on defaults for tau and delta from the matlab code:
  
   %{
   if n1 and n2 are very different, then
     tau should probably be bigger than 5*sqrt(n1*n2)

   increase tau to increase accuracy; decrease it for speed

   if the algorithm doesn't work well, try changing tau and delta
     i.e. if it diverges, try a smaller delta (e.g. delta < 2 is a 
     safe choice, but the algorithm may be slower than necessary).
  %}
*/	private double calcDefaultStepSize(int numRows,
			int numCols) {
		
		int r = 10;
		double df = r*(numRows+numCols-r);
		double m = Math.min(5*df, Math.round(0.99*numRows*numCols));
		double p = m / (numRows*numCols);
		return 1.2 / p;
	}

  
	private double calcDefaultThreshold(int numRows,
			int numCols) {
		
		return 5*Math.sqrt(numRows*numCols);
	}
  


  private class SVD {
  	public SVD() 
  	{}
  	
  	public void truncateAndThreshold(Configuration conf, int r, double threshold) throws IOException {
    	U = U.viewColumns(new Path(U.getRowPath().getParent(),U_THRESHOLDED_REL_PATH), 0, r); //in terms of indexes -- start index of 0, end index of r
    	U.setConf(conf);
    	V = V.viewColumns(new Path(U.getRowPath().getParent(),V_THRESHOLDED_REL_PATH), 0, r);
    	V.setConf(conf);
    	S = S.viewPart(0, r+1).plus(threshold*-1);  //viewPart(startIndex, length) -- since r is end index, want r+1 as length
			
		}

  	//TODO: this is a hack...1) Can't call it twice (will blow up), 2) do I really need to do this, or can I just compute X smarter?
  	//But still....lets just get it working please
  	public DistributedRowMatrix getDiagS(Configuration conf) throws IOException {
  		
  		//create a DistributedRowMatrix version of Diag(S)
  		Path diagSPath = new Path(U.getRowPath().getParent(), S_THRESHOLDED_REL_PATH);
  		
      FileSystem fs = FileSystem.get(diagSPath.toUri(), conf);
      SequenceFile.Writer writer = null;
      try {  	
        writer = new SequenceFile.Writer(fs,
            conf,
            diagSPath,
            IntWritable.class,
            VectorWritable.class);
        
        for (int row=0; row < S.size(); row++) {
  	      Vector v = new RandomAccessSparseVector(S.size());
  	      v.setQuick(row, S.getQuick(row));
  	      writer.append(new IntWritable(row), new VectorWritable(v));     
        }
      } finally {
      	Closeables.closeQuietly(writer);
      }
  		
  		DistributedRowMatrix diagS = new DistributedRowMatrix(diagSPath,
                              U.getOutputTempPath(),
                              S.size(),
                              S.size());
  		diagS.setConf(conf);
  		
  		return diagS;
  		
  	}
  	
		public DistributedRowMatrix U = null;
  	public Vector S = null;
  	public DistributedRowMatrix V = null;
  }

  private void writeIterationResults(int iterationNum, int rank, double relativeResidual, long iterationTiming) throws IOException {
  	log.info("SVTSolver: iterationNum=" + iterationNum + ",rank=" + Integer.toString(rank+1) + ",relativeResidual=" + relativeResidual + ",iterationTiming=" + iterationTiming);
  }
  
  private void writeTimingResults(int iterationNum, String label, long timing) throws IOException {
  	log.info("SVTSolver: iterationNum="+iterationNum + ",label=" + label + ",timing=" + timing);
  }
}
