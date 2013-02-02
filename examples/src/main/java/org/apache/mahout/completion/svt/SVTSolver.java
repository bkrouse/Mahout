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
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.PlusMult;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import static org.apache.mahout.math.function.Functions.*;
import org.apache.mahout.math.Vector;
//import org.apache.mahout.math.matrix.DoubleMatrix1D;
//import org.apache.mahout.math.matrix.DoubleMatrix2D;
//import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix2D;
//import org.apache.mahout.math.matrix.linalg.EigenvalueDecomposition;
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
 * 		<li>TODO: include the EPS option?
 * </ul>
 * </p>
 * <p>Outputs: matrix X stored in SVD format X = U*diag(S)*V'
 * <ul>
 * 		<li>U - numRows x rank left singular vectors
 * 		<li>S - rank x 1 singular values
 * 		<li>V - numColumns x rank right singular vectors
 * 		<li>numiter - number of iterations to achieve convergence
 * 		<li>TODO: include the output structure with data from each iteration?
					%   output - a structure with data from each iteration.  Includes:
					%       output.nuclearNorm  - nuclear norm of current iterate
					%       output.rank         - rank of current iterate
					%       output.time         - time taken for one iteraration
					%       output.residual     - the relative residual, norm(x-b)/norm(b)
 * </ul>
 * </p>
*/

public class SVTSolver {

  private static final Logger log = LoggerFactory.getLogger(SVTSolver.class);

  private static final double NANOS_IN_MILLI = 1.0e6;

  private static final int DEFAULT_MAXITER = 500;
  private static final double DEFAULT_TOL = 1.0e-4;
  private static final int R_INC = 4;  //make this larger for more accuracy

  public enum TimingSection {
    TBD
  }

  private final Map<TimingSection, Long> startTimes = new EnumMap<TimingSection, Long>(TimingSection.class);
  private final Map<TimingSection, Long> times = new EnumMap<TimingSection, Long>(TimingSection.class);

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
  	//check overwrite and output contents -- if something is there, delete or bail as appropriate
  	
  	//run algorithm to complete the matrix
  	
  	//optionally write the final U, S, V somewhere?  who knows, will I want to be able to inspect the intermediate ones too?
  	//write another "SVT Results" data structure out for reports on processing time?
  	
  	//write out the final completed matrix
  	
  	//TEMP stub step: just transpose the matrix from inputPath into outputPath location
    DistributedRowMatrix matrix = new DistributedRowMatrix(inputPath, workingPath, numRows, numCols);
    matrix.setConf(conf);
    
    Path outputPathDir = new Path(outputPath.toString().concat("-dir"));
    Path partPath = new Path(outputPathDir.toString().concat("/part-00000"));
    DistributedRowMatrix matrixTransposed = matrix.transpose(outputPathDir);
    
    //rename the single file and delete the directory
    FileSystem fs = outputPathDir.getFileSystem(conf);
    fs.rename(partPath, outputPath);
    fs.delete(outputPathDir,true);
    

    
  	return;
  }
  

  
  /*
  public SVTSolver.Result solve(SparseRowMatrix P,
			double tau,
			double delta) {

  	return solve(P, tau, delta, DEFAULT_MAXITER, DEFAULT_TOL);
  }
  
  public SVTSolver.Result solve(SparseRowMatrix P,
  									double tau,
  									double delta,
  									int maxiter,
  									double tol) {

*/

/*
  	int minDim = Math.min(P.numRows(), P.numCols());
  	int k0 = (int)Math.ceil(tau / (delta*estimateNorm(P, 1e-2)) );  	
  	SparseRowMatrix Y = P.times(k0*delta);
  	
  	int r=0, s=0;
  	Matrix U = null, V = null;
  	List<Double> Sigma = null;
  	for (int k=1; k<=maxiter; k++) {
  		log.info("SVT - Iteration {}", k);  		
  		s = r + 1;
  		while (true) {
  			//TODO: I think I need to tweak things here...since Lanczos is only working on a symmetric...I am 
  			//only getting back the left singular vector and the singular values...which means I need to run it again on A'*A to get right singular vectors
  			//But HOW does the Matlab version do it when the input matrix is not symmetric??
  			//Also, I may not be getting back the TOP singular values...might need to get 2-3x as many, and then sort??
  			//I probably need to "clean" the eigenvectors that come back too?  Seems that Lanczos should do this itself
  	    U = new DenseMatrix(s, P.numCols());
  	    Sigma = new ArrayList<Double>();
  	    LanczosSolver solver = new LanczosSolver();
  	    solver.solve(Y, s, U, Sigma, false);

  			if( (Sigma.get(s)<=tau) || (s==minDim) ) break;
  			s = Math.min(s + 5, minDim);
  		}
  	
  	//	sigma = diag(Sigma); r = sum(sigma > tau); -- setting rank to # of sing vals > tau
  	//	U = U(:,1:r); V = V(:,1:r); sigma = sigma(1:r) - tau; Sigma = diag(sigma);

  	//	//Returns x = projection of X on Omega
  	//	x = XonOmega(U*diag(sigma), V, Omega) -- this is a separate MEX function
  	
  	//  relRes = frob_norm(x-b)/normb;
  	//	fprintf('iteration %4d, rank is %2d, rel. residual is %.1e\n',k,r,norm(x-b)/normb);

  	//	if (relRes < tol)
  	//		break;
  	//	if (relRes > 1e5)
  	//		log.Error(Error!  Diverence!)
  	//		break;
  	
  	//	y = y + delta*(b-x);
  	//	updateSparse(Y,y,indx);  -- here's that separate MEX function again...
  	}	
  	
  	SVTSolver.Result result = new SVTSolver.Result();
  	result.U = U;
  	result.S = S;
  	result.V = V;
  	result.numiter = numiter;
  	return result;

	  return null;
  }
*/
  
  private void startTime(TimingSection section) {
    startTimes.put(section, System.nanoTime());
  }

  private void endTime(TimingSection section) {
    if (!times.containsKey(section)) {
      times.put(section, 0L);
    }
    times.put(section, times.get(section) + (System.nanoTime() - startTimes.get(section)));
  }

  public double getTimeMillis(TimingSection section) {
    return ((double) times.get(section)) / NANOS_IN_MILLI;
  }

  public class Result {
  	public Matrix U = null;
  	public Matrix S = null;
  	public Matrix V = null;
  	int numiter = 0;
  }

	//TODO: implement estimateNorm()
  private double estimateNorm(Matrix M, double tol) {
  	//matlab uses a power method here, until within tolerance.
  	return 0;
  }
  
  /**
   * exclude hidden (starting with dot) files
   */
  private static class ExcludeDotFiles implements PathFilter {
    @Override
    public boolean accept(Path file) {
      return !file.getName().startsWith(".");
    }
  }
}
