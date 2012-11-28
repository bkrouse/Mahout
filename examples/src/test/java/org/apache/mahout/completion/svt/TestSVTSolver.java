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

import org.apache.commons.math.analysis.BinaryFunction;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.function.Functions;
//import org.apache.mahout.math.matrix.DoubleMatrix1D;
//import org.apache.mahout.math.matrix.DoubleMatrix2D;
//import org.apache.mahout.math.matrix.linalg.EigenvalueDecomposition;
//import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix2D;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.BufferedWriter;

public final class TestSVTSolver extends SolverTest {
	private static final Logger log = LoggerFactory.getLogger(TestSVTSolver.class);

  @Test
  public void testSVTSolver() throws Exception {
  	
  	//Create a random FULL matrix, for a given rank and size.  
  	//NOTE: not sure what the right range of values should be for M...
  	int n1 = 150; int n2 = 300; int r = 10; int multiplier = 20;
  	Matrix N1 = randomDenseMatrix(n1, r, multiplier);
  	Matrix N2 = randomDenseMatrix(r, n2, multiplier);
  	Matrix M = N1.times(N2);
  	
  	//I'm not sure what this does, but sets some values:
  	int df = r*(n1+n2-r);
  	int oversampling = 5; 
  	int m = (int)Math.min(5*df,Math.round(.99*n1*n2) ); 
  	double p  = m/(n1*n2);

  	
  	//Omega = random samples of numbers from 1 to n1*n2.
  	Vector Omega = new DenseVector(m);
    Random rand = new Random(1234L);
    int n1_x_n2 = n1*n2;
    for (int i = 0; i < m; i++) {
      Omega.setQuick(i, rand.nextInt(n1_x_n2));
    }
    
  	//data = the values of M, corresponding to Omega.
    Vector data = extractSampledData(M, Omega); 
  	
  	//TODO: add in noise (later)
    double sigma = 0;
    //sigma = .05*stdev(data);
  	
    log.info("Matrix completion: {} x {} matrix", n1, n2);
    log.info(", rank {}, {} observations", r, m);
    log.info(", oversampling degrees of freedom by {); noise std is {}", m/df, sigma);
  	
		/*
		 if n1 and n2 are very different, then
		   tau should probably be bigger than 5*sqrt(n1*n2)
		
		 increase tau to increase accuracy; decrease it for speed
		
		 if the algorithm doesn't work well, try changing tau and delta
		   i.e. if it diverges, try a smaller delta (e.g. delta < 2 is a 
		   safe choice, but the algorithm may be slower than necessary).
	 */
  	double tau = 5*Math.sqrt(n1*n2); 
  	double delta = 1.2/p;
  	int maxiter = 500;
  	double tol = 1*10^(-4);

  	//TODO: fill this out properly...create a sparse matrix based on M and Omega above
  	Matrix P = new SparseRowMatrix(n1, n2);

  	
  	log.info("Solving by SVT...");
  	SVTSolver solver = new SVTSolver();
/*
    	SVTSolver.Result result = solver.solve(P, tau, delta, maxiter, tol);
 
>>>>>>> a few changes, pre-deployment
  	  	    
  	log.info("Calculating Xopt...");
  	Matrix U = result.U;
  	Matrix S = result.S;
  	Matrix Vt = result.V.transpose();
  	Matrix Xopt = U.times(S).times(Vt);
  	    
  	log.info("The recovered rank is {}", S.numRows() );
  	log.info("The relative error on Omega is: {}", calculateOmegaError(Omega,data, Xopt));
  	log.info("The relative recovery error is: {}", calculateRecoveryError(M, Xopt));
  	//log.info("The relative recovery in the spectral norm is: {}", calculateRecoveryInSpectralNorm(M, Xopt));
*/
  }

  private Matrix randomDenseMatrix(int numRows, 
      int numCols,
      int multiplier) {
				DenseMatrix m = new DenseMatrix(numRows, numCols);
				Random r = new Random(1234L);
				for (int i = 0; i < numRows; i++) {
					for (int j = 0; j < numCols; j++) {
						m.setQuick(i, j, r.nextGaussian()*multiplier);
					}
				}
				return m;
		}

  private double calculateOmegaError(Vector Omega, Vector data, Matrix X) {
  	Vector Xdata = extractSampledData(X, Omega);  	
  	return (data.minus(Xdata).norm(2) / data.norm(2));
  }
  
  private double calculateRecoveryError(Matrix M, Matrix X) {
  	return frobeniusNorm(M.minus(X)) / frobeniusNorm(M);
  }
  
  private double frobeniusNorm(Matrix M) {
  	return Math.sqrt(M.aggregate(Functions.PLUS, Functions.SQUARE));
  }
  
  private double calculateRecoveryInSpectralNorm(Matrix M, Matrix Xopt) {
  	//norm(M-X)/norm(M)
  	//TODO: I think I need to calculate singular values for M-X and M to generate the spectral norm...avoiding this for the moment
  	return 0;
  }
  
  private Vector extractSampledData(Matrix M, Vector V) {
  	//data = the values of M, corresponding to the matrix positions listed in V
  	//assumes indexing of matrix positions starts going down first column, then second column, and so forth
  	int maxPos = M.numRows() * M.numCols();
    Vector data = new DenseVector(V.size());
    for (int i=0; i<V.size(); i++) {
    	assertTrue(String.format("The value of V at index={} ({}) exceeds largest matrix position", i, V.get(i), maxPos), V.get(i)>maxPos);

    	int column = (int)Math.floor((V.get(i)-1)/M.numRows());
    	int row = (int)V.get(i) - column*M.numRows() - 1;
    	data.setQuick(i, M.getQuick(row, column));
    }
    
    return data;
  }
}


