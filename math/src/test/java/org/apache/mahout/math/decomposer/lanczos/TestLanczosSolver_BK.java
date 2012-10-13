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

package org.apache.mahout.math.decomposer.lanczos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Iterator;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.solver.EigenDecomposition;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TestLanczosSolver_BK extends SolverTest {
  private static final Logger log = LoggerFactory.getLogger(TestLanczosSolver.class);

  private static final double ERROR_TOLERANCE = 0.05;

  
  //@Test
  public void testOutputResults() throws Exception {
    int size = 100;
    Matrix m = randomHierarchicalSymmetricMatrix(size);

    Vector initialVector = new DenseVector(size);
    initialVector.assign(1.0 / Math.sqrt(size));
    LanczosSolver solver = new LanczosSolver();
    int desiredRank = 80;
    LanczosState state = new LanczosState(m, desiredRank, initialVector);
    // set initial vector?
    solver.solve(state, desiredRank, true);

		EigenDecomposition decomposition = new EigenDecomposition(m);
		Vector eigenvalues = decomposition.getRealEigenvalues();
     
    File file1 = new File("/Users/bkrouse/Documents/eclipseworkspaces/hadoop/mahout-testing/M.dat");
    FileWriter fw = new FileWriter(file1);
    BufferedWriter bw = new BufferedWriter(fw);
    for(int i=0; i<m.numRows(); i++) {
    	for(int j=0; j<m.numCols(); j++) {
    		bw.write(String.valueOf(m.get(i, j)));
        	if(j<(m.numCols()-1)) bw.write(" ");
    	}
    	bw.newLine();
    }
    bw.close();
    fw.close();
    
    //TODO: it seems that the iterator can mess up the ordering...switch to for loop instead
    File file2 = new File("/Users/bkrouse/Documents/eclipseworkspaces/hadoop/mahout-testing/Lanczos_EigenVectors.dat");
    fw = new FileWriter(file2);
    bw = new BufferedWriter(fw);
    Iterator<Vector> itr = state.singularVectors.values().iterator(); 
    while(itr.hasNext()) {
        Vector v = itr.next(); 
        for(int j=0; j<v.size(); j++) {
        	bw.write(String.valueOf(v.get(j)));
        	if(j<(v.size()-1)) bw.write(" ");
        }
        bw.newLine();
    }
    bw.close();
    fw.close();

    
    //TODO: it seems that the iterator can mess up the ordering...switch to for loop instead
    File file3 = new File("/Users/bkrouse/Documents/eclipseworkspaces/hadoop/mahout-testing/Lanczos_EigenValues.dat");
    fw = new FileWriter(file3);
    bw = new BufferedWriter(fw);
    Iterator<Double> itr2 = state.singularValues.values().iterator(); 
    while(itr2.hasNext()) {
        Double d = itr2.next(); 
    	bw.write(String.valueOf(d));
        bw.newLine();
    }
    bw.close();
    fw.close();
  }
  
  @Test
  public void testEigenvalueCheck() throws Exception {
    int numRows = 800;
    int numColumns = 500;
    Matrix corpus = randomHierarchicalMatrix(numRows, numColumns, false);

    Vector initialVector = new DenseVector(numColumns);
    initialVector.assign(1.0 / Math.sqrt(numColumns));
    int rank = 80;
    LanczosState state = new LanczosState(corpus, rank, initialVector);
	  
    LanczosSolver solver = new LanczosSolver();
	solver.solve(state, rank, false);
    
    assertEigenvalues(state, rank, false, ERROR_TOLERANCE);
  }

  
//  @Test
  public void testEigenvalueCheckSymmetric() throws Exception {
    int size = 100;
    Matrix m = randomHierarchicalSymmetricMatrix(size);

    Vector initialVector = new DenseVector(size);
    initialVector.assign(1.0 / Math.sqrt(size));
    LanczosSolver solver = new LanczosSolver();
    int desiredRank = 80;
    LanczosState state = new LanczosState(m, desiredRank, initialVector);
    // set initial vector?
    solver.solve(state, desiredRank, true);
    
    assertEigenvalues(state, desiredRank, true, ERROR_TOLERANCE);
  }


//  @Test
  public void testLanczosSolver() throws Exception {
    int numRows = 800;
    int numColumns = 500;
    Matrix corpus = randomHierarchicalMatrix(numRows, numColumns, false);
    Vector initialVector = new DenseVector(numColumns);
    initialVector.assign(1.0 / Math.sqrt(numColumns));
    int rank = 50;
    LanczosState state = new LanczosState(corpus, rank, initialVector);
    long time = timeLanczos(corpus, state, rank, false);
    assertTrue("Lanczos taking too long!  Are you in the debugger? :)", time < 10000);
    assertOrthonormal(state);
    for(int i = 0; i < rank/2; i++) {
      assertEigen(i, state.getRightSingularVector(i), corpus, ERROR_TOLERANCE, false);
    }
    //assertEigen(eigens, corpus, rank / 2, ERROR_TOLERANCE, false);
  }

//  @Test
  public void testLanczosSolverSymmetric() throws Exception {
    int numCols = 500;
    Matrix corpus = randomHierarchicalSymmetricMatrix(numCols);
    Vector initialVector = new DenseVector(numCols);
    initialVector.assign(1.0 / Math.sqrt(numCols));
    int rank = 30;
    LanczosState state = new LanczosState(corpus, rank, initialVector);
    long time = timeLanczos(corpus, state, rank, true);
    assertTrue("Lanczos taking too long!  Are you in the debugger? :)", time < 10000);
    //assertOrthonormal(state);
    //assertEigen(state, rank / 2, ERROR_TOLERANCE, true);
  }

  public static long timeLanczos(Matrix corpus, LanczosState state, int rank, boolean symmetric) {
    long start = System.currentTimeMillis();

    LanczosSolver solver = new LanczosSolver();
    // initialize!
    solver.solve(state, rank, symmetric);
    
    long end = System.currentTimeMillis();
    return end - start;
  }

}
