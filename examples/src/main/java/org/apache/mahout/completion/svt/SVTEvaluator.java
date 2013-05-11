package org.apache.mahout.completion.svt;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;

public class SVTEvaluator extends AbstractJob {
	
		private static String SAMPLED_MATRIX_SUBDIR = "/sampled-m";
	
	  @Override
	  public int run(String[] args) throws Exception {
	  	
	    addInputOption(); //Path to DistribtuedMatrix form of the input matrix
	    addOutputOption();
	    addOption("numPartitions", "np", "Number of partitions to split the input matrix into");
	    addOption("numRows", "nr", "Number of rows of the input matrix");
	    addOption("numCols", "nc", "Number of columns of the input matrix");
	    addOption("omega", "o", "Path to the vector OmegaRowBasedSorted as a SequenceFile<IntWritable,VectorWritable>, which describes the sampled values from input matrix (assume sorted and row-based indices!)");

	    Map<String, List<String>> pargs = parseArguments(args);
	    if (pargs == null) {
	      return -1;
	    }
	    
	    Path inputPath = getInputPath();
	    Path outputPath = getOutputPath();
	    Path tempPath = getTempPath();
	    int numPartitions = Integer.parseInt(getOption("numPartitions"));
	    int numRows = Integer.parseInt(getOption("numRows"));
	    int numCols = Integer.parseInt(getOption("numCols"));
	    Path omegaPath = new Path(getOption("omega"));

	    Configuration conf = getConf();
	    if (conf == null) {
	      throw new IOException("No Hadoop configuration present");
	    }
	    

	    Vector omega;
      SequenceFileIterator<IntWritable,VectorWritable> seqIterator =
        new SequenceFileIterator<IntWritable,VectorWritable>(omegaPath, true, getConf());      
      if(seqIterator.hasNext()) {
      	Pair<IntWritable,VectorWritable> record = seqIterator.next();
      	omega = record.getSecond().get();
      }
      else
      	throw new IllegalStateException("Didn't find omega vector as expected.");
	    
            
      //Read in the matrix m -- and based on Omega, write out the sampled matrix
    	FileSystem fs = tempPath.getFileSystem(getConf());
    	Path sampledMatrixPath = new Path(tempPath.toString().concat(SAMPLED_MATRIX_SUBDIR));
    	fs.delete(sampledMatrixPath);
      SequenceFile.Writer writer = null;

      try {
        writer = new SequenceFile.Writer(fs,
            conf,
            sampledMatrixPath,
            IntWritable.class,
            VectorWritable.class);
      	
      	SequenceFileIterator<IntWritable,VectorWritable> mIterator =
          new SequenceFileIterator<IntWritable,VectorWritable>(inputPath, true, getConf());
        
      	int mRowId=0, omegaIdx=0, mPos=1;
        while(mIterator.hasNext()) {
        	Pair<IntWritable,VectorWritable> record = mIterator.next();
        	VectorWritable vw = record.getSecond();      	
        	Vector mRow = vw.get();
   
        	for(int rowIdx=0; rowIdx < numCols; rowIdx++, mPos++)
        	{
        		if (omegaIdx >= omega.size())
        			mRow.setQuick(rowIdx, 0);
        		else if(omega.getQuick(omegaIdx) > mPos) 
        			mRow.setQuick(rowIdx, 0);
        		else
        			omegaIdx++;
        	}
        	      	
        	//TODO: consider pros/cons of RandomAccessSparseVector vs SequentialAccessSparseVector inside SVT algorithm
        	RandomAccessSparseVector mSparseRow = new RandomAccessSparseVector(mRow);

        	writer.append(new IntWritable(mRowId++), new VectorWritable(mSparseRow));     
        }

      } finally {
        Closeables.closeQuietly(writer);
      }
      

      //repartition
	  DistributedRowMatrix matrix = new DistributedRowMatrix(sampledMatrixPath, tempPath, 0, 0);
	  matrix.setConf(conf);
	  DistributedRowMatrix repartitionedMatrix = matrix.repartitionSequenceFile(numPartitions);

      

      //call SVTSolver
      SVTSolver svtSolver = new SVTSolver();
      svtSolver.solve(conf,
          repartitionedMatrix.getRowPath(),
          outputPath,
          new Path(tempPath, "svt-working"),
          numRows,
          numCols,
          true);
      
      
      
      
      //evaluate results from SVTSolver
      
      
      //clean up temp dirs?
      
	    return 0;
	  }

	  public static void main(String[] args) throws Exception {
	    ToolRunner.run(new SVTEvaluator(), args);
	  }

}
