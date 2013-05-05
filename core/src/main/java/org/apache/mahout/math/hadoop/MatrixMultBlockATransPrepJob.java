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

package org.apache.mahout.math.hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.join.CompositeInputFormat;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class MatrixMultBlockATransPrepJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(MatrixMultBlockATransPrepJob.class);

  private static final String NUM_BLOCKS = "MatrixMultBlockATransPrep.numBlocks";

  public static Configuration createMatrixMultBlockATransJobConf(Path aPath, 
                                                          Path outPath, 
                                                          int numBlocks) {
    return createMatrixMultBlockATransPrepJobConf(new Configuration(), aPath, outPath, numBlocks);
  }
  
  public static Configuration createMatrixMultBlockATransPrepJobConf(Configuration initialConf, 
                                                          Path aPath, 
                                                          Path outPath, 
                                                          int numBlocks) {
    JobConf conf = new JobConf(initialConf, MatrixMultBlockATransPrepJob.class);
    conf.setJobName("MatrixMultBlockATransPrepJob: " + aPath + " (numBlocks=" + numBlocks + ") -> " + outPath);
    conf.setInputFormat(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(conf, aPath);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setInt(NUM_BLOCKS, numBlocks);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(MatrixMultBlockATransPrepMapper.class);
    conf.setCombinerClass(IdentityReducer.class);
    conf.setReducerClass(IdentityReducer.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);

    return conf;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixMultBlockATransPrepJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addOption("inputPathATrans", "ia", "Path to the ATrans matrix", true);
    addOption("numBlocks", "n", "Number of blocks -- i.e. will split each ATrans row into this number of rows", true);

    addOption("outputPath", "op", "Path to the output matrix", false);

    Map<String, List<String>> argMap = parseArguments(strings);
    if (argMap == null) {
      return -1;
    }

    
    Configuration initialConf = getConf() == null ? new Configuration() : getConf();


    Configuration conf = MatrixMultBlockATransPrepJob.createMatrixMultBlockATransPrepJobConf(initialConf,
    																												new Path(getOption("inputPathATrans")),
    																												new Path(getOption("outputPath")),
                                                            Integer.parseInt(getOption("numBlocks")));
    RunningJob job = JobClient.runJob(new JobConf(conf));
    job.waitForCompletion();

    return 0;
  }

  public static class MatrixMultBlockATransPrepMapper extends MapReduceBase
      implements Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int numBlocks;
    private final IntWritable row = new IntWritable();

    @Override
    public void configure(JobConf conf) {
      numBlocks = conf.getInt(NUM_BLOCKS, Integer.MAX_VALUE);
    }

    @Override
    public void map(IntWritable index,
                    VectorWritable vw,
                    OutputCollector<IntWritable,VectorWritable> out,
                    Reporter reporter) throws IOException {

    	Vector v = vw.get();

    	int blockSize = (int) Math.floor(v.size() / numBlocks);

    	//currently only support this if blockSize goes evenly into v.size()
    	if (blockSize * numBlocks != v.size())
    		throw new IOException("numBlocks must go evenly into the columns of ATrans");

    	int startIdx = 0;
    	Vector outVector; 
  		VectorWritable outVectorWritable = new VectorWritable();
    	for (int i=0; i < numBlocks; i++)
  		{
  			outVector = new SequentialAccessSparseVector(blockSize);
  			Vector vectorBlock = v.viewPart(startIdx, blockSize);
  			Iterator<Vector.Element> it = vectorBlock.iterateNonZero();
  			while (it.hasNext()) {
  				Vector.Element e = it.next();
  				outVector.setQuick(e.index(), e.get());
  			}
  			
  			row.set(index.get()*numBlocks + i);
  			outVectorWritable.set(outVector);
  			out.collect(row, outVectorWritable);
    		startIdx += blockSize;
  		}
//      Vector outVector = new SequentialAccessSparseVector(blockSize);
//
//      int lastBlockIndex = 0, rowIdx = 0, colIdx = 0;
//      VectorWritable outVW = new VectorWritable();
//      Vector.Element e = null;      
//      Iterator<Vector.Element> it = v.iterateNonZero();
//      while (it.hasNext()) {
//        e = it.next();
//        
//        if(e.index()-lastBlockIndex >= blockSize) {
//        	//collect previous row
//        	outVW.set(outVector);
//        	rowIdx = (int) Math.floor((e.index() - 1) / blockSize);
//        	row.set(index.get()*numBlocks + rowIdx);
//        	out.collect(row, outVW);
//
//        	//create new outVector for current element
//        	outVector = new SequentialAccessSparseVector(blockSize);
//          lastBlockIndex = e.index();
//        }
//        
//        colIdx = e.index() % blockSize;
//        outVector.set(colIdx, e.get());
//      }
//      
//      //assuming we had something, lets collect the last row
//      if(e!=null) {
//	    	outVW.set(outVector);
//      	rowIdx = (int) Math.floor(e.index() / blockSize);
//      	row.set(index.get()*numBlocks + rowIdx);
//	    	out.collect(row, outVW);
//      }
    }
  }
}
