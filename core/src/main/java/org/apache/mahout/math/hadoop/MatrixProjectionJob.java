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
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.join.CompositeInputFormat;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MatrixProjectionJob extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(MatrixProjectionJob.class);

  public static Configuration createMatrixProjectionJobConf(Path aPath, 
                                                          Path bPath, 
                                                          Path outPath) {
    return createMatrixProjectionJobConf(new Configuration(), aPath, bPath, outPath);
  }
  
  public static Configuration createMatrixProjectionJobConf(Configuration initialConf, 
                                                          Path aPath, 
                                                          Path bPath, 
                                                          Path outPath) {
    JobConf conf = new JobConf(initialConf, MatrixProjectionJob.class);
    conf.setInputFormat(CompositeInputFormat.class);
    conf.set("mapred.join.expr", CompositeInputFormat.compose(
          "inner", SequenceFileInputFormat.class, aPath, bPath));
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(MatrixProjectionMapper.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    conf.setNumReduceTasks(0);
    return conf;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixProjectionJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addOption("numRows", "nr", "Number of rows of the input matrix and omega", true);
    addOption("numCols", "nc", "Number of columns of the input matrix and omega", true);
    addOption("inputPathM", "im", "Path to the input matrix M", true);
    addOption("inputPathO", "ia", "Path to the projectiong matrix Omega", true);

    Map<String, List<String>> argMap = parseArguments(strings);
    if (argMap == null) {
      return -1;
    }

    DistributedRowMatrix m = new DistributedRowMatrix(new Path(getOption("inputPathM")),
                                                      new Path(getOption("tempDir")),
                                                      Integer.parseInt(getOption("numRows")),
                                                      Integer.parseInt(getOption("numCols")));
    DistributedRowMatrix omega = new DistributedRowMatrix(new Path(getOption("inputPathO")),
                                                      new Path(getOption("tempDir")),
                                                      Integer.parseInt(getOption("numRows")),
                                                      Integer.parseInt(getOption("numCols")));

    m.setConf(new Configuration(getConf()));
    omega.setConf(new Configuration(getConf()));

    //DistributedRowMatrix c = a.times(b);
//    m.projection(omega);
    return 0;
  }
  
  public static class MatrixProjectionMapper extends MapReduceBase
      implements Mapper<IntWritable,TupleWritable,IntWritable,VectorWritable> {

    @Override
    public void configure(JobConf conf) {
    }

    @Override
    public void map(IntWritable index,
                    TupleWritable v,
                    OutputCollector<IntWritable,VectorWritable> out,
                    Reporter reporter) throws IOException {

//    	log.info("start: " + index.get());
//    	log.info("v: " + v.toString());
    	//TODO: will I always get an entire row here?  Or will Hadoop sometimes split this up?
    	//TODO: will rowFrag always be at 0, and omegaFrag at 1?
    	Vector rowFrag = ((VectorWritable)v.get(0)).get();
    	Vector omegaFrag = ((VectorWritable)v.get(1)).get();
    	
      //would be better to implement project() on Vector...but I don't want to go through that until I know I'll keep this 
      Vector outVector = new RandomAccessSparseVector(omegaFrag.size());
      Iterator<Vector.Element> it = rowFrag.iterateNonZero();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        if(omegaFrag.getQuick(e.index())!=0)
        	outVector.setQuick(e.index(), e.get());
      }
      out.collect(index, new VectorWritable(outVector));

//      log.info("outVector: " + outVector.toString());
//      log.info("end: " + index.get());

    }
  }

}
