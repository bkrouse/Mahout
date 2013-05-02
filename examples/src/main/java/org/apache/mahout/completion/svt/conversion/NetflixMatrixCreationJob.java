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

package org.apache.mahout.completion.svt.conversion;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
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
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.join.CompositeInputFormat;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class NetflixMatrixCreationJob extends AbstractJob {
	
  public static Configuration createNetflixMatrixCreationJobConf(Configuration initialConf,
  																												Path ratingsPath,
                                                          Path outPath) throws IOException {
    JobConf conf = new JobConf(initialConf, NetflixMatrixCreationJob.class);
    conf.setJobName("NetflixMatrixCreationJob: " + ratingsPath + " -> " + outPath);
    FileSystem fs = FileSystem.get(ratingsPath.toUri(), conf);
    ratingsPath = fs.makeQualified(ratingsPath);
    outPath = fs.makeQualified(outPath);
    FileInputFormat.setInputPaths(conf, ratingsPath);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(NetflixMatrixCreationMapper.class);
    conf.setReducerClass(NetflixMatrixCreationReducer.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);

    return conf;
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("Usage: NetflixMatrixCreationJob /path/to/ratings.seq /path/to/output");
      return;
    }

    Path ratingsPath = new Path(args[0]);
    Path outputPath = new Path(args[1]);
    Path matrixOutputPath = new Path(outputPath, "ratingsMatrix");

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);
    
    fs.delete(matrixOutputPath);

    
    //fire off a map reduce job to create the matrix
    Configuration myconf = NetflixMatrixCreationJob.createNetflixMatrixCreationJobConf(conf, 
																																					    		ratingsPath, 
																																					    		matrixOutputPath);
    RunningJob job = JobClient.runJob(new JobConf(myconf));
    job.waitForCompletion();

  
  }

  @Override
  public int run(String[] strings) throws Exception {
  	throw new Exception("not implemented");
  }

  public static class NetflixMatrixCreationMapper extends MapReduceBase
      implements Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private static final Pattern SEPARATOR = Pattern.compile(",");

    @Override
    public void map(IntWritable index,
                    VectorWritable vw,
                    OutputCollector<IntWritable,VectorWritable> out,
                    Reporter reporter) throws IOException {

    	Vector v = vw.get();
    	
      Integer rowID = (int) v.get(0);
      Integer movieID = (int) v.get(1);
      Integer rating = (int) v.get(2);
      
      Vector outputV = new DenseVector(2);
      outputV.set(0,movieID);
      outputV.set(1, rating);
    	
    	out.collect(new IntWritable(rowID), new VectorWritable(outputV));
    }
  }

  public static class NetflixMatrixCreationReducer extends MapReduceBase
      implements Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {

  	
  	private static final int NUM_MOVIES = 17770;
  	
    @Override
    public void reduce(IntWritable rowID,
                       Iterator<VectorWritable> it,
                       OutputCollector<IntWritable,VectorWritable> out,
                       Reporter reporter) throws IOException {
      if (!it.hasNext()) {
        return;
      }
      
      Vector accumulator = new RandomAccessSparseVector(NUM_MOVIES); 
      while (it.hasNext()) {
        Vector v = it.next().get();
        int movieID = (int) v.get(0);
        int rating = (int) v.get(1);
        accumulator.set(movieID-1, rating);
      }
      out.collect(rowID, new VectorWritable(new SequentialAccessSparseVector(accumulator)));
    }
  }
}


