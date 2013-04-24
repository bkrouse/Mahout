/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to You under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.apache.mahout.math.hadoop;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import com.google.common.io.Closeables;

/**
 * MatrixFrobeniusNormJob is a job for calculating the frobenius norm of a
 * DistributedRowMatrix. This job can be accessed using
 * DistributedRowMatrix.frobeniusNorm()
 */
public final class MatrixFrobeniusNormJob  {

  private MatrixFrobeniusNormJob() {
  }

  public static Double run(Configuration conf,
                           Path inputPath, Path outputPath) throws IOException {

  	try
  	{
      Job job = new Job(conf, "MatrixFrobeniusNormJob");
      job.setJarByClass(MatrixFrobeniusNormJob.class);

      
      outputPath.getFileSystem(job.getConfiguration())
                         .delete(outputPath, true);
      
      job.setInputFormatClass(SequenceFileInputFormat.class);
      FileInputFormat.setInputPaths(job, inputPath);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      FileOutputFormat.setOutputPath(job, outputPath);

      job.setMapperClass(MatrixFrobeniusNormMapper.class);
      job.setCombinerClass(MatrixFrobeniusNormCombiner.class);
      job.setReducerClass(MatrixFrobeniusNormReducer.class);
      job.setNumReduceTasks(1);
      job.setMapOutputKeyClass(NullWritable.class);
      job.setMapOutputValueClass(DoubleWritable.class);
      job.setOutputKeyClass(NullWritable.class);
      job.setOutputValueClass(DoubleWritable.class);
      job.submit();
      job.waitForCompletion(true);

      Path tmpFile = new Path(outputPath, "part-r-00000");
      SequenceFileValueIterator<DoubleWritable> iterator =
        new SequenceFileValueIterator<DoubleWritable>(tmpFile, true, conf);
      try {
        if (iterator.hasNext()) {
          return iterator.next().get();
        } else {
          return 0.0;
        }
      } finally {
        Closeables.closeQuietly(iterator);
      }
    } catch (Throwable thr) {
      if (thr instanceof IOException) {
        throw (IOException) thr;
      } else {
        throw new IOException(thr);
      }
    }

  }


 
  /**
   * Mapper for calculation of frobenius norm.
   */
  public static class MatrixFrobeniusNormMapper extends
      Mapper<Writable, VectorWritable, NullWritable, DoubleWritable> {

    private double sum;


    /**
     * The mapper computes a sum of the squares of each element
     */
    @Override
    public void map(Writable r, VectorWritable v, Context context)
      throws IOException, InterruptedException {
    	
    	sum = v.get().getLengthSquared();
      context.write(NullWritable.get(), new DoubleWritable(sum));
    }

  }
  	
  	
 

  /**
   * The combiner just spits out a single sum from mappers
   */
  public static class MatrixFrobeniusNormCombiner extends
      Reducer<NullWritable, DoubleWritable, NullWritable, DoubleWritable> {

    @Override
    public void reduce(NullWritable n,
                       Iterable<DoubleWritable> sums,
                       Context context) throws IOException, InterruptedException {

    	double runningSumSquares = 0;
    	
      for (DoubleWritable sum : sums) {
      	runningSumSquares += sum.get();
      }
      
      context.write(NullWritable.get(), new DoubleWritable(runningSumSquares));

    }
  }

  
  /**
   * The reducer returns the square root of the sums
   */
  public static class MatrixFrobeniusNormReducer extends
      Reducer<NullWritable, DoubleWritable, NullWritable, DoubleWritable> {

    @Override
    public void reduce(NullWritable n,
                       Iterable<DoubleWritable> sums,
                       Context context) throws IOException, InterruptedException {

    	double runningSumSquares = 0;
    	
      for (DoubleWritable sum : sums) {
      	runningSumSquares += sum.get();
      }
      
      context.write(NullWritable.get(), new DoubleWritable(Math.sqrt(runningSumSquares)));

    }
  }

  
}
