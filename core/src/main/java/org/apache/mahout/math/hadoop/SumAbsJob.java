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
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import com.google.common.io.Closeables;

/**
 * SumAbsJob is a job for calculating the sum of the absolute values across each column of a
 * DistributedRowMatrix. 
 */
public final class SumAbsJob extends AbstractJob {

  private static String VECTOR_CLASS = "org.apache.mahout.math.SequentialAccessSparseVector";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SumAbsJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));

    DistributedRowMatrix matrix = new DistributedRowMatrix(getInputPath(), getTempPath(), numRows, numCols, true);
    matrix.setConf(new Configuration(getConf()));
    double norm2est = matrix.norm2est(1);

    return 0;
  }
	
  private SumAbsJob() {
  }

  /**
   * Job for calculating the sum of the absolute values across each column of a DistributedRowMatrix
   *
   * @param initialConf
   * @param inputPath
   *          path to DistributedRowMatrix input
   * @param outputVectorTmpPath
   *          path for temporary files created during job
   * @return Vector containing column-wise mean of DistributedRowMatrix
   */
  public static Vector run(Configuration initialConf,
                           Path inputPath,
                           Path outputVectorTmpPath) throws IOException {

    try {
      Job job = new Job(initialConf, "SumAbsJob");
      job.setJarByClass(SumAbsJob.class);

      FileOutputFormat.setOutputPath(job, outputVectorTmpPath);
      
      outputVectorTmpPath.getFileSystem(job.getConfiguration())
                         .delete(outputVectorTmpPath, true);
      job.setNumReduceTasks(1);
      FileOutputFormat.setOutputPath(job, outputVectorTmpPath);
      FileInputFormat.addInputPath(job, inputPath);
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      FileOutputFormat.setOutputPath(job, outputVectorTmpPath);

      job.setMapperClass(SumAbsMapper.class);
      job.setReducerClass(SumAbsReducer.class);
      job.setMapOutputKeyClass(NullWritable.class);
      job.setMapOutputValueClass(VectorWritable.class);
      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(VectorWritable.class);
      job.submit();
      job.waitForCompletion(true);

      Path tmpFile = new Path(outputVectorTmpPath, "part-r-00000");
      SequenceFileValueIterator<VectorWritable> iterator =
        new SequenceFileValueIterator<VectorWritable>(tmpFile, true, initialConf);
      try {
        if (iterator.hasNext()) {
          return iterator.next().get();
        } else {
          return (Vector) new SequentialAccessSparseVector(0);
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


  public static class SumAbsMapper extends
      Mapper<Writable, VectorWritable, NullWritable, VectorWritable> {

    private Vector runningSum;

    /**
     * The mapper computes a running sum of the vectors the task has seen.
		 * Nothing is written at this stage
     */
    @Override
    public void map(Writable r, VectorWritable v, Context context)
      throws IOException {
      if (runningSum == null) {
          /*
           * If this is the first vector the mapper has seen, instantiate a new
           * vector using the parameter VECTOR_CLASS
           */
        runningSum = ClassUtils.instantiateAs(SumAbsJob.VECTOR_CLASS,
                                              Vector.class,
                                              new Class<?>[] { int.class },
                                              new Object[] { v.get().size()});
        runningSum.assign(v.get()).assign(Functions.ABS);
      } else {
        runningSum.assign(v.get(), Functions.PLUS_ABS);
      }
    }

    /**
     * The column-wise sum is written at the cleanup stage. A single reducer is
     * forced so null can be used for the key
     */
    @Override
    public void cleanup(Context context) throws InterruptedException,
      IOException {
      if (runningSum != null) {
        context.write(NullWritable.get(), new VectorWritable(runningSum));
      }
    }

  }

  /**
   * The reducer adds the partial column-wise sums from each of the mappers to
   * compute the total column-wise sum. 
   */
  public static class SumAbsReducer extends
      Reducer<NullWritable, VectorWritable, IntWritable, VectorWritable> {

    private static final IntWritable ONE = new IntWritable(1);

    Vector outputVector;
    VectorWritable outputVectorWritable = new VectorWritable();

    @Override
    public void reduce(NullWritable n,
                       Iterable<VectorWritable> vectors,
                       Context context) throws IOException, InterruptedException {

      /**
       * Add together partial column-wise sums from mappers
       */
      for (VectorWritable v : vectors) {
        if (outputVector == null) {
          outputVector = v.get();
        } else {
          outputVector.assign(v.get(), Functions.PLUS);
        }
      }

      /**
       * Write out results
       */
      if (outputVector != null) {
        outputVectorWritable.set(outputVector);
        context.write(ONE, outputVectorWritable);
      } else {
        Vector emptyVector = ClassUtils.instantiateAs(SumAbsJob.VECTOR_CLASS,
                                                      Vector.class,
                                                      new Class<?>[] { int.class },
                                                      new Object[] { 0 });
        context.write(ONE, new VectorWritable(emptyVector));
      }
    }
  }

}
