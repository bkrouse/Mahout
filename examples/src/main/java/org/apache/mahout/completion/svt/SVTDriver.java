/* Licensed to the Apache Software Foundation (ASF) under one or more
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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Vector;

/**
 * Mahout driver for SVTSolver
 * 
 */

public class SVTDriver extends AbstractJob {

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    addOption("stepSize", "ss", "Step size - delta", "0.05"); //no idea the right default
    addOption("tolerance", "err", "Tolerance - epsilon", "0.05"); //no idea the right default
    addOption("threshold", "tao", "Threshold - tao", "0.05"); //no idea the right default
    addOption("increment", "incr", "Increment - l", "1"); //no idea the right default
    addOption("maxIter", "iter", "Maximum iteration count - k_max", "1"); //no idea the right default
    addOption(DefaultOptionCreator.overwriteOption().create());

    Map<String, List<String>> pargs = parseArguments(args);
    if (pargs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));
    double stepSize = Double.parseDouble(getOption("stepSize"));
    double tolerance = Double.parseDouble(getOption("tolerance"));
    double threshold = Double.parseDouble(getOption("threshold"));
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


    SVTSolver solver = new SVTSolver();
    solver.solve(conf,
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
    ToolRunner.run(new SVTDriver(), args);
  }

}
