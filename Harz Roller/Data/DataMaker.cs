using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras;
//using NumSharp;
using Deedle;
using System.IO;
using static Tensorflow.Binding;
using System.Linq;

using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;




namespace Harz_Roller.Data
{
    class DataMaker
    {
        public struct TrainingSet
        {
            public float[,,] inputField;
            public float[] outputField;
        }

        public string dataDir;
        private Frame<int, string> df;
        public string dataFile;
        public string convertedDataFile;
        const string baseColumns = "Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore,MinuteOfYear";
        const string generatedColumns = "SOpen,SHigh,SLow,SClose,DVolume";
        //                               %^   %^   %^  %^    deviation

        NDArray x_train, y_train, x_test, y_test;

        public DataMaker(string dataDir)
        {
            this.dataDir = dataDir;
            //pd = new PandasNet.Pandas();
            dataFile = dataDir + "\\DataStitch.csv";
            convertedDataFile = dataDir + "\\DataStitchConverted.csv"

            ;
            if (!File.Exists(convertedDataFile)) convertedDataFile = dataFile;


        }
        //private Tensorflow.data   Tensorflow.Keras.Datasets.Mnist
        

        
        
        public TrainingSet generateTrainingSet( int size, Random random)
        {
            int inputSize = 2048;
            int outputSize = 1;//CAN'T BE ANYTHING ELSE
            int outputRange = 512;//timeframe it should predict ahead in terms of datapoints
            
            int index = 0;


            int[] columnOrder = { 1, 2, 3, 4, 5, 12 };//which orders should be take for the inputfield

            float[,,] inputField = new float[size, inputSize, 6];
            float[] outputField = new float[size];

            string[] data = File.ReadAllLines(convertedDataFile);
            string line = "";
            for (int i = 0; i < size; i++)
            {
                if ((i % 256) == 0) Console.WriteLine("{2}% Batched {0} out of {1}", i, size, i*100/size);
                index = random.Next(data.Length - inputSize - outputSize - outputRange);
                for (int o = 0; o < inputSize; o++)
                {
                    for (int u = 0; u < columnOrder.Length; u++)
                    {
                        inputField[i, o, u] = float.Parse(data[index + o + 1].Split(',')[columnOrder[u]]);
                    }
                    
                }
                outputField[i] = 1;
                for (int o = 0; o < outputRange; o++)
                {
                    outputField[i] *=  (1 -float.Parse(data[index + o + 1 + inputSize].Split(',')[13]));
                }
                outputField[i] = 1 -outputField[i];

            }
            Console.WriteLine("PROCESSED OK! \r\n");

            return new TrainingSet{ outputField = outputField, inputField = inputField };
        
        }
        /// <summary>
        /// calculates the mean of a set
        /// </summary>
        /// <param name="set">input set</param>
        /// <returns></returns>
        private float mean(float[] set)
        {
            float mean = 0;
            foreach(float val in set)
            {
                mean += val;
            }
            return mean / set.Length;
        }
        
        /// <summary>
        /// Returns the Highbit for change*100000000
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        private int hibit( float change)
        {
            int n = (int)(change * 1000000);

            n |= (n >> 1);
            n |= (n >> 2);
            n |= (n >> 4);
            n |= (n >> 8);
            n |= (n >> 16);
            return n - (n >> 1);
        }
        
        public void convertCSV()
        {
            DateTime starttime = DateTime.Now;
            string firstLine = "Open,High,Low,Close,Volume,MOY,15High,30High,60High,4hHigh,8High,15Low,30Low,60Low,4hLow,8Low,";
            int[] timeFrames = { 15, 30, 60, 240, 480 };

            int linecounter = 0;
            Console.WriteLine("starting Read");
            
            convertedDataFile = dataDir + "\\DataStitchConverted.csv";
            if (File.Exists(convertedDataFile)) File.Delete(convertedDataFile);

            //reads every line, splits the lines into words which are converted to floats
            float[][] bigCSV = File.ReadAllLines(dataFile)
                .Skip(1).Select(x => x.Split(',').Select(x => float.Parse(x)).ToArray()).ToArray();
            Console.WriteLine("{0} Read complete! Processing", DateTime.Now.Subtract(starttime).TotalSeconds);
            Calculator calc = new Calculator(bigCSV);

            bool[][] netPeakOutputs = new bool[bigCSV.GetLength(0)][];
            bool[][] netValleyOutputs = new bool[bigCSV.GetLength(0)][];

            for (int i = 0; i < bigCSV.GetLength(0) - timeFrames[timeFrames.Length-1]; i++)
            {
                netPeakOutputs[i] = calc.isPeak(timeFrames,4);
            }
            calc.setPointer();

            for (int i = 0; i < bigCSV.GetLength(0) - timeFrames[timeFrames.Length - 1]; i++)
            {
                netValleyOutputs[i] = calc.isValley(timeFrames, 4);
            }
            

            calc.makeDirectional(1);
            calc.makeDirectional(2);
            calc.makeDirectional(3);
            calc.makeDirectional(4);

            //bigCSV = calc.getDataPlane();


            
            Console.WriteLine("{0} Processing complete! building CSV ", DateTime.Now.Subtract(starttime).TotalSeconds);

            StreamWriter fS = new StreamWriter(File.OpenWrite(convertedDataFile));
            fS.AutoFlush = false;
            fS.WriteLine(firstLine);
            for (int i = 0; i < bigCSV.GetLength(0) - timeFrames[timeFrames.Length - 1]; i++)
            {
                fS.WriteLine("{0},{1},{2},{3}",
                    string.Join(',', bigCSV[i][1..6]),
                    bigCSV[i][12].ToString(),
                    string.Join(',',netPeakOutputs[i].Select(x => Convert.ToInt32(x))),
                    string.Join(',', netValleyOutputs[i].Select(x => Convert.ToInt32(x)))
                    );
            }
            Console.WriteLine("{0} Build complete! Writing to Disk", DateTime.Now.Subtract(starttime).TotalSeconds);
            fS.Flush();
            fS.Close();
            
            Console.WriteLine("{0} seconds and Produced {1} lines", DateTime.Now.Subtract(starttime).TotalSeconds, bigCSV.Length);
            Console.WriteLine("PROCESSED OK! Trimmed {0} Kilobytes in the process", new FileInfo(dataFile).Length / 1024 - new FileInfo(convertedDataFile).Length / 1024);
        }
/// <summary>
/// stitches al the csv data into one big csv file for ease of use
/// </summary>
/// <param name="qualityVolume"> the minimum volume for when to start using the data</param>
        public void stitchCSV( float qualityVolume = 5) 
        {
            
            if (File.Exists(dataFile)) File.Delete(dataFile);
            List<string> dataCSV = new List<string>();
            dataCSV.Add(baseColumns);

            bool qualitySet = false;
            int lastMinute = 0;
            if (qualityVolume == 0) qualitySet = true;

            foreach (string file in Directory.GetFiles(dataDir))
            {
                
                
                string fileName = Path.GetFileNameWithoutExtension(file);
                if (Path.GetExtension(file) != ".csv" || fileName.StartsWith("DataStitch")) continue;

                int minute = DateTime.Parse(fileName.Substring(fileName.Length - 7)).DayOfYear * 24 * 60;
                Console.WriteLine("minute alignment error: {0} for:", minute - lastMinute);

                

                ///adds the minute times to each

                foreach (string line in File.ReadAllLines(file))
                {
                    if (!qualitySet && float.Parse(line.Split(',')[5]) < qualityVolume) 
                    {
                        continue;
                    }
                    qualitySet = true;
                    dataCSV.Add(line +","+ minute++.ToString());
                }
                lastMinute = minute;
                Console.WriteLine(file + " PROCESSED OK! \r\n");

            }
            
            File.WriteAllLines(dataFile, dataCSV);
            Console.WriteLine("Final output: {0} ", dataFile);
            Console.WriteLine(" {0} rows  {1} Collumns {2} Kilobytes", dataCSV.Count, dataCSV[0].Split(',').Length, new FileInfo(dataFile).Length/1024);
            
        }
    }
}
