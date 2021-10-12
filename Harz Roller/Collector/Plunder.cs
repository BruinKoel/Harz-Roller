using System;
using System.Collections.Generic;
using System.Net;
using System.IO;
using System.Linq;
using System.IO.Compression;

namespace Harz_Roller.Plunder
{
    class Plunder
    {
        private WebClient client;
        public string dataDir;
        private Dictionary<string, string[]> pairs;
        const string webdir = "https://data.binance.vision/data/spot/monthly/klines";

        public Plunder(string dataDir = "")
        {
            if (dataDir == "") dataDir = Environment.CurrentDirectory + "\\BinanceHistoricalData";
            this.dataDir = dataDir;

            client = new WebClient();

            pairs = new Dictionary<string, string[]>();



            if (!Directory.Exists(dataDir)) Directory.CreateDirectory(dataDir);
            foreach (string pair in Directory.GetDirectories(dataDir))
            {
                pairs.Add(Path.GetFileName(pair),
                    Directory.GetDirectories(pair).Select(x => Path.GetFileName(x)).ToArray<string>());
            }

        }

        public void collect()
        {
            foreach (KeyValuePair<string, string[]> pair in pairs)
            {
                foreach (string timeInterval in pair.Value)
                {
                    DateTime beginning = new DateTime(2017, 1, 1);
                    Console.WriteLine("Searching for {0} {1} data!", pair.Key, timeInterval);
                    while (beginning < DateTime.Today)
                    {
                        string link = webdir +
                                "/" + pair.Key +
                                "/" + timeInterval +
                                "/" + pair.Key +
                                "-" + timeInterval +
                                "-" + beginning.ToString("yyyy-MM") + ".zip";
                        string file = dataDir +
                                "\\" + pair.Key +
                                "\\" + timeInterval +
                                "\\" + beginning.ToString("yyyy-MM") + ".zip";
                        if (File.Exists(file)) {
                            unZip(file);
                            beginning = beginning.AddMonths(1); 
                            continue; }

                        try
                        {
                            client.DownloadFile(link, file);
                            unZip(file);
                            Console.WriteLine("Written: {0}", file);
                        }
                        catch (Exception E)
                        {
                            Console.WriteLine("no file for {0}", beginning.ToString("yyyy-MM"));
                        }
                        
                        beginning = beginning.AddMonths(1);
                    }
                }
            }
        }

        public void unZip(string file)
        {

            ZipFile.ExtractToDirectory(file, Path.GetDirectoryName(file)+"\\csv");
            
        }
    }
}
