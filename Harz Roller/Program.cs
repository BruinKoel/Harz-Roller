using System;

namespace Harz_Roller
{
    class Program
    {
        static void Main(string[] args)
        {
            /*Collector.Collector collector = new Collector.Collector();
            collector.collect();
            Bclient client = new Bclient(args[0], args[1]);
            Console.WriteLine("Hello World!");*/
            Data.DataMaker dataMaker = new Data.DataMaker(@"D:\Daan Games\Visual projects\Harz Roller\Harz Roller\bin\Debug\netcoreapp3.1\BinanceHistoricalData\ETHUSDT\1m\csv");
            //dataMaker.stitchCSV();
            Models.ModelMaker model = new Models.ModelMaker(dataMaker);
            model.PrepareData();
            model.Run();
            
            
        }
    }
}
