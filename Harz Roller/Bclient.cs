using System;
using System.Collections.Generic;
using System.Text;
using Binance.Net;

namespace Harz_Roller
{
    class Bclient
    {
        protected string apiKey;
        protected BinanceClient Binance;
        public Bclient(string apiKey, string apiSecret)
        {
            this.apiKey = apiKey;
            this.Binance = new BinanceClient();
            Binance.SetApiCredentials(apiKey, apiSecret);
        }
    }
}
