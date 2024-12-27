using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SocketIOClient;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System.Linq;

namespace IoTDataProcess
{
    public class socketClientService
    {
        public SocketIOClient.SocketIO client; 
        public socketClientService()
        {
            client = new SocketIOClient.SocketIO("http://localhost:8080");
        }
        public async Task _SocketClient()
        {
            Console.WriteLine("start socket");
            client.OnConnected += async (sender, e) => {
                Console.WriteLine("connected to server !!!!!!!!!!!!");
                await client.EmitAsync("diagnosisPd", "socket.io clent message");
            };
            client.On("diagnosisEv", async response => {
                Console.WriteLine("receive raw: " + response);
                JArray jobject = JArray.Parse(response.ToString());
                JObject data = (JObject)jobject[0]["data"];
                string fullName = (string)data["FullName"];
                JObject diagClasses = (JObject)data["Classes"];
                Console.WriteLine("diagClass: " + diagClasses);
                JArray Prediction = (JArray)data["Prediction"];
                Console.WriteLine("Pred: " + Prediction);
                DateTime dateTime = DateTime.Parse((string)data["Timestamp"]);
                Console.WriteLine("Date time: " + dateTime);
                double maxPred = (double)Prediction[0];
                int idx = 0;
                for(int i = 1; i < Prediction.Count; i++)
                {
                    if((double)Prediction[i]>maxPred)
                    {
                        maxPred = (double)Prediction[i];
                        idx = i;
                    }
                }
                Console.WriteLine("maxPrd: " + maxPred + " indx: " + idx);
                string foundClass = (string)diagClasses[idx.ToString()];
                Console.WriteLine("foundClass: " + foundClass);
            });
            await client.ConnectAsync();
        }
    }
}
