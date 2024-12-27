using System;
using System.IO;
using RabbitMQ.Client;
using System.Text.Json;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Security.Cryptography;

namespace IoTDataProcess
{
    class Program
    {
        static string GenerateSecureRandomString(int length)
        {
            const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            StringBuilder result = new StringBuilder(length);
            byte[] uintBuffer = new byte[sizeof(uint)];

            using (RNGCryptoServiceProvider rng = new RNGCryptoServiceProvider())
            {
                while (length-- > 0)
                {
                    rng.GetBytes(uintBuffer);
                    uint num = BitConverter.ToUInt32(uintBuffer, 0);
                    result.Append(chars[(int)(num % (uint)chars.Length)]);
                }
            }

            return result.ToString();
        }
        async static Task MessageQueue()
        {
            var factory = new ConnectionFactory()
            {
                HostName = "localhost", // Địa chỉ RabbitMQ server
                UserName = "guest",     // Username
                Password = "guest"      // Password
            };
            // Tạo kết nối đến RabbitMQ
            using (var connection = factory.CreateConnection())
            {
                // Tạo kênh kết nối
                using (var channel = connection.CreateModel())
                {
                    // Tạo JSON object

                    int maxThread = 5;
                    List<Task> tasks = new List<Task>();
                    var k = 0;
                    
                    while (true)
                    {
                        for (int i = 0; i < maxThread; i++)
                        {
                            Console.WriteLine("sequence i: " + i.ToString());
                            tasks.Add(Task.Run(() => {
                         
                                //Console.WriteLine("Start send to Queue");
                                string filePath = "D:\\ai-server\\abc.dat";

                                // Đọc dữ liệu từ file thành byte array
                                byte[] fileBytes = File.ReadAllBytes(filePath);

                                var body = new byte[fileBytes.Length];
                                Buffer.BlockCopy(fileBytes, 0, body, 0, fileBytes.Length);
                                var properties = channel.CreateBasicProperties();
                                var file_name = Path.GetFileNameWithoutExtension(filePath);
                                StringBuilder _sb = new StringBuilder();
                                _sb.Append(file_name);
                                _sb.Append("_");
                                _sb.Append(GenerateSecureRandomString(10));
                                properties.Headers = new Dictionary<string, object>();
                                properties.Headers.Add("FullName", _sb.ToString());
                                properties.Headers.Add("SensorInfoId", 15);
                                properties.Headers.Add("LNName", "SPDC1");
                                properties.Headers.Add("LUName", "LU1");
                                properties.Headers.Add("GLUId", 1);
                                properties.Headers.Add("MLUId", 0);
                                properties.Headers.Add("EventId", 1);

                                channel.BasicPublish(exchange: "",
                                     routingKey: "file_queue",
                                     basicProperties: properties,
                                     body: body);

                                  //Console.WriteLine(" [x] Sent file and JSON object");
                            }));
                        
                        }
                        Task.WaitAll(tasks.ToArray());
                        await Task.Delay(4000);
                    }
                }
            }
        }
        async static Task Main(string[] args)
        {
            socketClientService socket = new socketClientService();
            await socket._SocketClient();
            await MessageQueue();
        }
    }
}
