using Accord.Imaging;
using Accord.IO;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace TrainImage
{
    class Program
    {
        public static void outputIt(String msg)
        {
            System.Console.WriteLine(string.Format("[{0}]: {1}", DateTime.Now.ToString("o"), msg));
        }
        public static void logIt(String msg)
        {
            System.Diagnostics.Trace.WriteLine(msg);
        }
        static void Main(string[] args)
        {
            System.Configuration.Install.InstallContext _args = new System.Configuration.Install.InstallContext(null, args);
            if (_args.IsParameterTrue("debug"))
            {
                System.Console.WriteLine("wait for debugger, press any key.");
                System.Console.ReadKey();
            }
            if (_args.Parameters.ContainsKey("test"))
            {
                test(_args.Parameters);
            }
            else
            {
                if (_args.Parameters.ContainsKey("xml"))
                {
                    train(_args.Parameters);
                }
            }
        }
        static void test(System.Collections.Specialized.StringDictionary args)
        {
            string bow_file = (args.ContainsKey("bow")) ? args["bow"] : string.Empty;
            string machine_file = (args.ContainsKey("machine")) ? args["machine"] : string.Empty;
            if(System.IO.File.Exists(bow_file) && System.IO.File.Exists(machine_file))
            {
                Serializer.Load(machine_file, out object machine);
                Serializer.Load(bow_file, out object bow);
                string test = args["test"];
                if (System.IO.File.Exists(test))
                {
                    Bitmap b = new Bitmap(test);
                    double[] fs = (bow as ITransform<Bitmap, double[]>).Transform(b);
                    bool o = (machine as Accord.MachineLearning.VectorMachines.SupportVectorMachine<Accord.Statistics.Kernels.Linear>).Decide(fs);
                    double score = (machine as Accord.MachineLearning.VectorMachines.SupportVectorMachine<Accord.Statistics.Kernels.Linear>).Score(fs);
                    double[] prob = (machine as Accord.MachineLearning.VectorMachines.SupportVectorMachine<Accord.Statistics.Kernels.Linear>).Probabilities(fs);
                    outputIt(string.Format("{0}: {1} score={2}, p1={3}, p2={4}",
                        System.IO.Path.GetFileName(test), o, score, prob[0], prob[1]));
                }
                if (System.IO.Directory.Exists(test))
                {
                    foreach (string ss in System.IO.Directory.GetFiles(test))
                    {
                        try
                        {
                            Bitmap b = new Bitmap(ss);
                            double[] fs = (bow as ITransform<Bitmap, double[]>).Transform(b);
                            bool o = (machine as Accord.MachineLearning.VectorMachines.SupportVectorMachine<Accord.Statistics.Kernels.Linear>).Decide(fs);
                            double score = (machine as Accord.MachineLearning.VectorMachines.SupportVectorMachine<Accord.Statistics.Kernels.Linear>).Score(fs);
                            double[] prob = (machine as Accord.MachineLearning.VectorMachines.SupportVectorMachine<Accord.Statistics.Kernels.Linear>).Probabilities(fs);
                            outputIt(string.Format("{0}: {1} score={2}, p1={3}, p2={4}",
                                System.IO.Path.GetFileName(ss), o, score, prob[0], prob[1]));
                        }
                        catch (Exception) { }
                    }
                }
            }
        }
        static XmlDocument loadXml(string filename)
        {
            XmlDocument ret = new XmlDocument();
            try
            {
                ret.Load(filename);
            }
            catch (Exception)
            {
            }
            return ret;
        }
        static void train(System.Collections.Specialized.StringDictionary args)
        {
            try
            {
                XmlDocument doc = loadXml(args["xml"]);
                if (doc.DocumentElement != null)
                {
                    List<Tuple<Bitmap, int>> data = new List<Tuple<Bitmap, int>>();
                    string s = doc.DocumentElement["positive"].InnerText;
                    foreach(string ss in System.IO.Directory.GetFiles(s))
                    {
                        try
                        {
                            Bitmap b = new Bitmap(ss);
                            data.Add(new Tuple<Bitmap, int>(b, 1));
                        }
                        catch (Exception) { }
                    }
                    s = doc.DocumentElement["negative"].InnerText;
                    foreach (string ss in System.IO.Directory.GetFiles(s))
                    {
                        try
                        {
                            Bitmap b = new Bitmap(ss);
                            data.Add(new Tuple<Bitmap, int>(b, -1));
                        }
                        catch (Exception) { }
                    }

                    // create bag-of-word
                    s = doc.DocumentElement["number"].InnerText;
                    int i = 500;
                    if(!Int32.TryParse(s,out i))
                    {
                        i = 500;
                    }
                    var surfBow = BagOfVisualWords.Create(numberOfWords: i);
                    Bitmap[] bmps = new Bitmap[data.Count];
                    for (i = 0; i < data.Count; i++)
                    {
                        bmps[i] = data[i].Item1;
                    }
                    //IBagOfWords<Bitmap> bow = surfBow.Learn(bmps);
                    //double[][] features = (bow as ITransform<Bitmap, double[]>).Transform(bmps);
                    surfBow.Learn(bmps);
                    double[][] features = surfBow.Transform(bmps);
                    int[] labels = new int[data.Count];
                    for (i = 0; i < labels.Length; i++)
                    {
                        labels[i] = data[i].Item2;
                    }
                    s = doc.DocumentElement["complexity"].InnerText;
                    if (!Int32.TryParse(s, out i))
                    {
                        i = 10000;
                    }
                    var teacher = new SequentialMinimalOptimization<Linear>()
                    {
                        Complexity = i // make a hard margin SVM
                    };
                    var svm = teacher.Learn(features, labels);
                    s = doc.DocumentElement["bow_output"].InnerText;
                    Serializer.Save(obj: surfBow, path: s);
                    s = doc.DocumentElement["machine_output"].InnerText;
                    Serializer.Save(obj: svm, path: s);
                }
            }
            catch(Exception ex)
            {
                logIt(ex.Message);
                logIt(ex.StackTrace);
            }
        }
    }
}
