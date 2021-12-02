using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork1
{
    class Neuron
    {
        public static Func<double, double> activationFunction;
        public static Func<double, double> activationFunctionDerivative;
        
        public int id;
        public double input;
        public int layer;
        public double error;

        public double Output
        {
            get
            {
                if (layer == -1)
                {
                    return 1;
                }

                if (layer == 0)
                {
                    return input;
                }

                return activationFunction(input);
            }
        }

        public Neuron(int id, int layer)
        {
            this.id = id;
            this.layer = layer;
            this.error = 0;
        }
    }

    struct Bond
    {
        public int idFrom;
        public int idTo;

        public Bond(int idFrom, int idTo)
        {
            this.idFrom = idFrom;
            this.idTo = idTo;
        }
    }
    
    public class StudentNetwork : BaseNetwork
    {
        private const int hiddenLayersCount = 2;
        private const double learningRate = 0.01;

        private Dictionary<Bond, double> weights;

        private Neuron biasNeuron;
        private List<Neuron[]> layers;

        private Func<double[],double[], double> lossFunction;
        private Func<double,double, double> lossFunctionDerivative;
        
        public StudentNetwork(int[] structure)
        {
            if (structure.Length != 4)
            {
                throw new ArgumentException("Мы же договаривались о 2 скрытых слоях...");
            }

            lossFunction = (output, aim) =>
            {
                double res = 0;
                for (int i = 0; i < aim.Length; i++)
                {
                    res += Math.Pow(aim[i] - output[i], 2);
                }

                return res * 0.5; // / n to become MSE
            };

            lossFunctionDerivative = (output, aim) => output - aim;

            Neuron.activationFunction = s => 1.0 / (1.0 + Math.Exp(-s));
            Neuron.activationFunctionDerivative = s => s * (1 - s);
            
            Random random = new Random();

            biasNeuron = new Neuron(0, -1);
            int id = 1;
            
            layers = new List<Neuron[]>();
            weights = new Dictionary<Bond, double>();
            for (int layer = 0; layer < structure.Length; layer++)
            {
                layers.Add(new Neuron[structure[layer]]);
                for (int i = 0; i < structure[layer]; i++)
                {
                    layers[layer][i] = new Neuron(id,layer);
                    // Добавляем bias
                    weights.Add(new Bond(0,id),random.NextDouble() * 2 - 1);
                    // А тут накидываем рандомные веса для предыдущих слоёв
                    if (layer > 0)
                    {
                        for (int k = 0; k < layers[layer-1].Length; k++)
                        {
                            weights.Add(new Bond(layers[layer-1][k].id,id),random.NextDouble() * 2 - 1);
                        }
                    }
                    
                    id++;
                }
            }
            
        }
        

        public void forwardPropagation(double[] input)
        {
            if (input.Length != layers[0].Length)
            {
                throw new ArgumentException("Вы мне подсунули какой-то странный входной массив.");
            }
            // Копируем наши данные от сенсоров сразу в их output
            for (int i = 0; i < layers[0].Length; i++)
            {
                layers[0][i].input = input[i];
            }

            for (int layer = 1; layer < layers.Count; layer++)
            {
                for (int neuron = 0; neuron < layers[layer].Length; neuron++)
                {
                    // Считаем скалярное произведение от предыдущих нейрончиков
                    double scalar = 0;
                    foreach (var prevNeuron in layers[layer - 1])
                    {
                        scalar += prevNeuron.Output * weights[new Bond(prevNeuron.id, layers[layer][neuron].id)];
                    }
                    // Добавялем к этому произведению bias
                    scalar += biasNeuron.Output * weights[new Bond(biasNeuron.id, layers[layer][neuron].id)];
                    // Получили наш вход
                    layers[layer][neuron].input = scalar;
                }
            }
        }

        public void backwardPropagation(Sample sample)
        {
            var aim = sample.outputVector;
            // Для выходного слоя применяем производную лосс-функции
            for (var i = 0; i < layers.Last().Length; i++)
            {
                layers.Last()[i].error = lossFunctionDerivative(layers.Last()[i].Output, aim[i]);
            }

            for (int layer = layers.Count - 1; layer >= 1; layer--)
            {
                foreach (var neuron in layers[layer])
                {
                    // Применяем производную функции активации
                    neuron.error *= Neuron.activationFunctionDerivative(neuron.Output);

                    // Считаем страшную сумму ошибок для предыдущего слоя и меняем веса
                    foreach (var prevNeuron in layers[layer - 1])
                    {
                        prevNeuron.error += neuron.error * weights[new Bond(prevNeuron.id, neuron.id)];
                        weights[new Bond(prevNeuron.id, neuron.id)] += learningRate * neuron.error * prevNeuron.Output;
                    }
                    
                    // Нельзя забывать про малыша bias!!!
                    biasNeuron.error += neuron.error * weights[new Bond(biasNeuron.id, neuron.id)];
                    weights[new Bond(biasNeuron.id, neuron.id)] += learningRate * neuron.error * biasNeuron.Output;
                    
                    // Мы прогнали ошибку дальше, откатываемся к изначальному виду
                    neuron.error = 0;
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int cnt = 0;
            while (true)
            {
                cnt++;
                forwardPropagation(sample.input);
                sample.ProcessPrediction(layers.Last().Select(n => n.Output).ToArray());
                if (sample.EstimatedError() <= acceptableError || cnt > 50)
                {
                    return cnt;
                }
                backwardPropagation(sample);
            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            var start = DateTime.Now;
            int totalSamplesCount = epochsCount * samplesSet.Count;
            int processedSamplesCount = 0;
            double accuracy = 0.0;
            for (int i = 0; i < epochsCount; i++)
            {
                int rightClassified = 0;
                foreach (var sample in samplesSet.samples)
                {
                    if (Train(sample, acceptableError, parallel) == 0)
                    {
                        rightClassified++;
                    }

                    processedSamplesCount++;
                }
                OnTrainProgress(1.0 * processedSamplesCount / totalSamplesCount, accuracy,DateTime.Now - start);
                accuracy = rightClassified * 1.0 / samplesSet.Count;
            }
            return accuracy;
        }

        protected override double[] Compute(double[] input)
        {
            if (input.Length != layers[0].Length)
            {
                throw new ArgumentException("У вас тут данных многовато...");
            }

            forwardPropagation(input);
            return layers.Last().Select(n => n.Output).ToArray();
        }
    }
}