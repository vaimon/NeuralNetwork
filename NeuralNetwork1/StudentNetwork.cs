using System;
using System.Collections.Generic;

namespace NeuralNetwork1
{
    class Neuron
    {
        public static Func<double, double> activationFunction;
        
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

        private Dictionary<Bond, double> weights;

        private Neuron biasNeuron;
        private List<Neuron[]> layers;

        public StudentNetwork(int[] structure)
        {
            if (structure.Length != 4)
            {
                throw new ArgumentException("Мы же договаривались о 2 скрытых слоях...");
            }

            //TODO Activation function
            
            Random random = new Random();

            biasNeuron = new Neuron(0, -1);
            int id = 1;
            
            layers = new List<Neuron[]>();
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
        

        public void forwardPropagation(Sample sample)
        {
            if (sample.input.Length != layers[0].Length)
            {
                throw new ArgumentException("Вы мне подсунули какой-то странный входной массив.");
            }
            for (int i = 0; i < layers[0].Length; i++)
            {
                layers[0][i].input = sample.input[i];
            }

            for (int layer = 1; layer < layers.Count; layer++)
            {
                for (int neuron = 0; neuron < layers[layer].Length; neuron++)
                {
                    double scalar = 0;
                    foreach (var prevNeuron in layers[layer - 1])
                    {
                        scalar += prevNeuron.Output * weights[new Bond(prevNeuron.id, layers[layer][neuron].id)];
                    }

                    layers[layer][neuron].input = scalar;
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            // Это разве не прямой путь к переобучению?...
            throw new NotImplementedException();
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            for (int i = 0; i < epochsCount; i++)
            {
                int rightClassified = 0;
                foreach (var sample in samplesSet.samples)
                {
                    
                    Train(sample, acceptableError, parallel);
                }
            }
            return 0.0;
        }

        protected override double[] Compute(double[] input)
        {
            if (input.Length != layers[0].Length)
            {
                throw new ArgumentException("У вас тут данных многовато...");
            }

            return null;
        }
    }
}