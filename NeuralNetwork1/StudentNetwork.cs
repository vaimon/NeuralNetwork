using System;
using System.Collections.Generic;

namespace NeuralNetwork1
{
    class Neuron
    {
        public int id;
        public double input;
        public double output;
        public double error;

        public static Func<double, double> activationFunction;

        public Neuron(int id)
        {
            this.id = id;
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

            Random random = new Random();

            biasNeuron = new Neuron(0);
            int id = 1;
            
            layers = new List<Neuron[]>();
            for (int i = 0; i < structure.Length; i++)
            {
                layers.Add(new Neuron[structure[i]]);
                for (int j = 0; j < structure[i]; j++)
                {
                    layers[i][j] = new Neuron(id);
                    // Добавляем bias
                    weights.Add(new Bond(0,id),random.NextDouble() * 2 - 1);
                    // А тут накидываем рандомные веса для предыдущих слоёв
                    if (i > 0)
                    {
                        for (int k = 0; k < layers[i-1].Length; k++)
                        {
                            weights.Add(new Bond(layers[i-1][k].id,id),random.NextDouble() * 2 - 1);
                        }
                    }
                    
                    id++;
                }
            }
            
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            throw new NotImplementedException();
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            throw new NotImplementedException();
        }

        protected override double[] Compute(double[] input)
        {
            throw new NotImplementedException();
        }
    }
}