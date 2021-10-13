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

using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;
using Harz_Roller.Data;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;

namespace Harz_Roller.Models
{
    class ModelMaker
    {
        private string dataDir;

        private string dataFile;
        private string convertedDataFile;

        Data.DataMaker dataMaker;

        Model model;
        NDArray x_train, y_train, x_test, y_test;
        float learning_rate = 0.1f;
        int display_step = 100;
        int batch_size = 50;
        int training_steps = 1000;
        int num_classes = 1;
        int num_features = 2048 * 6;

        float accuracy;
        IDatasetV2 train_data;

        public ModelMaker(DataMaker dataMaker)
        {
            this.dataDir = dataMaker.dataDir;

            this.dataFile = dataMaker.dataFile;
            this.convertedDataFile = dataMaker.convertedDataFile;
            this.dataMaker = dataMaker;
            //this.model = model;
        }

        public bool Run()
        {
            tf.enable_eager_execution();

            //PrepareData();

            // Build neural network model.
            var neural_net = new NeuralNet(new NeuralNetArgs
            {
                NumClasses = num_classes,
                NeuronOfHidden1 = 128,
                Activation1 = keras.activations.Relu,
                NeuronOfHidden2 = 256,
                Activation2 = keras.activations.Relu
            });

            // Cross-Entropy Loss.
            // Note that this will apply 'softmax' to the logits.
            Func<Tensor, Tensor, Tensor> cross_entropy_loss = (x, y) =>
            {
                // Convert labels to int 64 for tf cross-entropy function.
                y = tf.cast(y, tf.int64);
                // Apply softmax to logits and compute cross-entropy.
                var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
                // Average loss across the batch.
                return tf.reduce_mean(loss);
            };

            // Accuracy metric.
            Func<Tensor, Tensor, Tensor> accuracy = (y_pred, y_true) =>
            {
                // Predicted class is the index of highest score in prediction vector (i.e. argmax).
                var correct_prediction = tf.equal(tf.math.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
            };

            // Stochastic gradient descent optimizer.
            var optimizer = keras.optimizers.SGD(learning_rate);

            // Optimization process.
            Action<Tensor, Tensor> run_optimization = (x, y) =>
            {
                // Wrap computation inside a GradientTape for automatic differentiation.
                using var g = tf.GradientTape();
                // Forward pass.
                var pred = neural_net.Apply(x, training: true);
                var loss = cross_entropy_loss(pred, y);

                // Compute gradients.
                var gradients = g.gradient(loss, neural_net.trainable_variables);

                // Update W and b following gradients.
                optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables.Select(x => x as ResourceVariable)));
            };


            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data))
            {
                // Run the optimization to update W and b values.
                run_optimization(batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = neural_net.Apply(batch_x, training: true);
                    var loss = cross_entropy_loss(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // Test model on validation set.
            {
                var pred = neural_net.Apply(x_test, training: false);
                this.accuracy = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {this.accuracy}");
            }

            return this.accuracy > 0.92f;
        }
        /*public bool Run()
        {
            tf.enable_eager_execution();

            

            // Build neural network model.
            var neural_net = new NeuralNet(new NeuralNetArgs
            {
                NumClasses = num_classes,
                NeuronOfHidden1 = num_features,
                Activation1 = keras.activations.Relu,
                NeuronOfHidden2 = num_features /3,
                Activation2 = keras.activations.Relu,
                NeuronOfHidden3 = num_features /6 ,
                Activation3 = keras.activations.Relu
            });

            // Cross-Entropy Loss.
            // Note that this will apply 'softmax' to the logits.
            Func<Tensor, Tensor, Tensor> cross_entropy_loss = (x, y) =>
            {
                // Convert labels to int 64 for tf cross-entropy function.
                y = tf.cast(y, tf.int64);
                // Apply softmax to logits and compute cross-entropy.
                var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
                // Average loss across the batch.
                return tf.reduce_mean(loss);
            };

            // Accuracy metric.
            Func<Tensor, Tensor, Tensor> accuracy = (y_pred, y_true) =>
            {
                // Predicted class is the index of highest score in prediction vector (i.e. argmax).
                var correct_prediction = tf.equal(tf.math.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
            };

            // Stochastic gradient descent optimizer.
            var optimizer = keras.optimizers.SGD(learning_rate);

            // Optimization process.
            Action<Tensor, Tensor> run_optimization = (x, y) =>
            {
                // Wrap computation inside a GradientTape for automatic differentiation.
                using var g = tf.GradientTape();
                // Forward pass.
                var pred = neural_net.Apply(x, training: true);
                var loss = cross_entropy_loss(pred, y);

                // Compute gradients.
                var gradients = g.gradient(loss, neural_net.trainable_variables);

                // Update W and b following gradients.
                optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables.Select(x => x as ResourceVariable)));
            };


            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data))
            {
                // Run the optimization to update W and b values.
                run_optimization(batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = neural_net.Apply(batch_x, training: true);
                    var loss = cross_entropy_loss(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // Test model on validation set.
            {
                var pred = neural_net.Apply(x_test, training: false);
                this.accuracy = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {this.accuracy}");
            }

            return this.accuracy > 0.92f;
        }*/



        public void PrepareData(bool forceCSVRebuild = false, int trainSize = 1024, int testSize = 256)
        {
            
            string help = keras.utils.get_file("Data.csv", convertedDataFile);
            if (forceCSVRebuild || !File.Exists(convertedDataFile))
            {
                if (forceCSVRebuild || File.Exists(dataFile)) dataMaker.stitchCSV();
                dataMaker.convertCSV();
            }

            


            Data.DataMaker.TrainingSet training = dataMaker.generateTrainingSet(trainSize, new Random());
            x_train = training.inputField;
            y_train = training.outputField;
            Data.DataMaker.TrainingSet testing = dataMaker.generateTrainingSet(testSize, new Random());
            x_test = testing.inputField;
            y_test = testing.outputField;
            

            //x_train = np.array(getInputField(0, 126));
            //for(; x_train.size < 4096; x_train.)

            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));
            //(y_train, y_test) = (y_train.reshape((-1, num_classes)), y_test.reshape((-1, num_classes)));

            Console.WriteLine("x: {0}   {1}  y: {2}   {3}", x_test.GetShape(), x_train.GetShape(), y_test.GetShape(), y_train.GetShape());

            train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
            train_data = train_data
                .batch(batch_size)
                .take(training_steps);
        }


    }
    public class NeuralNet : Model
    {
        Layer fc1;
        Layer fc2;
        Layer fc3;
        Layer output;

        public NeuralNet(NeuralNetArgs args) :
            base(args)
        {
            var layers = keras.layers;

            // First fully-connected hidden layer.
            fc1 = layers.Dense(args.NeuronOfHidden1, activation: args.Activation1);

            // Second fully-connected hidden layer.
            fc2 = layers.Dense(args.NeuronOfHidden2, activation: args.Activation2);

            fc3 = layers.Dense(args.NeuronOfHidden3, activation: args.Activation3);

            output = layers.Dense(args.NumClasses);

            StackLayers(fc1, fc2, fc3, output);
        }

        // Set forward pass.
        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            inputs = fc1.Apply(inputs);
            inputs = fc2.Apply(inputs);
            inputs = output.Apply(inputs);
            if (!training.Value)
                inputs = tf.nn.softmax(inputs);
            return inputs;
        }
    }

    /// <summary>
    /// Network parameters.
    /// </summary>
    public class NeuralNetArgs : ModelArgs
    {
        /// <summary>
        /// 1st layer number of neurons.
        /// </summary>
        public int NeuronOfHidden1 { get; set; }
        public Activation Activation1 { get; set; }

        /// <summary>
        /// 2nd layer number of neurons.
        /// </summary>
        public int NeuronOfHidden2 { get; set; }
        public Activation Activation2 { get; set; }

        /// <summary>
        /// 3nd layer number of neurons.
        /// </summary>
        public int NeuronOfHidden3 { get; set; }
        public Activation Activation3 { get; set; }

        public int NumClasses { get; set; }
    }
}
