import "./styles.css";
import * as tf from "@tensorflow/tfjs/";

const model = tf.sequential();

const inputShape = 10;

const numToBinTensor = (num) =>
  tf.tensor(num.toString(2).padStart(inputShape, "0").split("").map(Number));

const fizzbuzzEncoder = (num) => {
  if (num % 15 === 0) {
    return tf.oneHot(3, 4);
  }

  if (num % 5 === 0) {
    return tf.oneHot(2, 4);
  }

  if (num % 3 === 0) {
    return tf.oneHot(1, 4);
  }

  return tf.oneHot(0, 4);
};

const [stackedX, stackedY] = tf.tidy(() => {
  let xs = [];
  let ys = [];

  for (let i = 1; i <= 1000; i++) {
    xs.push(numToBinTensor(i));
    ys.push(fizzbuzzEncoder(i));
  }

  return [tf.stack(xs), tf.stack(ys)];
});

// Add first layer with input shape
model.add(
  tf.layers.dense({
    inputShape: 10,
    units: 64,
    activation: "relu"
  })
);

model.add(
  tf.layers.dense({
    units: 8,
    activation: "relu"
  })
);

model.add(
  tf.layers.dense({
    units: 4,
    activation: "softmax"
  })
);

const learningRate = 0.005;
model.compile({
  optimizer: tf.train.adam(learningRate),
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"]
});

const allCallbacks = {
  // called when training starts
  onTrainBegin: (log) => console.log(log),
  // called when training ends
  onTrainEnd: (log) => console.log(log),
  // called on start of each pass of the training data
  onEpochBegin: (epoch, log) => console.log(epoch, log),
  // called on each successful pass on training data
  // * This is one of my favorite to actually use!
  onEpochEnd: (epoch, log) => console.log(epoch, log),
  // Runs before each batch - perfect for large batch training
  onBatchBegin: (batch, log) => console.log(batch, log),
  // runs after each batch - 32 in this case
  onBatchEnd: (batch, log) => console.log(batch, log)
};

console.log("about to call the fit function");

model
  .fit(stackedX, stackedY, {
    epochs: 100,
    shuffle: true,
    batchSize: 32,
    callbacks: allCallbacks
  })
  .then((history) => model.save("downloads://fizzbuzz-model"));

console.log("Training done!");

document.getElementById("app").innerHTML = `
<h1>${tf.version.tfjs}</h1>
`;
