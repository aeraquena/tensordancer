import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import * as tf from "@tensorflow/tfjs";

// Store full pose data (33 landmarks × 2 coords = 66 values)
export type PoseDatum = {
  person1Pose: number[]; // 66D: flatten x,y for all 33 landmarks
  person2Pose: number[]; // 66D: flatten x,y for all 33 landmarks
};

// Normalization / tensor metadata returned by convertToTensor
type NormalizationData = {
  inputs: any;
  labels: any;
  inputMax: any;
  inputMin: any;
  labelMax: any;
  labelMin: any;
};

// Helper: Flatten 33 landmarks into 66D array [x0,y0,x1,y1,...,x32,y32]
export function flattenPose(landmarks: NormalizedLandmark[]): number[] {
  const pose: number[] = [];
  for (let i = 0; i < 33; i++) {
    pose.push(landmarks[i]?.x ?? 0);
    pose.push(landmarks[i]?.y ?? 0);
  }
  return pose;
}

// Helper: Unflatten 66D array into 33 NormalizedLandmarks {x: number, y: number}
export function unflattenPose(pose: number[]): NormalizedLandmark[] {
  const landmarks: NormalizedLandmark[] = [];
  for (let i = 0; i < 33; i++) {
    landmarks.push({
      x: pose[i * 2] ?? 0,
      y: pose[i * 2 + 1] ?? 0,
      z: 0, // MediaPipe landmarks have z, setting to 0 as default
      visibility: 1, // Optional: set default visibility
    });
  }
  return landmarks;
}

// Define model architecture
// Updated model: 66D input → 66D output
function createModel() {
  // Create a small MLP with a mix of linear and ReLU layers.
  // Input: single scalar (hand X). Output: single scalar (predicted mouse Y).
  const model = tf.sequential();

  // Input: 66D pose (33 landmarks × 2)
  // First hidden layer: expand to a richer representation and apply non-linearity
  model.add(
    tf.layers.dense({
      inputShape: [66],
      units: 128,
      activation: "relu",
      useBias: true,
    })
  );

  // Add more model layers to increase accuracy

  // Second hidden layer: narrower representation
  model.add(tf.layers.dense({ units: 64, activation: "relu", useBias: true }));

  // Third hidden layer: smaller feature set
  model.add(tf.layers.dense({ units: 32, activation: "relu", useBias: true }));

  // Final output layer: linear activation for regression
  // Output: 66D predicted pose
  model.add(
    tf.layers.dense({ units: 66, activation: "linear", useBias: true })
  );

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data: PoseDatum[]): NormalizationData {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.
  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d: PoseDatum) => d.person1Pose);
    const labels = data.map((d: PoseDatum) => d.person2Pose);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 66]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 66]);

    // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

// Train model from data
export async function run(
  data: PoseDatum[]
): Promise<{ model: any; tensorData: NormalizationData } | void> {
  // Create the model
  const model = createModel();

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);

  // Return the trained model and normalization data to callers.
  return { model, tensorData };
}

async function trainModel(model: any, inputs: any, labels: any) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    // adam optimizer as it is quite effective in practice and requires no configuration.
    loss: tf.losses.meanSquaredError,
    // this is a function that will tell the model how well it is doing on learning
    // each of the batches (data subsets) that it is shown. Here we use
    // meanSquaredError to compare the predictions made by the model with the true values.
    metrics: ["mse"],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    // size of the data subsets that the model will see on each iteration of training.
    // Common batch sizes tend to be in the range 32-512
    epochs,
    // number of times the model is going to look at the entire dataset that you provide it
    shuffle: true,
  });
}

// Predict full 66D pose from input pose
// Normalize a single pose, run predict, un-normalize and return array of output pose
export function predictPose(
  model: any,
  inputPose: number[], // 66D
  normalizationData: NormalizationData
): number[] {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  return tf.tidy(() => {
    // create a normalized tensor for the single input
    const inputTensor = tf.tensor2d([inputPose], [1, 66]);
    const normalized = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    // predict (returns a Tensor)
    const pred = model.predict(normalized) as any;
    // un-normalize prediction
    const unNorm = pred.mul(labelMax.sub(labelMin)).add(labelMin) as any;
    // read single pose (array) value
    return Array.from(unNorm.dataSync()) as number[];
  });
}
