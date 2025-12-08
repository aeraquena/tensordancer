import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export const JOINTS = {
  NOSE: 0,
  /*  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,*/
  LEFT_SHOULDER: 1,
  RIGHT_SHOULDER: 2,
  LEFT_ELBOW: 3,
  RIGHT_ELBOW: 4,
  LEFT_WRIST: 5,
  RIGHT_WRIST: 6,
  LEFT_PINKY: 7,
  RIGHT_PINKY: 8,
  LEFT_INDEX: 9,
  RIGHT_INDEX: 10,
  LEFT_THUMB: 11,
  RIGHT_THUMB: 12,
  LEFT_HIP: 13,
  RIGHT_HIP: 14,
  LEFT_KNEE: 15,
  RIGHT_KNEE: 16,
  LEFT_ANKLE: 17,
  RIGHT_ANKLE: 18,
  LEFT_HEEL: 19,
  RIGHT_HEEL: 20,
  LEFT_FOOT_INDEX: 21,
  RIGHT_FOOT_INDEX: 22,
};

/*export const POSE_CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [27, 29],
  [27, 31],
  [29, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [28, 32],
  [30, 32],
];*/

// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
export const createPoseLandmarker = async (
  poseLandmarker: PoseLandmarker | undefined,
  runningMode: "IMAGE" | "VIDEO"
) => {
  const demosSection = document.getElementById("demos") as HTMLElement;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`,
      // https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
      delegate: "GPU",
    },
    runningMode: runningMode,
    numPoses: 2,
  });
  demosSection.classList.remove("invisible");
  return poseLandmarker;
};

// Removes joints 1-10 (face, except for nose) from list of joints
export const removeFaceFromJoints = (
  joints: NormalizedLandmark[]
): NormalizedLandmark[] => {
  let bodyJoints = [];
  for (let i = 0; i < joints.length; i++) {
    if (i === 0 || i > 10) {
      bodyJoints.push(joints[i]);
    }
  }
  return bodyJoints;
};
